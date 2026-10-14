"""Background indexing jobs for :class:`OctopusIndex`."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


class IndexJob:
    """Handle for one background index operation.

    Jobs deliberately keep their outcome in memory. The host can persist the
    returned report or status in its own job system when it needs cross-process
    durability.
    """

    def __init__(self, operation: str, *, loop: asyncio.AbstractEventLoop) -> None:
        self.id = uuid4().hex
        self.operation = operation
        self.status = "queued"
        self.created_at = datetime.now(timezone.utc)
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.report: Any = None
        self.error: BaseException | None = None
        self._future = loop.create_future()
        self._sources: list[Any] = []
        self.replace = False

    @property
    def sources(self) -> tuple[Any, ...]:
        return tuple(self._sources)

    def add_sources(self, sources: list[Any]) -> None:
        """Merge source requests into a still-queued add job."""
        existing = {_source_key(source) for source in self._sources}
        for source in sources:
            key = _source_key(source)
            if key not in existing:
                self._sources.append(source)
                existing.add(key)

    async def wait(self) -> "IndexJob":
        """Wait for completion and return this job without masking its error."""
        await self._future
        return self

    def result(self):
        """Return the report or raise the operation's original error."""
        if not self._future.done():
            raise RuntimeError(f"Index job '{self.id}' is not complete.")
        if self.status == "cancelled":
            raise RuntimeError(f"Index job '{self.id}' was cancelled.")
        if self.error is not None:
            raise self.error
        return self.report

    def snapshot(self) -> dict[str, Any]:
        return {
            "job_id": self.id,
            "operation": self.operation,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "error": str(self.error) if self.error else None,
            "report": self.report.to_dict() if hasattr(self.report, "to_dict") else self.report,
        }

    def _complete(self) -> None:
        self.completed_at = datetime.now(timezone.utc)
        if not self._future.done():
            self._future.set_result(self)


class IndexJobQueue:
    """One-worker queue owned by one ``OctopusIndex`` instance."""

    def __init__(self, owner, *, max_pending_jobs: int = 1, busy_policy: str = "coalesce"):
        if max_pending_jobs < 1:
            raise ValueError("max_pending_jobs must be at least 1")
        if busy_policy not in {"coalesce", "reject", "wait"}:
            raise ValueError("busy_policy must be coalesce, reject, or wait")
        self.owner = owner
        self.max_pending_jobs = max_pending_jobs
        self.busy_policy = busy_policy
        self._queue: asyncio.Queue[IndexJob] | None = None
        self._worker: asyncio.Task | None = None
        self._pending: dict[str, IndexJob] = {}
        self._jobs: dict[str, IndexJob] = {}

    def submit_add(self, sources: list[Any]) -> IndexJob:
        return self._submit("add", sources=sources)

    def submit_update(self) -> IndexJob:
        return self._submit("update")

    def submit_create(self, *, replace: bool = False) -> IndexJob:
        return self._submit("create", replace=replace)

    def get(self, job_id: str) -> IndexJob | None:
        return self._jobs.get(job_id)

    def _submit(
        self,
        operation: str,
        *,
        sources: list[Any] | None = None,
        replace: bool = False,
    ) -> IndexJob:
        loop = asyncio.get_running_loop()
        source_list = list(sources or [])

        can_coalesce = operation != "create" or self._pending.get(operation, None) is None or self._pending[operation].replace == replace
        if self.busy_policy == "coalesce" and operation in self._pending and can_coalesce:
            job = self._pending[operation]
            if operation == "add":
                job.add_sources(source_list)
            return job

        if (
            self._queue is not None
            and self._queue.qsize() >= self.max_pending_jobs
            and self.busy_policy == "reject"
        ):
            from brown_octopus.errors import IndexBusyError

            raise IndexBusyError(
                f"Indexing queue is full for operation '{operation}'."
            )

        if self._queue is None:
            self._queue = asyncio.Queue()
        job = IndexJob(operation, loop=loop)
        job.replace = replace
        job.add_sources(source_list)
        self._jobs[job.id] = job
        self._pending[operation] = job
        self._queue.put_nowait(job)
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())
        return job

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            try:
                job = await asyncio.wait_for(self._queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                if self._queue.empty():
                    return
                continue

            if self._pending.get(job.operation) is job:
                self._pending.pop(job.operation, None)
            job.status = "running"
            job.started_at = datetime.now(timezone.utc)
            try:
                # The embedding and snapshot stores contain synchronous work.
                # Run the complete lifecycle in one dedicated thread so the
                # host event loop remains responsive.
                job.report = await asyncio.to_thread(self._execute, job)
                job.status = "completed"
            except asyncio.CancelledError:
                job.status = "cancelled"
                raise
            except BaseException as exc:
                job.error = exc
                job.status = "failed"
            finally:
                job._complete()
                self._queue.task_done()

    def _execute(self, job: IndexJob):
        def run():
            if job.operation == "add":
                return asyncio.run(self.owner.add(*job.sources))
            if job.operation == "update":
                return asyncio.run(self.owner.update())
            if job.operation == "create":
                return asyncio.run(self.owner.create(replace=job.replace))
            raise RuntimeError(f"Unknown index operation '{job.operation}'.")

        return run()

    async def close(self) -> None:
        if self._worker is None:
            return
        if self._queue is not None:
            while not self._queue.empty():
                job = self._queue.get_nowait()
                job.status = "cancelled"
                job._complete()
                self._queue.task_done()
        # Do not cancel an active indexing operation: its worker may already
        # be inside a synchronous embedding/database call. Let it finish so
        # the snapshot publication remains atomic and predictable.
        await self._worker
        self._worker = None


def _source_key(source: Any) -> tuple[str, str]:
    source_id = getattr(source, "source_id", None)
    if source_id is not None:
        return ("source_id", str(source_id))
    url = getattr(source, "url", None)
    if url is not None:
        return ("url", str(url))
    return ("object", str(id(source)))
