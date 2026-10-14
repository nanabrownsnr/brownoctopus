import asyncio
import threading

import pytest

from brown_octopus import IndexBusyError, IndexJob, OctopusIndex


def _source(name):
    return object()


@pytest.mark.anyio
async def test_submit_add_runs_off_event_loop_and_coalesces_pending_sources(tmp_path):
    index = OctopusIndex(index_path=tmp_path / "index")
    started = threading.Event()
    release = threading.Event()
    calls = []

    async def fake_add(*sources):
        started.set()
        await asyncio.to_thread(release.wait)
        calls.append(sources)
        return {"sources": len(sources)}

    index.add = fake_add
    first_source = _source("first")
    second_source = _source("second")
    third_source = _source("third")

    first = index.submit_add(first_source)
    assert isinstance(first, IndexJob)
    assert await asyncio.to_thread(started.wait, 2)
    second = index.submit_add(second_source)
    merged = index.submit_add(third_source)

    assert second is merged
    assert second.status == "queued"
    release.set()

    await first.wait()
    await second.wait()

    assert first.status == "completed"
    assert second.status == "completed"
    assert calls == [
        (first_source,),
        (second_source, third_source),
    ]
    await index.close()


@pytest.mark.anyio
async def test_reject_policy_rejects_only_when_pending_capacity_is_full(tmp_path):
    index = OctopusIndex(
        index_path=tmp_path / "index",
        max_pending_jobs=1,
        busy_policy="reject",
    )
    started = threading.Event()
    release = threading.Event()

    async def fake_add(*sources):
        started.set()
        await asyncio.to_thread(release.wait)
        return None

    index.add = fake_add
    first = index.submit_add(_source("first"))
    assert await asyncio.to_thread(started.wait, 2)

    # One pending job is allowed while the first job is active.
    second = index.submit_add(_source("second"))
    with pytest.raises(IndexBusyError):
        index.submit_add(_source("third"))

    release.set()
    await first.wait()
    await second.wait()
    await index.close()


@pytest.mark.anyio
async def test_failed_job_preserves_original_exception_and_status(tmp_path):
    index = OctopusIndex(index_path=tmp_path / "index")

    async def fake_update():
        raise ValueError("synthetic update failure")

    index.update = fake_update
    job = index.submit_update()

    await job.wait()

    assert job.status == "failed"
    assert "synthetic update failure" in job.snapshot()["error"]
    with pytest.raises(ValueError, match="synthetic update failure"):
        job.result()
    await index.close()


@pytest.mark.anyio
async def test_submit_create_dispatches_replace_flag(tmp_path):
    index = OctopusIndex(index_path=tmp_path / "index")
    received = []

    async def fake_create(*, replace=False):
        received.append(replace)
        return "created"

    index.create = fake_create
    job = index.submit_create(replace=True)

    await job.wait()

    assert job.status == "completed"
    assert job.result() == "created"
    assert received == [True]
    await index.close()


@pytest.mark.anyio
async def test_create_jobs_with_different_replace_flags_do_not_coalesce(tmp_path):
    index = OctopusIndex(index_path=tmp_path / "index")
    started = threading.Event()
    release = threading.Event()
    received = []

    async def fake_create(*, replace=False):
        started.set()
        await asyncio.to_thread(release.wait)
        received.append(replace)
        return replace

    index.create = fake_create
    active = index.submit_create(replace=False)
    assert await asyncio.to_thread(started.wait, 2)
    pending_false = index.submit_create(replace=False)
    pending_true = index.submit_create(replace=True)

    assert pending_false is not pending_true
    release.set()
    await active.wait()
    await pending_false.wait()
    await pending_true.wait()

    assert received == [False, False, True]
    await index.close()


@pytest.mark.anyio
async def test_close_cancels_pending_jobs_and_drains_active_result(tmp_path):
    index = OctopusIndex(index_path=tmp_path / "index")
    started = threading.Event()
    release = threading.Event()

    async def fake_update():
        started.set()
        await asyncio.to_thread(release.wait)
        return "updated"

    index.update = fake_update
    active = index.submit_update()
    assert await asyncio.to_thread(started.wait, 2)
    pending = index.submit_update()

    # The second update is coalesced with the first only if it is still
    # pending; here the first is active, so it remains a separate job.
    assert pending is not active

    release.set()
    await index.close()

    assert pending.status == "cancelled"
    await active.wait()
    assert active.status == "completed"


@pytest.mark.anyio
async def test_background_job_does_not_block_host_event_loop(tmp_path):
    index = OctopusIndex(index_path=tmp_path / "index")
    started = threading.Event()
    release = threading.Event()
    ticks = 0

    async def fake_update():
        started.set()
        await asyncio.to_thread(release.wait)
        return "updated"

    index.update = fake_update
    job = index.submit_update()
    assert await asyncio.to_thread(started.wait, 2)

    for _ in range(5):
        ticks += 1
        await asyncio.sleep(0)

    release.set()
    await job.wait()
    assert ticks == 5
    assert job.status == "completed"
    await index.close()
