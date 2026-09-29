from pathlib import Path
from collections.abc import Iterable
from contextlib import contextmanager
from threading import Condition, RLock
import json
import hashlib
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version

from brown_octopus.analyzer import initialize_analyzer
from brown_octopus.analyzer import DeterministicIntentAnalyzer
from brown_octopus.config import OctopusConfig
from brown_octopus.context_manager import ActiveCapabilityContext
from brown_octopus.contracts import CapabilitySource, RetrievalResult, SessionStore
from brown_octopus.index_store import (
    load_embeddings,
    load_tools,
    save_embeddings,
    load_metadata,
    save_metadata,
    save_tools,
    save_snapshot_atomic,
)
# These names remain module-level compatibility seams for legacy bootstrap,
# scripts, and integrations that monkeypatch the historical retrieval path.
from brown_octopus.mcp_discovery import discover_universe
from brown_octopus.observability import configure_logging
from brown_octopus.pipeline import CapabilityPipeline, process_turn
from brown_octopus.retriever import (
    build_embeddings,
    QwenCandidateRetriever,
    initialize_retriever,
    load_index,
    refresh_index,
    retrieve_tools,
)
from brown_octopus.strategies import BoundedMaxGapSelectionStrategy
from brown_octopus.session_store import InMemorySessionStore
from brown_octopus.sources import load_mcp_urls
from brown_octopus.sources import (
    CapabilityDiscoveryResult,
    CapabilityUpdateReport,
    LocalMcpCatalogSource,
)
from brown_octopus.errors import (
    CapabilityUpdateError,
    IncompatibleIndexError,
    IndexLoadError,
    InitializationError,
    MissingIndexError,
    ModelInitializationError,
)
from brown_octopus.tool_registry import capability_id, get_tools, set_tools


class SnapshotReadWriteLock:
    """Permit concurrent retrieval snapshots and a short exclusive swap."""

    def __init__(self) -> None:
        self._condition = Condition(RLock())
        self._readers = 0
        self._writer = False

    @contextmanager
    def read(self):
        with self._condition:
            while self._writer:
                self._condition.wait()
            self._readers += 1
        try:
            yield
        finally:
            with self._condition:
                self._readers -= 1
                if self._readers == 0:
                    self._condition.notify_all()

    @contextmanager
    def write(self):
        with self._condition:
            while self._writer or self._readers:
                self._condition.wait()
            self._writer = True
        try:
            yield
        finally:
            with self._condition:
                self._writer = False
                self._condition.notify_all()


class Octopus:
    def __init__(
        self,
        index_path: str | Path = "data/indexes/default",
        catalog_path: str | Path = "data/mcps.json",
        *,
        config: OctopusConfig | None = None,
        capability_source: CapabilitySource | None = None,
        session_store: SessionStore | None = None,
    ):
        self.config = config or OctopusConfig.from_env()
        configure_logging(self.config.log_level)
        self.index_path = Path(index_path) if index_path != "data/indexes/default" else self.config.index_path
        self.catalog_path = Path(catalog_path) if catalog_path != "data/mcps.json" else self.config.catalog_path
        self.capability_source = capability_source or LocalMcpCatalogSource(self.catalog_path)
        self._snapshot_lock = SnapshotReadWriteLock()

        self.analyzer = DeterministicIntentAnalyzer()
        self.candidate_retriever = None
        self.selection_strategy = None
        self.pipeline = None
        self.sessions = session_store or InMemorySessionStore()
        # Runtime-only fallback definitions for injected pipelines. The
        # canonical production universe remains the shared tool registry.
        self._runtime_tool_definitions: dict[str, dict] = {}

    @property
    def active_tools(self) -> dict[str, int]:
        """Compatibility view of the default session's active state."""
        return self.sessions.get("default").active_state.copy()

    @active_tools.setter
    def active_tools(self, value: dict[str, int]) -> None:
        """Retain assignment compatibility without runtime state duplication."""
        self.sessions.mutate(
            "default",
            lambda state: setattr(state, "active_state", value.copy()),
        )

    @property
    def turn(self) -> int:
        """Compatibility view of the default session's turn counter."""
        return self.sessions.get("default").turn

    @turn.setter
    def turn(self, value: int) -> None:
        self.sessions.mutate("default", lambda state: setattr(state, "turn", value))

    async def initialize(self) -> list[dict]:
        try:
            initialize_analyzer()
            initialize_retriever()
        except Exception as exc:
            raise ModelInitializationError(
                "Brown Octopus could not load its required runtime models. "
                "Run:\n\n    brown-octopus setup-models\n\n"
                "Then retry initialization. No model downloads are performed "
                "automatically by initialize()."
            ) from exc

        tools_path = self.index_path / "tools.json"
        embeddings_path = self.index_path / "embeddings.pt"
        metadata_path = self.index_path / "metadata.json"

        has_legacy_files = (
            tools_path.exists()
            and embeddings_path.exists()
            and metadata_path.exists()
        )
        has_published_snapshot = (self.index_path / "current.json").exists()
        if not has_legacy_files and not has_published_snapshot:
            raise MissingIndexError(
                f"Brown Octopus index is incomplete at '{self.index_path}'. "
                "Expected tools.json, embeddings.pt, and metadata.json. "
                "Run `await octopus.update()` with a working CapabilitySource "
                "or provide a valid persisted index."
            )

        try:
            metadata = load_metadata(self.index_path)
        except Exception as exc:
            raise IndexLoadError(
                f"Brown Octopus could not load the index at '{self.index_path}'. "
                "Check that the persisted snapshot is readable and complete."
            ) from exc

        if metadata.get("version") != 1:
            raise IncompatibleIndexError(
                "Brown Octopus index version is unsupported: "
                f"{metadata.get('version')}. Rebuild it with this package version."
            )

        try:
            tools = load_tools(self.index_path)
            embeddings = load_embeddings(self.index_path)
        except Exception as exc:
            raise IndexLoadError(
                f"Brown Octopus could not load the index at '{self.index_path}'. "
                "Check that the persisted snapshot is readable and complete."
            ) from exc

        with self._snapshot_lock.write():
            set_tools(tools)
            load_index(tools=tools, embeddings=embeddings)

        self._build_pipeline()

        return tools

    def _process_session(
        self,
        query: str,
        session_id: str,
        allowed_mcp_urls: Iterable[str] | None = None,
    ) -> dict:
        scoped_urls = (
            tuple(allowed_mcp_urls)
            if allowed_mcp_urls is not None
            else None
        )

        def operation(state):
            state.turn += 1
            pipeline_context = getattr(self.pipeline, "context_manager", None)

            def tool_lookup(ids):
                requested = list(ids)
                registered = {
                    capability_id(tool): tool for tool in get_tools(requested)
                }
                for identity in requested:
                    if identity not in registered:
                        tool = self._runtime_tool_definitions.get(identity)
                        if tool is not None:
                            registered[identity] = tool
                return [registered[identity] for identity in requested if identity in registered]

            if pipeline_context is not None:
                tool_lookup = getattr(pipeline_context, "tool_lookup", tool_lookup)
            runtime_context = ActiveCapabilityContext(
                tool_lookup,
                ttl=self.config.ttl,
                active_cap=self.config.active_cap,
            )
            runtime_context.active_state = dict(state.active_state)
            with self._snapshot_lock.read():
                if self.pipeline is not None:
                    result = self.pipeline.process(
                        query,
                        state.turn,
                        context_manager=runtime_context,
                        session_id=session_id,
                        allowed_mcp_urls=scoped_urls,
                    )
                else:
                    if scoped_urls is None:
                        result = process_turn(
                            query=query,
                            active_tools=runtime_context.active_state,
                            turn=state.turn,
                        )
                    else:
                        result = process_turn(
                            query=query,
                            active_tools=runtime_context.active_state,
                            turn=state.turn,
                            allowed_mcp_urls=scoped_urls,
                        )
                    runtime_context.active_state = result["active_state"].copy()
                    result["active_tools"] = result["tools"]
                    result["tool_ids"] = [capability_id(tool) for tool in result["tools"]]
                    result["strategy"] = "brown_octopus_v3"
            self._runtime_tool_definitions.update(
                {
                    capability_id(tool): tool
                    for tool in result.get("retrieved_tools", [])
                }
            )
            state.active_state = dict(runtime_context.active_state)
            result["session_id"] = session_id
            result["turn"] = state.turn
            result["session"] = {
                "session_id": session_id,
                "turn": state.turn,
                "retrieved_count": len(result["retrieved_tools"]),
                "active_count": len(result["active_tools"]),
                "strategy": result["strategy"],
                "allowed_mcp_urls": (
                    list(scoped_urls)
                    if scoped_urls is not None
                    else None
                ),
                "timing_ms": result.get("timing_ms", {}),
            }
            return result

        return self.sessions.mutate(session_id, operation)

    def retrieve(
        self,
        query: str,
        session_id: str = "default",
        allowed_mcp_urls: Iterable[str] | None = None,
    ) -> list[dict]:
        return self._process_session(
            query,
            session_id,
            allowed_mcp_urls,
        )["active_tools"]

    def retrieve_result(
        self,
        query: str,
        session_id: str = "default",
        allowed_mcp_urls: Iterable[str] | None = None,
    ) -> RetrievalResult:
        """Return a stable, harness-neutral retrieval response."""
        result = self._process_session(
            query,
            session_id,
            allowed_mcp_urls,
        )
        tools = result["active_tools"]
        retrieved_tools = result["retrieved_tools"]
        return RetrievalResult(
            tool_ids=[capability_id(tool) for tool in tools],
            tools=tools,
            metadata={
                **result["session"],
                "retrieved_tools": retrieved_tools,
                "active_tools": tools,
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "tool_representation": "name + description",
            },
            retrieved_tools=retrieved_tools,
            session_id=session_id,
            turn=result["turn"],
        )

    def process(
        self,
        query: str,
        session_id: str = "default",
        allowed_mcp_urls: Iterable[str] | None = None,
    ) -> dict:
        return self._process_session(
            query,
            session_id,
            allowed_mcp_urls,
        )

    def reset(self) -> None:
        self.reset_session("default")

    def reset_session(self, session_id: str) -> None:
        self.sessions.reset(session_id)

    def delete_session(self, session_id: str) -> None:
        self.sessions.delete(session_id)

    def get_session(self, session_id: str = "default") -> dict | None:
        return self.sessions.snapshot(session_id)

    def _build_pipeline(self) -> None:
        self.config.validate()
        candidate_retriever = QwenCandidateRetriever()
        self.candidate_retriever = candidate_retriever
        selector = BoundedMaxGapSelectionStrategy()
        self.selection_strategy = selector
        self.pipeline = CapabilityPipeline(
            self.analyzer,
            candidate_retriever,
            selector,
        )

    @staticmethod
    def _tool_signature(tool: dict) -> str:
        relevant = {
            key: tool.get(key)
            for key in (
                "capability_id",
                "id",
                "name",
                "description",
                "input_schema",
                "parameters",
                "mcp_url",
                "tool_name",
            )
        }
        return json.dumps(relevant, sort_keys=True, default=str)

    @staticmethod
    def _source_key(tool: dict) -> str | None:
        return (
            tool.get("source_id")
            or tool.get("mcp_url")
            or tool.get("mcp_name")
        )

    @staticmethod
    def _package_version() -> str:
        try:
            return package_version("brown-octopus")
        except PackageNotFoundError:
            return "unknown"

    @classmethod
    def _tool_universe_fingerprint(cls, tools: list[dict]) -> str:
        signatures = sorted(cls._tool_signature(tool) for tool in tools)
        return hashlib.sha256("\n".join(signatures).encode()).hexdigest()

    async def _discover_snapshot(self) -> CapabilityDiscoveryResult:
        try:
            discovered = await self.capability_source.discover()
        except Exception as exc:
            raise CapabilityUpdateError(
                "Brown Octopus capability discovery failed before a new "
                "snapshot could be prepared. The existing index was preserved."
            ) from exc
        if isinstance(discovered, CapabilityDiscoveryResult):
            return discovered
        if isinstance(discovered, list):
            return CapabilityDiscoveryResult(tools=discovered)
        raise CapabilityUpdateError(
            "CapabilitySource.discover() must return a list of tool definitions "
            "or CapabilityDiscoveryResult."
        )

    def _merge_snapshot_tools(
        self,
        discovered: CapabilityDiscoveryResult,
    ) -> tuple[list[dict], CapabilityUpdateReport]:
        try:
            previous = load_tools(self.index_path)
        except Exception:
            previous = []

        new_by_id = {capability_id(tool): tool for tool in discovered.tools}
        previous_by_id = {capability_id(tool): tool for tool in previous}
        failed_sources = set(discovered.failed_sources)

        merged = dict(new_by_id)
        for identity, tool in previous_by_id.items():
            if identity in merged:
                continue
            if (
                not discovered.authoritative
                or self._source_key(tool) in failed_sources
            ):
                merged[identity] = tool

        added = sorted(set(merged) - set(previous_by_id))
        removed = sorted(set(previous_by_id) - set(merged))
        changed = sorted(
            identity
            for identity in set(merged) & set(previous_by_id)
            if self._tool_signature(merged[identity])
            != self._tool_signature(previous_by_id[identity])
        )
        tools = list(merged.values())
        return tools, CapabilityUpdateReport(
            added=added,
            changed=changed,
            removed=removed,
            failed_sources=dict(discovered.failed_sources),
            tool_count=len(tools),
            authoritative=discovered.authoritative,
        )

    async def update(self) -> CapabilityUpdateReport:
        """Prepare and atomically install a host-requested capability snapshot."""
        try:
            initialize_analyzer()
            initialize_retriever()
            discovered = await self._discover_snapshot()
            tools, report = self._merge_snapshot_tools(discovered)
            embeddings = build_embeddings(tools) if tools else None
            metadata = {
                "version": 1,
                "index_format_version": 1,
                "brown_octopus_version": self._package_version(),
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "tool_count": len(tools),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "tool_universe_fingerprint": self._tool_universe_fingerprint(tools),
            }
            save_snapshot_atomic(self.index_path, tools, embeddings, metadata)
        except CapabilityUpdateError:
            raise
        except Exception as exc:
            raise CapabilityUpdateError(
                "Brown Octopus could not prepare the new capability index. "
                "The existing in-memory and persisted snapshot was preserved."
            ) from exc

        with self._snapshot_lock.write():
            set_tools(tools)
            load_index(tools=tools, embeddings=embeddings)
            self._build_pipeline()
        return report
