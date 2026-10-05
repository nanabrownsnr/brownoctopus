from pathlib import Path
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import replace
from threading import Condition, RLock
import logging
import json
import hashlib
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version

from brown_octopus.analyzer import initialize_analyzer
from brown_octopus.analyzer import DeterministicIntentAnalyzer
from brown_octopus.config import OctopusConfig
from brown_octopus.context_manager import ActiveCapabilityContext
from brown_octopus.contracts import CapabilitySource, RetrievalResult, SessionStore
from brown_octopus.embedding import EmbeddingProvider
from brown_octopus.embedding import LocalEmbeddingProvider, MODEL_NAME
from brown_octopus.index_store import (
    IndexStore,
    LocalIndexStore,
    load_embeddings,
    load_metadata,
    load_tools,
    save_embeddings,
    save_metadata,
    save_snapshot_atomic,
    save_tools,
)
# These names remain module-level compatibility seams for legacy bootstrap,
# scripts, and integrations that monkeypatch the historical retrieval path.
from brown_octopus.mcp_discovery import discover_universe
from brown_octopus.observability import configure_logging
from brown_octopus.pipeline import CapabilityPipeline, process_turn
from brown_octopus.retriever import (
    build_embeddings,
    get_embedding_provider,
    LocalCandidateRetriever,
    VectorSearchCandidateRetriever,
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
    ModelInitializationError,
)
from brown_octopus.tool_registry import capability_id, get_tools, set_tools


logger = logging.getLogger("brown_octopus")


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
        embedding_provider: EmbeddingProvider | None = None,
        index_store: IndexStore | None = None,
        capability_ttl: int | None = None,
        active_cap: int | None = None,
    ):
        self.config = config or OctopusConfig.from_env()
        if capability_ttl is not None or active_cap is not None:
            self._configure_context_limits(
                capability_ttl=capability_ttl,
                active_cap=active_cap,
            )
        configure_logging(self.config.log_level)
        self.index_path = Path(index_path) if index_path != "data/indexes/default" else self.config.index_path
        self.catalog_path = Path(catalog_path) if catalog_path != "data/mcps.json" else self.config.catalog_path
        self.capability_source = capability_source or LocalMcpCatalogSource(self.catalog_path)
        self.embedding_provider = embedding_provider
        self.index_store = index_store or LocalIndexStore(self.index_path)
        self._snapshot_lock = SnapshotReadWriteLock()

        self.analyzer = DeterministicIntentAnalyzer()
        self.candidate_retriever = None
        self.selection_strategy = None
        self.pipeline = None
        self.sessions = session_store or InMemorySessionStore()
        # Runtime-only fallback definitions for injected pipelines. The
        # canonical production universe remains the shared tool registry.
        self._runtime_tool_definitions: dict[str, dict] = {}

    def _configure_context_limits(
        self,
        *,
        capability_ttl: int | None = None,
        active_cap: int | None = None,
    ) -> None:
        """Configure runtime session limits before initialization."""
        if getattr(self, "pipeline", None) is not None:
            raise RuntimeError(
                "Runtime context limits must be configured before initialize()."
            )

        updates = {
            key: value
            for key, value in {
                "ttl": capability_ttl,
                "active_cap": active_cap,
            }.items()
            if value is not None
        }
        if not updates:
            return

        configured = replace(self.config, **updates)
        configured.validate()
        self.config = configured

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
            if self.embedding_provider is None:
                initialize_retriever()
            else:
                initialize_retriever(self.embedding_provider)
        except Exception as exc:
            raise ModelInitializationError(
                "Brown Octopus could not load its required runtime models. "
                "Run:\n\n    brown-octopus setup-models\n\n"
                "Then retry initialization. No model downloads are performed "
                "automatically by initialize()."
            ) from exc

        if not self.index_store.exists():
            # A runtime may be started before its capability universe has been
            # provisioned.  Treat this as a valid empty staged-provisioning
            # state; create() or update() can publish the first snapshot later.
            logger.warning(
                "Capability index not found at %s; starting with an empty "
                "capability universe.",
                self.index_path,
            )
            with self._snapshot_lock.write():
                set_tools([])
                load_index(tools=[], embeddings=None)
                self._build_pipeline()
            return []

        try:
            metadata = self.index_store.load_metadata()
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

        configured_provider = self.embedding_provider
        indexed_provider = metadata.get("embedding_provider")
        if indexed_provider is None:
            indexed_provider = {
                "type": "LocalEmbeddingProvider",
                "model_id": MODEL_NAME,
            }
        elif indexed_provider.get("type") == "QwenEmbeddingProvider":
            # Indexes created by the pre-generic-provider implementation remain
            # valid when they use the unchanged default Qwen model.
            indexed_provider = dict(indexed_provider)
            indexed_provider["type"] = "LocalEmbeddingProvider"
        if configured_provider is None:
            configured_provider = (
                get_embedding_provider()
                or LocalEmbeddingProvider(model_name=MODEL_NAME)
            )
        configured_identity = {
            "type": type(configured_provider).__name__,
            "model_id": configured_provider.model_id,
        }
        if any(
            indexed_provider.get(key) != value
            for key, value in configured_identity.items()
            if indexed_provider.get(key) is not None
        ):
            raise IncompatibleIndexError(
                "Brown Octopus index was built with a different embedding provider "
                f"({indexed_provider.get('model_id')}); configured provider is "
                f"{configured_identity['model_id']}. Rebuild the index with "
                "the configured provider."
            )
        indexed_dimension = indexed_provider.get("dimension")
        configured_dimension = getattr(configured_provider, "dimension", None)
        if (
            indexed_dimension is not None
            and configured_dimension is not None
            and indexed_dimension != configured_dimension
        ):
            raise IncompatibleIndexError(
                "Brown Octopus index embedding dimension does not match the "
                "configured provider. Rebuild the index with that provider."
            )

        try:
            tools = self.index_store.load_tools()
            embeddings = (
                None
                if getattr(self.index_store, "supports_native_vector_search", False)
                else self.index_store.load_embeddings()
            )
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
        if getattr(self.index_store, "supports_native_vector_search", False):
            provider = self.embedding_provider or get_embedding_provider()
            if provider is None:
                raise RuntimeError(
                    "Native vector retrieval requires an initialized embedding provider."
                )
            candidate_retriever = VectorSearchCandidateRetriever(
                self.index_store,
                provider,
            )
        else:
            candidate_retriever = LocalCandidateRetriever(
                embedding_provider=self.embedding_provider,
            )
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
            previous = self.index_store.load_tools()
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
            if self.embedding_provider is None:
                initialize_retriever()
            else:
                initialize_retriever(self.embedding_provider)
            discovered = await self._discover_snapshot()
            tools, report = self._merge_snapshot_tools(discovered)
            if tools:
                if self.embedding_provider is None:
                    embeddings = build_embeddings(tools)
                else:
                    embeddings = build_embeddings(
                        tools,
                        embedding_provider=self.embedding_provider,
                    )
            else:
                embeddings = None
            active_provider = (
                self.embedding_provider
                or get_embedding_provider()
                or LocalEmbeddingProvider(model_name=MODEL_NAME)
            )
            metadata = {
                "version": 1,
                "index_format_version": 1,
                "brown_octopus_version": self._package_version(),
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "tool_count": len(tools),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "tool_universe_fingerprint": self._tool_universe_fingerprint(tools),
                "embedding_provider": {
                    "type": type(active_provider).__name__,
                    "model_id": active_provider.model_id,
                    "dimension": getattr(active_provider, "dimension", None),
                },
            }
            # Preserve the historical module-level seam used by integrations
            # and tests while allowing custom stores to own publication.
            if isinstance(self.index_store, LocalIndexStore):
                save_snapshot_atomic(
                    self.index_path,
                    tools,
                    embeddings,
                    metadata,
                )
            else:
                self.index_store.save_snapshot_atomic(tools, embeddings, metadata)
        except CapabilityUpdateError:
            raise
        except Exception as exc:
            raise CapabilityUpdateError(
                "Brown Octopus could not prepare the new capability index. "
                "The existing in-memory and persisted snapshot was preserved."
            ) from exc

        with self._snapshot_lock.write():
            set_tools(tools)
            load_index(
                tools=tools,
                embeddings=(
                    None
                    if getattr(self.index_store, "supports_native_vector_search", False)
                    else embeddings
                ),
            )
            self._build_pipeline()
        return report

    def reset_index(self) -> None:
        """Remove this engine's persisted capability index.

        This is an explicit setup operation. Runtime retrieval must be
        reinitialized after a subsequent ``update()`` or ``create()``.
        Session state is intentionally not modified here.
        """
        self.index_store.reset()
        set_tools([])
        load_index(tools=[], embeddings=None)
        self.pipeline = None
        self.candidate_retriever = None
        self.selection_strategy = None
        self._runtime_tool_definitions.clear()
