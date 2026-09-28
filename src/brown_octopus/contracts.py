"""Stable contracts between Octopus pipeline components."""

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TypeVar


T = TypeVar("T")


@dataclass
class CapabilityDiscoveryResult:
    """A capability snapshot plus source-level discovery health."""

    tools: list[dict] = field(default_factory=list)
    successful_sources: list[str] = field(default_factory=list)
    failed_sources: dict[str, str] = field(default_factory=dict)
    authoritative: bool = True


@dataclass
class SessionState:
    """Serializable capability-context state for one host-owned session."""

    session_id: str
    turn: int = 0
    active_state: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """The capability context returned to an external harness."""

    tool_ids: list[str]
    tools: list[dict]
    scores: dict[str, float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_tools: list[dict] = field(default_factory=list)
    session_id: str = "default"
    turn: int = 0

    @property
    def active_tools(self) -> list[dict]:
        """Alias making the final active context explicit to callers."""
        return self.tools

    @property
    def active_state(self) -> dict[str, int]:
        """Compatibility view of the context manager's active-state timestamps."""
        value = self.metadata.get("active_state", {})
        return dict(value) if isinstance(value, dict) else {}


class IntentAnalyzer(Protocol):
    def analyze(self, query: str) -> list[dict]: ...


class CandidateRetriever(Protocol):
    def retrieve(self, query: str, intent: dict) -> list[dict]: ...


class SelectionStrategy(Protocol):
    name: str

    def select(
        self,
        query: str,
        intent: dict,
        candidates: list[dict],
    ) -> list[dict]: ...


class CapabilityContextManager(Protocol):
    def update(self, retrieved_tools: list[dict], turn: int) -> RetrievalResult: ...

    def reset(self) -> None: ...


class CapabilitySource(Protocol):
    """Host-replaceable source of the current capability universe."""

    async def discover(self) -> CapabilityDiscoveryResult: ...


class SessionStore(Protocol):
    """Session state boundary with an atomic mutation operation.

    External implementations should make ``mutate`` transactional or protect
    it with a distributed lock. A get/mutate/save sequence is intentionally
    not part of the contract because it is unsafe across processes.
    """

    def get(self, session_id: str) -> SessionState: ...

    def mutate(self, session_id: str, operation: Callable[[SessionState], T]) -> T: ...

    def reset(self, session_id: str) -> SessionState: ...

    def delete(self, session_id: str) -> None: ...

    def snapshot(self, session_id: str) -> dict | None: ...
