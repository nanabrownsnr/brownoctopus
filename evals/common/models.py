from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalResult:
    tool_ids: list[str]
    scores: dict[str, float] | None
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
