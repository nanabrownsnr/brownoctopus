"""Selection strategy implementations.

The strategy boundary is intentionally small: candidate retrieval has already
run, and a strategy only decides which candidate definitions are relevant.
"""

from typing import Any

from brown_octopus.errors import StrategyUnavailableError
from brown_octopus.selection import select_min4_bounded_max_gap


class BoundedMaxGapSelectionStrategy:
    """Frozen Brown Octopus V3 per-intent selector."""

    name = "brown_octopus_v3"

    def select(self, query: str, intent: dict, candidates: list[dict]) -> list[dict]:
        return select_min4_bounded_max_gap(candidates)


class JevSelectionStrategy:
    """Select relevant candidate capabilities with Jev relevance filtering."""

    name = "jev_relevance"

    def __init__(
        self,
        *,
        threshold: float = 0.2,
        mode: str = "listwise",
        reranker: Any = None,
        api_key: str | None = None,
    ) -> None:
        self.threshold = threshold
        self.mode = mode
        self._reranker = reranker
        self.api_key = api_key

    @property
    def reranker(self) -> Any:
        if self._reranker is None:
            try:
                from jev_reranker import JevReranker
            except ImportError as exc:
                raise StrategyUnavailableError(
                    "Jev is not installed. Install the 'jev' extra."
                ) from exc

            self._reranker = JevReranker(
                api_key=self.api_key,
                mode=self.mode,
            )
        return self._reranker

    @staticmethod
    def _document(candidate: dict) -> str:
        return f"{candidate.get('name', '')}: {candidate.get('description') or ''}"

    def select(self, query: str, intent: dict, candidates: list[dict]) -> list[dict]:
        if not candidates:
            return []

        documents = [self._document(candidate) for candidate in candidates]
        response = self.reranker.relevance_rerank(
            intent.get("text") or query,
            documents,
            threshold=self.threshold,
            return_documents=False,
        )

        selected = []
        for item in response.get("results", []):
            index = item.get("document_index")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                continue
            candidate = dict(candidates[index])
            candidate["jev_score"] = float(item.get("score", 0.0))
            selected.append(candidate)
        return selected


class CallableSelectionStrategy:
    """Small adapter useful for integrations and deterministic tests."""

    def __init__(self, function, name: str = "custom") -> None:
        self.function = function
        self.name = name

    def select(self, query: str, intent: dict, candidates: list[dict]) -> list[dict]:
        return self.function(query, intent, candidates)
