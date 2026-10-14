"""Standard ranked-retrieval metrics for capability evaluation."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from math import log2


def _ids(items: Iterable[str | dict]) -> list[str]:
    return [item if isinstance(item, str) else item.get("name", "") for item in items]


def relevance_grades(required: Iterable[str], supporting: Iterable[str]) -> dict[str, int]:
    grades = {tool_id: 1 for tool_id in supporting}
    grades.update({tool_id: 3 for tool_id in required})
    return grades


def recall_at_k(retrieved: Sequence[str | dict], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 1.0
    return len(set(_ids(retrieved[:k])) & relevant_set) / len(relevant_set)


def precision_at_k(retrieved: Sequence[str | dict], relevant: Iterable[str], k: int) -> float:
    selected = _ids(retrieved[:k])
    if not selected:
        return 0.0
    relevant_set = set(relevant)
    return sum(tool_id in relevant_set for tool_id in selected) / len(selected)


def _dcg(grades: Sequence[int]) -> float:
    return sum(grade / log2(index + 2) for index, grade in enumerate(grades))


def ndcg_at_k(
    retrieved: Sequence[str | dict],
    grades: dict[str, int],
    k: int,
) -> float:
    """Compute graded nDCG@k using required=3 and supporting=1 labels."""
    actual = [grades.get(tool_id, 0) for tool_id in _ids(retrieved[:k])]
    ideal = sorted(grades.values(), reverse=True)[:k]
    ideal_dcg = _dcg(ideal)
    return _dcg(actual) / ideal_dcg if ideal_dcg else 1.0


def ranked_metrics(
    retrieved: Sequence[str | dict],
    *,
    required: Iterable[str],
    supporting: Iterable[str],
    ks: Iterable[int] = (1, 4, 8, 16),
) -> dict[str, float | int | bool]:
    required = list(required)
    supporting = list(supporting)
    relevant = set(required) | set(supporting)
    grades = relevance_grades(required, supporting)
    result: dict[str, float | int | bool] = {
        "retrieved_count": len(retrieved),
        "complete_required": set(required) <= set(_ids(retrieved)),
    }
    for k in ks:
        result[f"recall_at_{k}"] = recall_at_k(retrieved, required, k)
        result[f"precision_at_{k}"] = precision_at_k(retrieved, relevant, k)
        result[f"ndcg_at_{k}"] = ndcg_at_k(retrieved, grades, k)
    return result
