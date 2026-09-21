from __future__ import annotations

import math
from typing import Iterable


def _dcg(relevances: Iterable[int]) -> float:
    return sum((2**int(relevance) - 1) / math.log2(rank + 2) for rank, relevance in enumerate(relevances))


def evaluate_toolret_result(
    ranked_tool_ids: list[str],
    labels: dict[str, int],
    k: int = 10,
) -> dict[str, float]:
    """Match ToolRet's pytrec_eval definitions for one query.

    Completeness is the official evaluator's binary ``Recall@k == 1`` value.
    The ranked list is intentionally not padded when it contains fewer than k.
    """
    ranked = ranked_tool_ids[:k]
    relevances = [labels.get(tool_id, 0) for tool_id in ranked]
    ideal = sorted(labels.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    recall = sum(relevance > 0 for relevance in relevances) / max(
        1, sum(relevance > 0 for relevance in labels.values())
    )
    return {
        "ndcg@10": _dcg(relevances) / idcg if idcg else 0.0,
        "recall@10": recall,
        "completeness@10": float(recall == 1.0),
    }
