"""Deterministic candidate-cutoff policies used by retrieval evaluations.

These policies operate on an already ranked list.  They deliberately do not
change Brown Octopus' production selector; experiments must be measurable
before a new policy becomes a default.
"""

from __future__ import annotations

from statistics import median


def _bounded(ranking: list[dict], max_k: int) -> list[dict]:
    return ranking[: max(0, max_k)]


def gap_cluster_k(
    ranking: list[dict],
    *,
    min_k: int = 4,
    max_k: int = 16,
    min_gap_percent: float = 2.0,
    outlier_multiple: float = 1.75,
) -> dict:
    """Choose a prefix at the first unusually large score gap.

    The current bounded-max-gap policy chooses the *largest* gap.  This
    experiment instead treats adjacent gaps as a distribution and chooses the
    first gap that is both operationally meaningful and an outlier relative to
    the local gap distribution.  It is conservative: a minimum floor is
    always retained and a flat ranking reaches the cap.

    The return shape mirrors the experiment scripts and includes diagnostics so
    evaluation output can explain every cutoff.
    """

    if not ranking:
        return {"k": 0, "reason": "empty", "gap_percent": None}

    head = ranking[: max_k + 1]
    peak = float(head[0].get("score", 0.0))
    if len(head) < 2 or peak == 0:
        return {
            "k": min(len(ranking), max_k),
            "reason": "short-or-zero-peak",
            "gap_percent": None,
        }

    gaps = []
    for index in range(len(head) - 1):
        gap = (float(head[index].get("score", 0.0)) - float(head[index + 1].get("score", 0.0)))
        gaps.append({"k": index + 1, "gap_percent": gap / peak * 100.0})

    baseline = median(item["gap_percent"] for item in gaps)
    minimum = max(min_gap_percent, baseline * outlier_multiple)
    eligible = [item for item in gaps if item["gap_percent"] >= minimum]

    if eligible:
        boundary = eligible[0]
        cutoff = max(min_k, boundary["k"])
        cutoff = min(cutoff, max_k, len(ranking))
        return {
            "k": cutoff,
            "reason": "first-gap-cluster-boundary",
            "gap_percent": boundary["gap_percent"],
            "baseline_gap_percent": baseline,
            "threshold_gap_percent": minimum,
        }

    return {
        "k": min(len(ranking), max_k),
        "reason": "fallback-cap",
        "gap_percent": None,
        "baseline_gap_percent": baseline,
        "threshold_gap_percent": minimum,
    }


def fixed_k(ranking: list[dict], k: int = 4) -> dict:
    """Return a fixed-prefix policy in the common experiment shape."""

    return {"k": min(len(ranking), max(0, k)), "reason": f"fixed-{k}"}
