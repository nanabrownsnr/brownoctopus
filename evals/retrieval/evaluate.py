"""Evaluate ranked retrieval results against the reviewed task dataset."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable

from evals.common.metrics import score_context


def evaluate_ranking(
    tasks: Iterable[dict],
    rankings: dict[str, list[dict]],
    *,
    selector: Callable[[list[dict]], dict],
    universe_count: int,
) -> list[dict]:
    """Score one ranked result per task.

    ``rankings`` is keyed by task/turn query.  A task with multiple turns is
    evaluated turn-by-turn, while single-turn tasks use their top-level query.
    Missing rankings are reported explicitly instead of being silently scored
    as failures.
    """

    rows = []
    for task in tasks:
        turns = task.get("turns") or [{"query": task["query"], **task}]
        for turn_number, turn in enumerate(turns, start=1):
            query = turn["query"]
            ranking = rankings.get(query)
            if ranking is None:
                rows.append({
                    "task_id": task["id"],
                    "turn": turn_number,
                    "query": query,
                    "status": "missing-ranking",
                })
                continue
            decision = selector(ranking)
            selected = ranking[: decision["k"]]
            selected_ids = [item.get("name") for item in selected]
            required = turn.get("required_tools", task.get("required_tools", []))
            supporting = turn.get("supporting_tools", task.get("supporting_tools", []))
            metrics = score_context(selected_ids, required, supporting, universe_count)
            rows.append({
                "task_id": task["id"],
                "turn": turn_number,
                "query": query,
                "status": "ok",
                "policy_k": decision["k"],
                "policy_reason": decision.get("reason"),
                **metrics,
                "payload_characters": sum(
                    len(str(item.get("name", ""))) + len(str(item.get("description", "")))
                    for item in selected
                ),
            })
    return rows


def aggregate(rows: Iterable[dict]) -> dict:
    """Produce stable aggregate metrics suitable for JSON/CSV reports."""

    valid = [row for row in rows if row.get("status") == "ok"]
    if not valid:
        return {"rows": 0, "missing_rankings": 0}

    def average(name: str) -> float:
        return sum(float(row[name]) for row in valid) / len(valid)

    return {
        "rows": len(valid),
        "missing_rankings": sum(row.get("status") != "ok" for row in rows),
        "required_recall": average("required_recall"),
        "complete_required_rate": sum(row["complete_required"] for row in valid) / len(valid),
        "supporting_recall": average("supporting_recall"),
        "average_tool_count": average("tool_count"),
        "average_context_reduction": average("context_reduction"),
        "average_payload_characters": average("payload_characters"),
    }


def compare_policies(
    tasks: Iterable[dict],
    rankings: dict[str, list[dict]],
    policies: dict[str, Callable[[list[dict]], dict]],
    *,
    universe_count: int,
) -> dict:
    """Evaluate several policies and return per-policy rows and aggregates."""

    result = {}
    for name, selector in policies.items():
        rows = evaluate_ranking(
            tasks,
            rankings,
            selector=selector,
            universe_count=universe_count,
        )
        result[name] = {"summary": aggregate(rows), "rows": rows}
    return result
