"""Run the current Brown Octopus selector and write a retrieval baseline."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
from pathlib import Path

from brown_octopus import Octopus
from evals.datasets.internal import load_tasks
from evals.retrieval.metrics import ranked_metrics

BENCHMARK_KS = (1, 4, 5, 8, 16)


def _turns(tasks: list[dict], limit: int | None):
    count = 0
    for task in tasks:
        for turn_number, turn in enumerate(task.get("turns", []), start=1):
            if limit is not None and count >= limit:
                return
            count += 1
            yield task, turn_number, turn


def _coverage(required: list[str], diagnostics: dict) -> dict[str, object]:
    candidate_ids = set(_candidate_ranked(diagnostics))
    selected_ids = {
        tool_id
        for item in diagnostics.get("per_intent", [])
        for tool_id in item.get("selected_tool_ids", [])
        if tool_id
    }
    required_set = set(required)
    missing_from_candidates = sorted(required_set - candidate_ids)
    present_but_not_selected = sorted((required_set & candidate_ids) - selected_ids)
    return {
        "required_count": len(required_set),
        "candidate_count": len(candidate_ids),
        "candidate_recall": (
            len(required_set & candidate_ids) / len(required_set)
            if required_set else 1.0
        ),
        "missing_from_candidates": missing_from_candidates,
        "present_but_not_selected": present_but_not_selected,
    }


def _candidate_ranked(diagnostics: dict) -> list[str]:
    """Flatten per-intent candidate rankings without duplicate tool IDs."""
    ranked = []
    seen = set()
    for item in diagnostics.get("per_intent", []):
        for tool_id in item.get("candidate_tool_ids", []):
            if tool_id and tool_id not in seen:
                seen.add(tool_id)
                ranked.append(tool_id)
    return ranked


async def run(args) -> dict:
    octopus = Octopus(index_path=args.index_path)
    await octopus.initialize()
    tasks = load_tasks(args.tasks)
    rows = []
    for task, turn_number, turn in _turns(tasks, args.limit):
        result = octopus.retrieve_result(
            turn["query"],
            session_id=f"baseline-{task['id']}",
        )
        current = ranked_metrics(
            result.retrieved_tools,
            required=turn.get("required_tools", []),
            supporting=turn.get("supporting_tools", []),
            ks=BENCHMARK_KS,
        )
        active = ranked_metrics(
            result.tools,
            required=turn.get("required_tools", []),
            supporting=turn.get("supporting_tools", []),
        )
        metrics = result.metadata.get("retrieval_metrics", {})
        timing = result.metadata.get("timing_ms", {})
        diagnostics = result.metadata.get("retrieval_diagnostics", {})
        candidate_ranked = _candidate_ranked(diagnostics)
        rows.append({
            "task_id": task["id"],
            "category": task["category"],
            "turn": turn_number,
            "query": turn["query"],
            "current_turn": current,
            "active_context": active,
            "active_tool_characters": metrics.get("active_tool_characters"),
            "timing_ms": timing,
            "coverage": _coverage(turn.get("required_tools", []), diagnostics),
            "candidate_metrics": ranked_metrics(
                candidate_ranked,
                required=turn.get("required_tools", []),
                supporting=turn.get("supporting_tools", []),
                ks=BENCHMARK_KS,
            ),
            "candidate_ranked_tool_ids": candidate_ranked,
            "retrieval_diagnostics": diagnostics,
            "selection_strategy": (
                result.metadata.get("selection_strategy")
                or getattr(octopus.selection_strategy, "name", None)
            ),
        })

    def average(path: tuple[str, ...]) -> float | None:
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            if isinstance(value, (int, float)):
                values.append(float(value))
        return statistics.mean(values) if values else None

    summary = {
        "rows": len(rows),
        "index_path": str(args.index_path),
        "selection_strategy": rows[0].get("selection_strategy") if rows else None,
        "current_turn": {
            f"recall_at_{k}": average(("current_turn", f"recall_at_{k}"))
            for k in BENCHMARK_KS
        },
        "current_turn_precision": {
            f"precision_at_{k}": average(("current_turn", f"precision_at_{k}"))
            for k in BENCHMARK_KS
        },
        "current_turn_ndcg": {
            f"ndcg_at_{k}": average(("current_turn", f"ndcg_at_{k}"))
            for k in BENCHMARK_KS
        },
        "current_turn_average_k": average(("current_turn", "retrieved_count")),
        "active_context_average_k": average(("active_context", "retrieved_count")),
        "active_context_average_characters": average(("active_tool_characters",)),
        "total_latency_ms": average(("timing_ms", "total")),
        "candidate_recall": average(("coverage", "candidate_recall")),
        "candidate_ranked": {
            f"recall_at_{k}": average(("candidate_metrics", f"recall_at_{k}"))
            for k in BENCHMARK_KS
        },
        "candidate_ranked_precision": {
            f"precision_at_{k}": average(("candidate_metrics", f"precision_at_{k}"))
            for k in BENCHMARK_KS
        },
        "candidate_ranked_ndcg": {
            f"ndcg_at_{k}": average(("candidate_metrics", f"ndcg_at_{k}"))
            for k in BENCHMARK_KS
        },
        "turns_with_candidate_coverage_failure": sum(
            bool(row["coverage"]["missing_from_candidates"])
            for row in rows
        ),
        "turns_with_selection_failure_after_candidate_coverage": sum(
            bool(row["coverage"]["present_but_not_selected"])
            for row in rows
        ),
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=Path("data/indexes/default"))
    parser.add_argument("--tasks", type=Path, default=Path("data/evals/octopus_capability_benchmark.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval/current-baseline.json"))
    args = parser.parse_args()
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
