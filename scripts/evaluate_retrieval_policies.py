"""Evaluate cutoff policies against a ranked-results CSV.

The CSV format is the one emitted by ``experiment_adaptive_cutoff.py``:
``query`` and ``tool_name`` are required, while ``score`` and ``rank`` are
optional.  This lets us evaluate historical experiments without reloading a
model or contacting MCP servers.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evals.datasets.internal import load_tasks, validate_tasks
from evals.retrieval.evaluate import compare_policies
from evals.retrieval.policies import fixed_k, gap_cluster_k
from scripts.experiment_adaptive_cutoff import bounded_max_gap, add_gap_statistics


def load_rankings(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            query = row["query"]
            item = {
                "name": row.get("tool_name") or row.get("name"),
                "score": float(row.get("score", 0.0)),
            }
            grouped.setdefault(query, []).append(item)
    for query, ranking in grouped.items():
        ranking.sort(key=lambda item: item["score"], reverse=True)
        for rank, item in enumerate(ranking, start=1):
            item["rank"] = rank
        grouped[query] = add_gap_statistics(ranking)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, default=Path("data/evals/octopus_capability_benchmark.json"))
    parser.add_argument("--universe", type=Path, default=Path("data/indexes/default/tools.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval/policy-evaluation.json"))
    args = parser.parse_args()

    tasks = load_tasks(args.tasks)
    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    validate_tasks(tasks, universe)
    rankings = load_rankings(args.rankings)
    policies = {
        "fixed_4": lambda ranking: fixed_k(ranking, 4),
        "bounded_max_gap": lambda ranking: {
            "k": len(bounded_max_gap(ranking)),
            "reason": "bounded-max-gap",
        },
        "gap_cluster": gap_cluster_k,
    }
    result = compare_policies(
        tasks,
        rankings,
        policies,
        universe_count=len(universe),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    for name, value in result.items():
        print(name, json.dumps(value["summary"], sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
