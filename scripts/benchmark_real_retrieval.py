"""Measure retrieval with the configured Brown Octopus runtime.

Unlike ``benchmark_retrieval.py`` this script loads the actual analyzer,
embedding provider, persisted index, and optional index backend.  It is
intended for controlled local/CI runs, not for application startup.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import tracemalloc
from pathlib import Path

from brown_octopus import Octopus
from brown_octopus.device import detect_compute_device
from evals.datasets.internal import load_tasks


def _queries(tasks: list[dict], limit: int | None) -> list[str]:
    values = []
    for task in tasks:
        for turn in task.get("turns", []):
            if turn["query"] not in values:
                values.append(turn["query"])
    return values[:limit] if limit else values


async def run(args) -> dict:
    index_store = None
    if args.postgres_url:
        from brown_octopus.index_store import PostgresIndexStore
        index_store = PostgresIndexStore(args.postgres_url, index_name=args.index_name)
    octopus = Octopus(index_path=args.index_path, index_store=index_store)
    await octopus.initialize()
    queries = _queries(load_tasks(args.tasks), args.limit)
    samples = []
    for query in queries:
        tracemalloc.start()
        tracemalloc.reset_peak()
        started = time.perf_counter()
        result = octopus.retrieve_result(query, session_id=f"benchmark-{len(samples)}")
        elapsed = (time.perf_counter() - started) * 1000.0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        timing = result.metadata.get("timing_ms", {})
        metrics = result.metadata.get("retrieval_metrics", {})
        samples.append({
            "query": query,
            "total_ms": elapsed,
            "analysis_ms": timing.get("analysis"),
            "retrieval_ms": timing.get("retrieval"),
            "selection_ms": timing.get("selection"),
            "context_update_ms": timing.get("context_update"),
            "query_characters": metrics.get("query_characters", len(query)),
            "active_tool_count": metrics.get("active_tool_count", len(result.tools)),
            "active_tool_characters": metrics.get("active_tool_characters"),
            "retrieved_tool_count": metrics.get("retrieved_tool_count"),
            "python_peak_bytes": peak,
        })

    def median(name):
        values = [row[name] for row in samples if row[name] is not None]
        return statistics.median(values) if values else None

    provider = octopus.embedding_provider
    return {
        "backend": type(index_store).__name__ if index_store else "LocalIndexStore",
        "index_path": str(args.index_path),
        "index_size": len(octopus.index_store.load_tools()),
        "provider": type(provider).__name__ if provider else None,
        "device": getattr(provider, "device", None) or detect_compute_device(),
        "query_count": len(samples),
        "median_total_ms": median("total_ms"),
        "median_analysis_ms": median("analysis_ms"),
        "median_retrieval_ms": median("retrieval_ms"),
        "median_selection_ms": median("selection_ms"),
        "median_context_update_ms": median("context_update_ms"),
        "median_active_tool_count": median("active_tool_count"),
        "median_active_tool_characters": median("active_tool_characters"),
        "peak_python_bytes": max((row["python_peak_bytes"] for row in samples), default=0),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-path", type=Path, default=Path("data/indexes/default"))
    parser.add_argument("--tasks", type=Path, default=Path("data/evals/octopus_capability_benchmark.json"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--postgres-url")
    parser.add_argument("--index-name", default="default")
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval/real-baseline.json"))
    args = parser.parse_args()
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "samples"}, indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
