"""Benchmark retrieval cost against query size and capability-universe size.

This is a model-free orchestration baseline. It measures the Brown Octopus
pipeline, context update, selection, and serialized capability payload without
network access or model startup noise.

Examples:
    uv run python scripts/benchmark_retrieval.py --sizes 10 100 1000 10000 --json
    uv run python scripts/benchmark_retrieval.py --sizes 100 1000 --candidate-k 16 64
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import tracemalloc

from brown_octopus.context_manager import ActiveCapabilityContext
from brown_octopus.pipeline import CapabilityPipeline
from brown_octopus.strategies import BoundedMaxGapSelectionStrategy
from brown_octopus.tool_registry import capability_id


QUERY_PROFILES = {
    "short": "find email",
    "medium": "find the latest customer email and prepare a response",
    "long": "find the latest customer email and prepare a response " + "with context " * 35,
    "very_long": "find the latest customer email and prepare a response " + "with context " * 160,
}


def synthetic_tools(size: int) -> list[dict]:
    return [
        {
            "capability_id": f"synthetic-tool-{index}",
            "name": f"synthetic_tool_{index}",
            "description": (
                f"Synthetic capability {index}. Search, inspect, and operate on "
                "a bounded test resource."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "resource": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
            "source_id": "synthetic",
        }
        for index in range(size)
    ]


class SyntheticAnalyzer:
    def analyze(self, query: str) -> list[dict]:
        return [{"action": "find", "target": query, "retrieval_text": query}]


class SyntheticCandidateRetriever:
    name = "synthetic_linear_candidate_search"

    def __init__(self, tools: list[dict], candidate_k: int | None) -> None:
        self.tools = tools
        self.candidate_k = candidate_k

    def retrieve(self, query, intent, allowed_mcp_urls=None):
        # Deliberately scan the complete universe so the benchmark exposes
        # index-size scaling separately from the selector and context stages.
        ranked = []
        for index, tool in enumerate(self.tools):
            ranked.append({**tool, "score": 1.0 - (index / (len(self.tools) + 1))})
        return ranked if self.candidate_k is None else ranked[: self.candidate_k]


def _lookup(tools: list[dict]):
    by_id = {capability_id(tool): tool for tool in tools}
    return lambda ids: [by_id[item] for item in ids if item in by_id]


def _summary(samples: list[dict]) -> dict:
    def stats(key: str) -> dict:
        values = [sample[key] for sample in samples]
        return {
            "median_ms": round(statistics.median(values), 3),
            "p95_ms": round(statistics.quantiles(values, n=20)[18], 3)
            if len(values) >= 2
            else round(values[0], 3),
        }

    first = samples[0]
    return {
        "query_characters": first["query_characters"],
        "retrieval_text_characters": first["retrieval_text_characters"],
        "candidate_count": first["candidate_count"],
        "retrieved_tool_count": first["retrieved_tool_count"],
        "active_tool_count": first["active_tool_count"],
        "retrieved_tool_characters": first["retrieved_tool_characters"],
        "active_tool_characters": first["active_tool_characters"],
        "timing_ms": {
            "total": stats("total_ms"),
            "analysis": stats("analysis_ms"),
            "retrieval": stats("retrieval_ms"),
            "selection": stats("selection_ms"),
            "context_update": stats("context_update_ms"),
        },
        "python_peak_bytes": max(sample["python_peak_bytes"] for sample in samples),
    }


async def run_case(size: int, profile: str, candidate_k: int | None, repeats: int) -> dict:
    tools = synthetic_tools(size)
    pipeline = CapabilityPipeline(
        SyntheticAnalyzer(),
        SyntheticCandidateRetriever(tools, candidate_k),
        BoundedMaxGapSelectionStrategy(),
    )
    query = QUERY_PROFILES[profile]
    samples = []
    for _ in range(repeats):
        context = ActiveCapabilityContext(_lookup(tools), ttl=8, active_cap=30)
        tracemalloc.start()
        tracemalloc.reset_peak()
        started = time.perf_counter()
        result = pipeline.retrieve(query, turn=1, context_manager=context)
        wall_ms = (time.perf_counter() - started) * 1000
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        timing = result["timing_ms"]
        metrics = result["retrieval_metrics"]
        samples.append({
            "total_ms": wall_ms,
            "analysis_ms": timing["analysis"],
            "retrieval_ms": timing["retrieval"],
            "selection_ms": timing["selection"],
            "context_update_ms": timing["context_update"],
            "python_peak_bytes": peak,
            **metrics,
        })
    return {
        "index_size": size,
        "query_profile": profile,
        "candidate_k": candidate_k,
        "repeats": repeats,
        **_summary(samples),
    }


async def run_all(sizes: list[int], candidate_ks: list[int | None], repeats: int) -> list[dict]:
    cases = [
        run_case(size, profile, candidate_k, repeats)
        for size in sizes
        for profile in QUERY_PROFILES
        for candidate_k in candidate_ks
    ]
    return await asyncio.gather(*cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 100, 1000, 10000])
    parser.add_argument("--candidate-k", nargs="+", type=int, default=[None], dest="candidate_ks")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    if any(size < 1 for size in args.sizes) or args.repeats < 1:
        parser.error("sizes and repeats must be positive")
    candidate_ks = [None if value is None else value for value in args.candidate_ks]
    results = asyncio.run(run_all(args.sizes, candidate_ks, args.repeats))
    print(json.dumps(results, indent=2 if args.as_json else None, sort_keys=True))


if __name__ == "__main__":
    main()
