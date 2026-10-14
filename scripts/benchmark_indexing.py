"""Benchmark local indexing without models, MCPs, or network access.

Examples:
    uv run python scripts/benchmark_indexing.py --sizes 10 100 1000
    uv run python scripts/benchmark_indexing.py --sizes 100 --json

The synthetic provider deliberately makes embedding work deterministic. The
timings measure Brown Octopus lifecycle, merge, embedding dispatch, and local
snapshot publication while keeping model inference out of the benchmark.
Use a real provider separately when measuring hardware throughput.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import time
import tracemalloc
from pathlib import Path

import torch

from brown_octopus import CapabilityDiscoveryResult, OctopusIndex


class SyntheticSource:
    def __init__(self, source_id: str, tools: list[dict]) -> None:
        self.source_id = source_id
        self.tools = tools

    async def discover(self) -> CapabilityDiscoveryResult:
        return CapabilityDiscoveryResult(
            tools=list(self.tools),
            successful_sources=[self.source_id],
        )


class SyntheticEmbeddingProvider:
    model_id = "synthetic-benchmark-provider"
    dimension = 8


def _tools(source_id: str, count: int, offset: int = 0) -> list[dict]:
    return [
        {
            "capability_id": f"{source_id}-tool-{index}",
            "name": f"tool_{index}",
            "description": f"Synthetic capability {index} for benchmark {source_id}",
            "source_id": source_id,
        }
        for index in range(offset, offset + count)
    ]


def _install_synthetic_runtime() -> None:
    """Disable model/runtime work so this benchmark is deterministic."""
    import brown_octopus.octopus as octopus_module

    octopus_module.initialize_analyzer = lambda: None
    octopus_module.initialize_retriever = lambda *args: None
    octopus_module.build_embeddings = lambda tools, embedding_provider=None: torch.ones(
        (len(tools), SyntheticEmbeddingProvider.dimension), dtype=torch.float32
    )
    octopus_module.set_tools = lambda tools: None
    octopus_module.load_index = lambda **kwargs: None


def _cuda_memory() -> dict[str, int] | None:
    if not torch.cuda.is_available():
        return None
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _snapshot_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


async def _timed(operation, index_path: Path):
    tracemalloc.start()
    tracemalloc.reset_peak()
    started = time.perf_counter()
    report = await operation()
    wall_ms = (time.perf_counter() - started) * 1000
    _, peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return report, wall_ms, peak_python_bytes, _snapshot_bytes(index_path)


def _run_real_embedding(size: int, model: str, device: str, batch_size: int) -> dict:
    """Measure real local embedding only; never downloads a model implicitly."""
    from brown_octopus.embedding import LocalEmbeddingProvider

    os.environ.setdefault("BROWN_OCTOPUS_ALLOW_MODEL_DOWNLOAD", "0")
    provider = LocalEmbeddingProvider(
        model_name=model,
        device=device,
        batch_size=batch_size,
    )
    texts = [
        f"Synthetic capability description {index} for embedding benchmark"
        for index in range(size)
    ]
    tracemalloc.start()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    vectors = provider.embed(texts)
    elapsed_ms = (time.perf_counter() - started) * 1000
    _, peak_python_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "size": size,
        "elapsed_ms": round(elapsed_ms, 3),
        "texts_per_second": round(size / (elapsed_ms / 1000), 3)
        if elapsed_ms
        else None,
        "model_id": provider.model_id,
        "device": provider.device,
        "batch_size": provider.batch_size,
        "dimension": int(vectors.shape[-1]) if hasattr(vectors, "shape") else None,
        "python_peak_bytes": peak_python_bytes,
        "cuda": _cuda_memory(),
    }


async def _run_size(size: int) -> dict:
    with tempfile.TemporaryDirectory(prefix="brown-octopus-benchmark-") as directory:
        index_path = Path(directory) / "index"
        base_source = SyntheticSource("base", _tools("base", size))
        index = OctopusIndex(
            sources=[base_source],
            index_path=index_path,
            embedding_provider=SyntheticEmbeddingProvider(),
        )
        index._octopus._build_pipeline = lambda: None

        create_report, create_wall_ms, create_memory, create_snapshot = await _timed(
            lambda: index.create(replace=True), index_path
        )
        unchanged_report, unchanged_wall_ms, unchanged_memory, unchanged_snapshot = (
            await _timed(index.update, index_path)
        )

        added_source = SyntheticSource("added", _tools("added", 1))
        add_report, add_wall_ms, add_memory, add_snapshot = await _timed(
            lambda: index.add(added_source), index_path
        )

        changes = {}
        for percentage in (1, 10, 100):
            changed_count = max(1, (size * percentage) // 100)
            changed_tools = _tools("changed", changed_count)
            # Keep the original capability IDs so this measures changed rows,
            # rather than an append-only add.
            for index_number, tool in enumerate(changed_tools):
                tool["capability_id"] = f"base-tool-{index_number}"
                tool["source_id"] = "base"
                tool["name"] = f"tool_{index_number}"
            base_source.tools = list(base_source.tools)
            for tool in changed_tools:
                base_source.tools[int(tool["capability_id"].rsplit("-", 1)[1])][
                    "description"
                ] += f" changed-{percentage}"
            report, wall_ms, memory, snapshot = await _timed(index.update, index_path)
            changes[f"{percentage}%"] = {
                "wall_ms": round(wall_ms, 3),
                "report_ms": round(report.duration_ms or 0, 3),
                "embedded_count": report.embedded_count,
                "changed_count": report.changed_count,
                "tool_count": report.tool_count,
                "python_peak_bytes": memory,
                "snapshot_bytes": snapshot,
            }

        await index.close()
        return {
            "size": size,
            "create": {
                "wall_ms": round(create_wall_ms, 3),
                "report_ms": round(create_report.duration_ms or 0, 3),
                "embedded_count": create_report.embedded_count,
                "tool_count": create_report.tool_count,
                "python_peak_bytes": create_memory,
                "snapshot_bytes": create_snapshot,
            },
            "unchanged_update": {
                "wall_ms": round(unchanged_wall_ms, 3),
                "report_ms": round(unchanged_report.duration_ms or 0, 3),
                "embedded_count": unchanged_report.embedded_count,
                "tool_count": unchanged_report.tool_count,
                "python_peak_bytes": unchanged_memory,
                "snapshot_bytes": unchanged_snapshot,
            },
            "incremental_add": {
                "wall_ms": round(add_wall_ms, 3),
                "report_ms": round(add_report.duration_ms or 0, 3),
                "embedded_count": add_report.embedded_count,
                "added_count": add_report.added_count,
                "tool_count": add_report.tool_count,
                "python_peak_bytes": add_memory,
                "snapshot_bytes": add_snapshot,
            },
            "changes": changes,
        }


async def _run_all(sizes: list[int]) -> list[dict]:
    return await asyncio.gather(*(_run_size(size) for size in sizes))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 100, 1000])
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "--real-embedding",
        action="store_true",
        help="measure the cached local embedding model instead of synthetic lifecycle work",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if any(size < 1 for size in args.sizes):
        parser.error("sizes must be positive")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")

    if args.real_embedding:
        results = [
            _run_real_embedding(
                size,
                args.model,
                None if args.device == "auto" else args.device,
                args.batch_size,
            )
            for size in args.sizes
        ]
    else:
        _install_synthetic_runtime()
        results = asyncio.run(_run_all(args.sizes))
    if args.as_json:
        print(json.dumps(results, indent=2))
        return
    for result in results:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
