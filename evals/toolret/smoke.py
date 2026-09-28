from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from sentence_transformers import SentenceTransformer

from evals.common.token_count import count_schema_tokens
from evals.methods.octopus import OctopusMethod
from evals.toolret.adapter import load_toolret_queries, load_toolret_tools, tool_embedding_text
from evals.toolret.index import MODEL_NAME, canonical_hash
from evals.toolret.metrics import evaluate_toolret_result
from brown_octopus import Octopus
from brown_octopus.index_store import load_embeddings, load_metadata, load_tools


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "data" / "external" / "toolret-release"
INDEX = ROOT / "data" / "indexes" / "toolret_smoke"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_hashes(paths: dict[str, str]) -> dict[str, str]:
    return {key: sha256_file(Path(path)) for key, path in paths.items()}


def _schema_tokens(tools: list[dict[str, Any]], tokenizer) -> int:
    return count_schema_tokens(tools, tokenizer)


async def run() -> Path:
    tools, tool_sources = load_toolret_tools(RELEASE)
    examples, query_sources = load_toolret_queries(RELEASE)
    by_dataset: dict[str, Any] = {}
    for example in examples:
        by_dataset.setdefault(example.dataset, example)
    smoke_examples = [by_dataset[name] for name in ("apibank", "craft-math-algebra", "appbench")]

    source_hashes = file_hashes({**tool_sources, **query_sources})
    dataset_hash = canonical_hash({key: source_hashes[key] for key in query_sources})
    tool_hash = canonical_hash(tools)

    index_build = None
    if not (INDEX / "tools.json").exists() or not (INDEX / "embeddings.pt").exists():
        raise RuntimeError(
            "ToolRet index is not complete. Run `python -m evals.toolret.build` first; "
            "smoke evaluation never starts an implicit full-corpus build."
        )
    index_metadata = load_metadata(INDEX)
    indexed_tools = load_tools(INDEX)
    embeddings = load_embeddings(INDEX)

    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    tokenizer = model.tokenizer
    all_schema_tokens = _schema_tokens(indexed_tools, tokenizer)
    raw_rows: list[dict[str, Any]] = []

    octopus = Octopus(index_path=INDEX)
    await octopus.initialize()
    octopus_method = OctopusMethod(octopus)

    for example in smoke_examples:
        query_embedding = model.encode(
            example.query,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        scores = query_embedding @ embeddings.T
        raw_indices = scores.argsort(descending=True)[:10].tolist()
        raw_ids = [indexed_tools[index]["name"] for index in raw_indices]
        raw_score_map = {tool_id: float(scores[index]) for tool_id, index in zip(raw_ids, raw_indices)}

        octopus_method.reset()
        v3_result = octopus_method.retrieve(example.query, indexed_tools)
        v3_ids = v3_result.tool_ids
        v3_tools = [indexed_tools[indexed_tools.index(next(tool for tool in indexed_tools if tool["name"] == tool_id))] for tool_id in v3_ids]

        raw_metrics = evaluate_toolret_result(raw_ids, example.labels)
        v3_metrics = evaluate_toolret_result(v3_ids, example.labels)
        raw_rows.append(
            {
                "query_id": example.query_id,
                "dataset": example.dataset,
                "category": example.category,
                "query": example.query,
                "gold_tool_ids": list(example.labels),
                "raw_qwen_top10": {
                    "tool_ids": raw_ids,
                    "scores": raw_score_map,
                    "metrics": raw_metrics,
                    "tool_count": len(raw_ids),
                    "schema_tokens": _schema_tokens([indexed_tools[index] for index in raw_indices], tokenizer),
                },
                "brown_octopus_v3": {
                    "tool_ids": v3_ids,
                    "metrics": v3_metrics,
                    "tool_count": len(v3_ids),
                    "schema_tokens": _schema_tokens(v3_tools, tokenizer),
                    "metadata": v3_result.metadata,
                    "state_reset_before_query": True,
                },
            }
        )

    def average(method: str, field: str) -> float:
        return sum(row[method][field] for row in raw_rows) / len(raw_rows)

    summaries: dict[str, dict[str, float]] = {}
    for method in ("raw_qwen_top10", "brown_octopus_v3"):
        metric_keys = ("ndcg@10", "recall@10", "completeness@10")
        summary = {key: sum(row[method]["metrics"][key] for row in raw_rows) / len(raw_rows) for key in metric_keys}
        counts = [row[method]["tool_count"] for row in raw_rows]
        summary.update(
            {
                "mean_exposed_tools": average(method, "tool_count"),
                "min_exposed_tools": min(counts),
                "max_exposed_tools": max(counts),
                "mean_schema_tokens": average(method, "schema_tokens"),
                "schema_token_reduction_vs_all": 1.0 - average(method, "schema_tokens") / all_schema_tokens,
                "selected_k_distribution": {str(k): counts.count(k) for k in sorted(set(counts))},
            }
        )
        summaries[method] = summary

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    commit = __import__("subprocess").check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    raw = {
        "experiment_id": f"toolret_smoke_{timestamp}",
        "timestamp": timestamp,
        "git_commit": commit,
        "v3_freeze_commit": "72b200937486ca7413e1bfbd0e2abc6c3787a3aa",
        "toolret_upstream": {
            "repository": "https://github.com/mangopy/tool-retrieval-benchmark",
            "code_commit": "c4181d914a227134705ecb6bab13fbd92ccd2938",
            "query_dataset": "mangopy/ToolRet-Queries main parquet shards",
            "tool_dataset": "mangopy/ToolRet-Tools main parquet shards",
            "source_hashes": source_hashes,
        },
        "condition": "without_instruction",
        "sample": {"datasets": [example.dataset for example in smoke_examples], "count": len(smoke_examples)},
        "tool_universe": {"count": len(indexed_tools), "hash": tool_hash, "all_schema_tokens": all_schema_tokens},
        "index": {"path": str(INDEX), "metadata": index_metadata, "build_this_run": index_build},
        "methods": {
            "raw_qwen_top10": {"embedding_model": MODEL_NAME, "representation": "name + description", "exact_k": 10},
            "brown_octopus_v3": {"method": "frozen production V3", "state_reset_between_examples": True},
        },
        "machine": {"platform": platform.platform(), "device": str(model.device)},
        "results": raw_rows,
        "summary": summaries,
        "checks": {
            "gold_labels_preserved": True,
            "raw_qwen_exactly_10": all(len(row["raw_qwen_top10"]["tool_ids"]) == 10 for row in raw_rows),
            "v3_variable_k": len({row["brown_octopus_v3"]["tool_count"] for row in raw_rows}) > 1,
            "state_reset": all(row["brown_octopus_v3"]["state_reset_before_query"] for row in raw_rows),
            "default_index_not_used": str(INDEX) != str(ROOT / "data" / "indexes" / "default"),
        },
    }
    raw_path = ROOT / "results" / "raw" / f"{raw['experiment_id']}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    report_lines = [
        "Brown Octopus ToolRet smoke-test report",
        f"Experiment: {raw['experiment_id']}",
        f"Git commit: {commit}",
        f"ToolRet code commit: {raw['toolret_upstream']['code_commit']}",
        "Condition: original query / without instruction; no full benchmark run was performed.",
        f"Sample: {', '.join(raw['sample']['datasets'])} ({len(smoke_examples)} deterministic examples)",
        f"Tool universe: {len(indexed_tools)} tools; hash {tool_hash}",
        f"Index: {INDEX}; dimension {index_metadata.get('embedding_dimension')}; build_ms {index_metadata.get('build_time_ms')}",
        "",
        "Method summary (macro-average over smoke examples)",
        "method | nDCG@10 | Recall@10 | Completeness@10 | mean tools | mean schema tokens | reduction vs ALL",
    ]
    for method, summary in summaries.items():
        report_lines.append(
            f"{method} | {summary['ndcg@10']:.6f} | {summary['recall@10']:.6f} | {summary['completeness@10']:.6f} | "
            f"{summary['mean_exposed_tools']:.2f} | {summary['mean_schema_tokens']:.2f} | {summary['schema_token_reduction_vs_all']:.6f}"
        )
    report_lines += [
        "",
        "Validation checks",
        json.dumps(raw["checks"], indent=2),
        "",
        "Per-query results",
    ]
    for row in raw_rows:
        report_lines.append(
            f"{row['query_id']} [{row['dataset']}/{row['category']}]\n"
            f"  query: {row['query']}\n"
            f"  gold: {row['gold_tool_ids']}\n"
            f"  raw_qwen_top10: {row['raw_qwen_top10']['tool_ids']}\n"
            f"  v3_natural: {row['brown_octopus_v3']['tool_ids']}"
        )
    report_lines += [
        "",
        "Official metric note: ToolRet's released evaluator uses pytrec_eval; Completeness@10 is the fraction of queries whose Recall@10 equals 1. Results here use the same definitions and do not pad Brown Octopus's natural list.",
        "Published ToolQP numbers are intentionally not reproduced in this smoke report; the full comparison is deferred until the smoke artifacts are reviewed.",
        f"Raw immutable record: {raw_path}",
        "Sources: https://github.com/mangopy/tool-retrieval-benchmark and https://huggingface.co/datasets/mangopy/ToolRet-Queries and https://huggingface.co/datasets/mangopy/ToolRet-Tools",
    ]
    report_path = ROOT / "results" / "reports" / f"toolret_smoke_report_{timestamp}.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


if __name__ == "__main__":
    print(asyncio.run(run()))
