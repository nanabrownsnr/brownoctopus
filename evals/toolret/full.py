"""Full, non-tuned ToolRet run using the smoke-tested configuration."""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

from evals.common.token_count import count_schema_tokens
from evals.methods.octopus import OctopusMethod
from evals.toolret.adapter import load_toolret_queries, load_toolret_tools
from evals.toolret.index import MODEL_NAME, canonical_hash
from evals.toolret.metrics import evaluate_toolret_result
from octopus import Octopus
from octopus.index_store import load_embeddings, load_metadata, load_tools


ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "data" / "external" / "toolret-release"
INDEX = ROOT / "data" / "indexes" / "toolret_smoke"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[int], fraction: float) -> float:
    values = sorted(values)
    index = min(len(values) - 1, int(round((len(values) - 1) * fraction)))
    return float(values[index])


async def run() -> Path:
    if not (INDEX / "tools.json").exists() or not (INDEX / "embeddings.pt").exists():
        raise RuntimeError("Completed ToolRet index is required before launching the full run")
    tools, tool_sources = load_toolret_tools(RELEASE)
    examples, query_sources = load_toolret_queries(RELEASE, datasets=None)
    indexed_tools = load_tools(INDEX)
    tool_by_id = {tool["name"]: tool for tool in indexed_tools}
    embeddings = load_embeddings(INDEX)
    index_metadata = load_metadata(INDEX)
    if len(indexed_tools) != len(tools) or canonical_hash(indexed_tools) != canonical_hash(tools):
        raise RuntimeError("Persisted ToolRet index does not match the official loaded tool corpus")

    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    tokenizer = model.tokenizer
    all_schema_tokens = count_schema_tokens(indexed_tools, tokenizer)
    octopus = Octopus(index_path=INDEX)
    await octopus.initialize()
    octopus_method = OctopusMethod(octopus)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    experiment_id = f"toolret_full_{timestamp}"
    raw_dir = ROOT / "results" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{experiment_id}.jsonl"
    config_path = raw_dir / f"{experiment_id}.config.json"
    if raw_path.exists() or config_path.exists():
        raise RuntimeError("Refusing to overwrite an existing full-run artifact")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    config = {
        "experiment_id": experiment_id,
        "timestamp": timestamp,
        "git_commit": commit,
        "v3_freeze_commit": "72b200937486ca7413e1bfbd0e2abc6c3787a3aa",
        "condition": "without_instruction",
        "methods": ["raw_qwen_top10", "brown_octopus_v3"],
        "embedding_model": MODEL_NAME,
        "embedding_representation": "name + description",
        "selector": {"min_tools": 4, "max_tools": 16, "min_gap_percent": 2.0},
        "memory": {"ttl": 8, "active_cap": 30, "reset_between_examples": True},
        "toolret_code_commit": "c4181d914a227134705ecb6bab13fbd92ccd2938",
        "tool_sources": {key: sha256(Path(value)) for key, value in tool_sources.items()},
        "query_sources": {key: sha256(Path(value)) for key, value in query_sources.items()},
        "tool_universe_hash": canonical_hash(tools),
        "tool_count": len(tools),
        "query_count": len(examples),
        "index_metadata": index_metadata,
        "machine": {"platform": platform.platform(), "device": str(model.device)},
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    method_records: dict[str, list[dict]] = {"raw_qwen_top10": [], "brown_octopus_v3": []}
    with raw_path.open("x", encoding="utf-8") as output:
        for number, example in enumerate(examples, start=1):
            query_embedding = model.encode(example.query, convert_to_tensor=True, normalize_embeddings=True)
            scores = query_embedding @ embeddings.T
            raw_indices = scores.argsort(descending=True)[:10].tolist()
            raw_ids = [indexed_tools[index]["name"] for index in raw_indices]
            raw_tools = [indexed_tools[index] for index in raw_indices]
            raw_record = {
                "query_id": example.query_id,
                "dataset": example.dataset,
                "category": example.category,
                "query": example.query,
                "gold_tool_ids": list(example.labels),
                "tool_ids": raw_ids,
                "metrics": evaluate_toolret_result(raw_ids, example.labels),
                "tool_count": len(raw_ids),
                "schema_tokens": count_schema_tokens(raw_tools, tokenizer),
                "method": "raw_qwen_top10",
            }

            octopus_method.reset()
            v3 = octopus_method.retrieve(example.query, indexed_tools)
            v3_tools = [tool_by_id[tool_id] for tool_id in v3.tool_ids]
            v3_record = {
                "query_id": example.query_id,
                "dataset": example.dataset,
                "category": example.category,
                "query": example.query,
                "gold_tool_ids": list(example.labels),
                "tool_ids": v3.tool_ids,
                "metrics": evaluate_toolret_result(v3.tool_ids, example.labels),
                "tool_count": len(v3.tool_ids),
                "schema_tokens": count_schema_tokens(v3_tools, tokenizer),
                "metadata": v3.metadata,
                "method": "brown_octopus_v3",
                "state_reset_before_query": True,
            }
            output.write(json.dumps({"raw_qwen_top10": raw_record, "brown_octopus_v3": v3_record}) + "\n")
            output.flush()
            method_records["raw_qwen_top10"].append(raw_record)
            method_records["brown_octopus_v3"].append(v3_record)
            if number % 25 == 0:
                print(f"Completed {number}/{len(examples)} queries", flush=True)

    summaries = {}
    for method, records in method_records.items():
        counts = [record["tool_count"] for record in records]
        summaries[method] = {
            "query_count": len(records),
            **{key: sum(record["metrics"][key] for record in records) / len(records) for key in ("ndcg@10", "recall@10", "completeness@10")},
            "mean_tools": sum(counts) / len(counts),
            "median_tools": percentile(counts, 0.50),
            "p90_tools": percentile(counts, 0.90),
            "p95_tools": percentile(counts, 0.95),
            "max_tools": max(counts),
            "mean_schema_tokens": sum(record["schema_tokens"] for record in records) / len(records),
            "schema_token_reduction_vs_all": 1.0 - (sum(record["schema_tokens"] for record in records) / len(records)) / all_schema_tokens,
            "selected_k_distribution": {str(k): counts.count(k) for k in sorted(set(counts))},
        }
    summary_path = ROOT / "results" / "reports" / f"{experiment_id}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"config": config, "summary": summaries}, indent=2), encoding="utf-8")
    return summary_path


if __name__ == "__main__":
    print(asyncio.run(run()))
