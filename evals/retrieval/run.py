import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from evals.common.metrics import score_context
from evals.common.token_count import count_schema_tokens
from evals.datasets.internal import DATASET_PATH, load_tasks, sha256_file, validate_tasks
from evals.methods.all_tools import AllTools
from evals.methods.bm25 import BM25
from evals.methods.dense import DenseTopK, MODEL_NAME
from evals.methods.octopus import OctopusMethod
from brown_octopus import Octopus
from brown_octopus.index_store import load_embeddings, load_tools


INDEX_PATH = Path("data/indexes/default")
RAW_DIR = Path("results/raw")
REPORT_DIR = Path("results/reports")


def canonical_hash(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def reset_method(method) -> None:
    if isinstance(method, OctopusMethod):
        method.octopus.reset()


def task_record(task, method, tools, tokenizer) -> dict:
    reset_method(method)
    tool_map = {tool["name"]: tool for tool in tools}
    all_schema_tokens = count_schema_tokens(tools, tokenizer)
    turn_records = []
    union_ids = []

    for turn_number, turn in enumerate(task["turns"], start=1):
        result = method.retrieve(turn["query"], tools)
        exposed_tools = [tool_map[tool_id] for tool_id in result.tool_ids]
        selected_schema_tokens = count_schema_tokens(exposed_tools, tokenizer)
        metrics = score_context(
            result.tool_ids,
            turn.get("required_tools", []),
            turn.get("supporting_tools", []),
            len(tools),
        )
        for tool_id in result.tool_ids:
            if tool_id not in union_ids:
                union_ids.append(tool_id)
        turn_records.append(
            {
                "turn": turn_number,
                "query": turn["query"],
                "tool_ids": result.tool_ids,
                "scores": result.scores,
                "latency_ms": result.latency_ms,
                "schema_tokens": selected_schema_tokens,
                "all_schema_tokens": all_schema_tokens,
                "schema_token_reduction": 1 - selected_schema_tokens / all_schema_tokens,
                "metrics": metrics,
                "metadata": result.metadata,
            }
        )

    union_tools = [tool_map[tool_id] for tool_id in union_ids]
    union_schema_tokens = count_schema_tokens(union_tools, tokenizer)
    union_metrics = score_context(
        union_ids,
        task["required_tools"],
        task["supporting_tools"],
        len(tools),
    )
    exposed_counts = [turn["metrics"]["tool_count"] for turn in turn_records]
    return {
        "task_id": task["id"],
        "category": task["category"],
        "method": method.name,
        "turns": turn_records,
        "metrics": {
            "required_recall": union_metrics["required_recall"],
            "complete_required": union_metrics["complete_required"],
            "supporting_recall": union_metrics["supporting_recall"],
            "tool_count_union": len(union_ids),
            "context_reduction_union": union_metrics["context_reduction"],
            "mean_exposed_tools_per_turn": statistics.mean(exposed_counts),
            "max_exposed_tools_per_turn": max(exposed_counts),
            "union_schema_tokens": union_schema_tokens,
        },
    }


def warmup_and_measure(methods, tasks, tools, tokenizer, warmup_runs, measured_runs):
    latency = {}
    sample_turns = [turn for task in tasks for turn in task["turns"]][:5]
    for method in methods:
        reset_method(method)
        for index in range(warmup_runs):
            turn = sample_turns[index % len(sample_turns)]
            method.retrieve(turn["query"], tools)

        measured = []
        for _ in range(measured_runs):
            reset_method(method)
            for task in tasks:
                for turn in task["turns"]:
                    measured.append(method.retrieve(turn["query"], tools).latency_ms)
        latency[method.name] = {
            "warmup_runs": warmup_runs,
            "measured_runs": measured_runs,
            "measured_call_count": len(measured),
            "mean_ms": statistics.mean(measured),
            "median_ms": statistics.median(measured),
            "p95_ms": percentile(measured, 0.95),
            "stdev_ms": statistics.stdev(measured) if len(measured) > 1 else 0.0,
        }
    return latency


def summary_rows(records: list[dict], latency: dict) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["method"]].append(record)

    rows = []
    for method, method_records in sorted(grouped.items()):
        turns = [turn for record in method_records for turn in record["turns"]]
        tool_counts = [turn["metrics"]["tool_count"] for turn in turns]
        rows.append(
            {
                "method": method,
                "task_count": len(method_records),
                "turn_count": len(turns),
                "required_recall": statistics.mean(turn["metrics"]["required_recall"] for turn in turns),
                "complete_required_rate": statistics.mean(turn["metrics"]["complete_required"] for turn in turns),
                "supporting_recall": statistics.mean(turn["metrics"]["supporting_recall"] for turn in turns),
                "mean_exposed_tools_per_turn": statistics.mean(tool_counts),
                "median_exposed_tools_per_turn": statistics.median(tool_counts),
                "p90_exposed_tools_per_turn": percentile(tool_counts, 0.90),
                "p95_exposed_tools_per_turn": percentile(tool_counts, 0.95),
                "max_exposed_tools_per_turn": max(tool_counts),
                "context_reduction_per_turn": statistics.mean(turn["metrics"]["context_reduction"] for turn in turns),
                "mean_schema_tokens_per_turn": statistics.mean(turn["schema_tokens"] for turn in turns),
                "schema_token_reduction": statistics.mean(turn["schema_token_reduction"] for turn in turns),
                "mean_unique_tools_per_task": statistics.mean(record["metrics"]["tool_count_union"] for record in method_records),
                "mean_union_schema_tokens_per_task": statistics.mean(record["metrics"]["union_schema_tokens"] for record in method_records),
                "latency_mean_ms": latency[method]["mean_ms"],
                "latency_median_ms": latency[method]["median_ms"],
                "latency_p95_ms": latency[method]["p95_ms"],
                "latency_stdev_ms": latency[method]["stdev_ms"],
            }
        )
    return rows


def category_rows(records: list[dict], latency: dict) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["category"], record["method"])].append(record)
    rows = []
    for (category, method), category_records in sorted(grouped.items()):
        turns = [turn for record in category_records for turn in record["turns"]]
        rows.append(
            {
                "category": category,
                "method": method,
                "task_count": len(category_records),
                "turn_count": len(turns),
                "required_recall": statistics.mean(turn["metrics"]["required_recall"] for turn in turns),
                "complete_required_rate": statistics.mean(turn["metrics"]["complete_required"] for turn in turns),
                "supporting_recall": statistics.mean(turn["metrics"]["supporting_recall"] for turn in turns),
                "mean_exposed_tools_per_turn": statistics.mean(turn["metrics"]["tool_count"] for turn in turns),
                "context_reduction_per_turn": statistics.mean(turn["metrics"]["context_reduction"] for turn in turns),
                "mean_schema_tokens_per_turn": statistics.mean(turn["schema_tokens"] for turn in turns),
                "schema_token_reduction": statistics.mean(turn["schema_token_reduction"] for turn in turns),
                "latency_mean_ms": latency[method]["mean_ms"],
                "latency_p95_ms": latency[method]["p95_ms"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(warmup_runs: int = 5, measured_runs: int = 5) -> tuple[Path, Path]:
    tools = load_tools(INDEX_PATH)
    embeddings = load_embeddings(INDEX_PATH)
    tasks = load_tasks(DATASET_PATH, expected_count=50)
    validate_tasks(tasks, tools)

    octopus = Octopus(index_path=INDEX_PATH)
    import asyncio

    asyncio.run(octopus.initialize())
    import brown_octopus.retriever as runtime_retriever

    dense_methods = [
        DenseTopK(embeddings=embeddings, model=runtime_retriever.model, k=k)
        for k in (4, 8, 16)
    ]
    for method in dense_methods:
        method.initialize(tools)
    bm25_methods = [BM25(k) for k in (4, 8, 16)]
    for method in bm25_methods:
        method.initialize(tools)
    methods = [AllTools(), *bm25_methods, *dense_methods, OctopusMethod(octopus)]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    latency = warmup_and_measure(
        methods, tasks, tools, tokenizer, warmup_runs, measured_runs
    )
    records = [
        task_record(task, method, tools, tokenizer)
        for task in tasks
        for method in methods
    ]

    commit = git_commit()
    experiment = {
        "experiment_id": datetime.now(timezone.utc).strftime("retrieval_%Y%m%dT%H%M%S_%fZ"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "dataset": str(DATASET_PATH),
        "dataset_version": "octopus_capability_benchmark_v2_milestone2",
        "dataset_hash": sha256_file(DATASET_PATH),
        "tool_universe": str(INDEX_PATH / "tools.json"),
        "tool_universe_hash": sha256_file(INDEX_PATH / "tools.json"),
        "tool_count": len(tools),
        "embedding_model": MODEL_NAME,
        "tool_representation": "name + description",
        "intent_analyzer": "spaCy en_core_web_trf",
        "index_version": 1,
        "dense_k_values": [4, 8, 16],
        "bm25_k_values": [4, 8, 16],
        "method": "brown_octopus_v3",
        "octopus_selection": "min_4_bounded_max_gap",
        "octopus_min_tools": 4,
        "octopus_max_tools": 16,
        "octopus_min_gap_percent": 2.0,
        "octopus_ttl": 8,
        "octopus_max_active_tools": 30,
        "latency_warmup_runs": warmup_runs,
        "latency_measured_runs": measured_runs,
        "machine": platform.platform(),
        "python": platform.python_version(),
        "latency": latency,
        "records": records,
    }

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{experiment['experiment_id']}.json"
    with raw_path.open("x", encoding="utf-8") as file:
        json.dump(experiment, file, indent=2)
    write_csv(REPORT_DIR / "retrieval_summary.csv", summary_rows(records, latency))
    write_csv(REPORT_DIR / "category_summary.csv", category_rows(records, latency))
    return raw_path, REPORT_DIR / "retrieval_summary.csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--measured-runs", type=int, default=5)
    args = parser.parse_args()
    raw, summary = run(args.warmup_runs, args.measured_runs)
    print(f"Raw results: {raw}")
    print(f"Summary: {summary}")
