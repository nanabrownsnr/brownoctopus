import argparse
import csv
import hashlib
import json
import math
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
from evals.methods.ablation import EvaluationOctopusMethod, OctopusFullMethod
from brown_octopus import Octopus
from brown_octopus.index_store import load_tools


INDEX_PATH = Path("data/indexes/default")
RAW_DIR = Path("results/raw")
REPORT_DIR = Path("results/reports")


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]


def names(values) -> str:
    return ";".join(values)


def task_record(task, method, tools, tokenizer) -> dict:
    method.reset()
    tool_map = {tool["name"]: tool for tool in tools}
    all_schema_tokens = count_schema_tokens(tools, tokenizer)
    turns = []
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
        turns.append(
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
    union_metrics = score_context(
        union_ids,
        task["required_tools"],
        task["supporting_tools"],
        len(tools),
    )
    counts = [turn["metrics"]["tool_count"] for turn in turns]
    return {
        "task_id": task["id"],
        "category": task["category"],
        "method": method.name,
        "turns": turns,
        "metrics": {
            "required_recall": union_metrics["required_recall"],
            "complete_required": union_metrics["complete_required"],
            "supporting_recall": union_metrics["supporting_recall"],
            "tool_count_union": len(union_ids),
            "context_reduction_union": union_metrics["context_reduction"],
            "mean_exposed_tools_per_turn": statistics.mean(counts),
        },
    }


def build_methods(octopus: Octopus):
    return [
        OctopusFullMethod(octopus),
        EvaluationOctopusMethod("octopus_no_intent", "no_intent"),
        EvaluationOctopusMethod("octopus_fixed_k_1", "fixed_k", 1),
        EvaluationOctopusMethod("octopus_fixed_k_2", "fixed_k", 2),
        EvaluationOctopusMethod("octopus_fixed_k_4", "fixed_k", 4),
        EvaluationOctopusMethod("octopus_fixed_k_8", "fixed_k", 8),
        EvaluationOctopusMethod("octopus_fixed_k_16", "fixed_k", 16),
        EvaluationOctopusMethod("octopus_no_memory", "no_memory"),
    ]


def aggregate_rows(records: list[dict], tools, mode: str) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        key = record["method"] if mode == "overall" else (record["category"], record["method"])
        grouped[key].append(record)
    rows = []
    for key, selected_records in sorted(grouped.items(), key=lambda item: str(item[0])):
        turns = [turn for record in selected_records for turn in record["turns"]]
        counts = [turn["metrics"]["tool_count"] for turn in turns]
        row = {
            "method": key if mode == "overall" else key[1],
            "task_count": len(selected_records),
            "turn_count": len(turns),
            "required_recall": statistics.mean(turn["metrics"]["required_recall"] for turn in turns),
            "complete_required_rate": statistics.mean(turn["metrics"]["complete_required"] for turn in turns),
            "supporting_recall": statistics.mean(turn["metrics"]["supporting_recall"] for turn in turns),
            "mean_exposed_tools_per_turn": statistics.mean(counts),
            "median_exposed_tools_per_turn": statistics.median(counts),
            "p90_exposed_tools_per_turn": percentile(counts, 0.90),
            "p95_exposed_tools_per_turn": percentile(counts, 0.95),
            "max_exposed_tools_per_turn": max(counts),
            "mean_schema_tokens_per_turn": statistics.mean(turn["schema_tokens"] for turn in turns),
            "schema_token_reduction": statistics.mean(turn["schema_token_reduction"] for turn in turns),
            "mean_unique_tools_per_task": statistics.mean(record["metrics"]["tool_count_union"] for record in selected_records),
        }
        if mode != "overall":
            row["category"] = key[0]
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def full_records(records):
    return [record for record in records if record["method"] == "octopus_full"]


def selection_distribution(records: list[dict]) -> list[dict]:
    sizes = []
    for record in full_records(records):
        for turn in record["turns"]:
            sizes.extend(
                detail["cutoff"]
                for detail in turn["metadata"]["intent_details"]
            )
    return [
        {"statistic": "mean", "value": statistics.mean(sizes)},
        {"statistic": "median", "value": statistics.median(sizes)},
        {"statistic": "p90", "value": percentile(sizes, 0.90)},
        {"statistic": "p95", "value": percentile(sizes, 0.95)},
        {"statistic": "max", "value": max(sizes)},
        *[
            {"statistic": f"k_{k}", "value": sizes.count(k)}
            for k in range(1, 17)
        ],
    ]


def compound_intent_rows(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        if record["category"] != "compound":
            continue
        for turn in record["turns"]:
            details = turn["metadata"]["intent_details"]
            rows.append(
                {
                    "task_id": record["task_id"],
                    "method": record["method"],
                    "turn": turn["turn"],
                    "query": turn["query"],
                    "detected_intent_count": len(details),
                    "intent_number": "|".join(str(detail["intent_number"]) for detail in details),
                    "selected_tools_per_intent": "|".join(str(detail["cutoff"]) for detail in details),
                    "intent_texts": " || ".join(detail["text"] for detail in details),
                }
            )
    return rows


def state_rows(records: list[dict], tasks_by_id: dict) -> list[dict]:
    lookup = {(record["task_id"], record["method"]): record for record in records}
    rows = []
    for task_id, full in sorted((key[0], value) for key, value in lookup.items() if key[1] == "octopus_full"):
        if full["category"] != "multi_turn":
            continue
        no_memory = lookup[(task_id, "octopus_no_memory")]
        task = tasks_by_id[task_id]
        for full_turn, no_memory_turn, task_turn in zip(full["turns"], no_memory["turns"], task["turns"]):
            def state_field(turn, field):
                return names(turn["metadata"].get(field, []))

            required = set(task_turn.get("required_tools", []))
            supporting = set(task_turn.get("supporting_tools", []))
            full_exposed = set(full_turn["tool_ids"])
            no_memory_exposed = set(no_memory_turn["tool_ids"])

            rows.append(
                {
                    "task_id": task_id,
                    "turn": full_turn["turn"],
                    "query": full_turn["query"],
                    "full_newly_retrieved": state_field(full_turn, "newly_retrieved_tool_ids"),
                    "full_retained": state_field(full_turn, "retained_tool_ids"),
                    "full_expired": state_field(full_turn, "expired_tool_ids"),
                    "full_exposed": names(full_turn["tool_ids"]),
                    "full_required_hits": names(sorted(required & full_exposed)),
                    "full_required_misses": names(sorted(required - full_exposed)),
                    "full_supporting_hits": names(sorted(supporting & full_exposed)),
                    "full_supporting_misses": names(sorted(supporting - full_exposed)),
                    "full_required_recall": full_turn["metrics"]["required_recall"],
                    "full_supporting_recall": full_turn["metrics"]["supporting_recall"],
                    "no_memory_newly_retrieved": state_field(no_memory_turn, "newly_retrieved_tool_ids"),
                    "no_memory_retained": state_field(no_memory_turn, "retained_tool_ids"),
                    "no_memory_expired": state_field(no_memory_turn, "expired_tool_ids"),
                    "no_memory_exposed": names(no_memory_turn["tool_ids"]),
                    "no_memory_required_hits": names(sorted(required & no_memory_exposed)),
                    "no_memory_required_misses": names(sorted(required - no_memory_exposed)),
                    "no_memory_supporting_hits": names(sorted(supporting & no_memory_exposed)),
                    "no_memory_supporting_misses": names(sorted(supporting - no_memory_exposed)),
                    "no_memory_required_recall": no_memory_turn["metrics"]["required_recall"],
                    "no_memory_supporting_recall": no_memory_turn["metrics"]["supporting_recall"],
                }
            )
    return rows


def failure_rows(records: list[dict], tasks_by_id: dict, universe_count: int) -> list[dict]:
    rows = []
    for record in full_records(records):
        task = tasks_by_id[record["task_id"]]
        for turn_record, turn_task in zip(record["turns"], task["turns"]):
            exposed = set(turn_record["tool_ids"])
            for field, label in (("required_tools", "required"), ("supporting_tools", "supporting")):
                for missed in set(turn_task.get(field, [])) - exposed:
                    details = turn_record["metadata"]["intent_details"]
                    rank_matches = []
                    for detail in details:
                        match = next((item for item in detail["ranking"] if item["name"] == missed), None)
                        if match:
                            rank_matches.append((match["rank"], detail))
                    rank, nearest = min(rank_matches, key=lambda item: item[0]) if rank_matches else (None, None)
                    cutoffs = {str(detail["intent_number"]): detail["cutoff"] for detail in details}
                    selected_before_memory = missed in set(turn_record["metadata"].get("selected_before_memory_ids", []))
                    if selected_before_memory:
                        cause = "state_or_ttl"
                    elif not details:
                        cause = "intent_decomposition"
                    elif rank is None or rank > 16:
                        cause = "retrieval_ranking"
                    else:
                        cause = "cutoff"
                    rows.append(
                        {
                            "task_id": record["task_id"],
                            "category": record["category"],
                            "turn": turn_record["turn"],
                            "query": turn_record["query"],
                            "miss_type": label,
                            "missed_tool": missed,
                            "underlying_dense_rank": rank,
                            "nearest_intent": nearest["text"] if nearest else "",
                            "detected_intents": " || ".join(detail["text"] for detail in details),
                            "max_gap_cutoffs": json.dumps(cutoffs, sort_keys=True),
                            "selected_before_memory": selected_before_memory,
                            "cause": cause,
                        }
                    )
    return rows


def hidden_dependency_rows(records: list[dict], tasks_by_id: dict) -> list[dict]:
    rows = []
    for record in full_records(records):
        task = tasks_by_id[record["task_id"]]
        if task["category"] != "hidden_dependency":
            continue
        for turn_record, turn_task in zip(record["turns"], task["turns"]):
            details = turn_record["metadata"]["intent_details"]
            for supporting in turn_task.get("supporting_tools", []):
                matches = []
                for detail in details:
                    match = next((item for item in detail["ranking"] if item["name"] == supporting), None)
                    if match:
                        matches.append({"intent": detail["text"], "rank": match["rank"], "cutoff": detail["cutoff"]})
                best = min((item["rank"] for item in matches), default=None)
                rows.append(
                    {
                        "task_id": record["task_id"],
                        "query": turn_record["query"],
                        "supporting_tool": supporting,
                        "best_dense_rank": best,
                        "rank_and_cutoff_by_intent": json.dumps(matches, sort_keys=True),
                        "selected_before_memory": supporting in set(turn_record["metadata"].get("selected_before_memory_ids", [])),
                    }
                )
    return rows


def single_capability_rows(records: list[dict], tasks_by_id: dict) -> list[dict]:
    """Compare the frozen Full result with the already-frozen Milestone 2 dense runs."""
    dense_raw = []
    for path in sorted(RAW_DIR.glob("retrieval_*.json"), reverse=True):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dataset_version") == "octopus_capability_benchmark_v2_milestone2":
            dense_raw = payload.get("records", [])
            if any(record.get("method") == "dense_4" for record in dense_raw):
                break
    dense_lookup = {(record["task_id"], record["method"]): record for record in dense_raw}
    full_lookup = {(record["task_id"], record["method"]): record for record in records}
    rows = []
    for task_id, task in tasks_by_id.items():
        if task["category"] != "single_capability":
            continue
        required = set(task["required_tools"])
        full_turn = full_lookup[(task_id, "octopus_full")]["turns"][0]
        row = {
            "task_id": task_id,
            "query": task["query"],
            "required_tools": names(sorted(required)),
            "octopus_full_required_hits": names(sorted(required & set(full_turn["tool_ids"]))),
            "octopus_full_required_misses": names(sorted(required - set(full_turn["tool_ids"]))),
        }
        for method_name in ("dense_4", "dense_8", "dense_16"):
            dense_turn = dense_lookup[(task_id, method_name)]["turns"][0]
            row[f"{method_name}_required_hits"] = names(sorted(required & set(dense_turn["tool_ids"])))
            row[f"{method_name}_required_misses"] = names(sorted(required - set(dense_turn["tool_ids"])))
        rows.append(row)
    return rows


def run() -> Path:
    tools = load_tools(INDEX_PATH)
    tasks = load_tasks(DATASET_PATH, expected_count=50)
    validate_tasks(tasks, tools)
    tasks_by_id = {task["id"]: task for task in tasks}
    octopus = Octopus(index_path=INDEX_PATH)
    import asyncio

    asyncio.run(octopus.initialize())
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
    methods = build_methods(octopus)
    records = [
        task_record(task, method, tools, tokenizer)
        for task in tasks
        for method in methods
    ]
    commit = git_commit()
    payload = {
        "experiment_id": datetime.now(timezone.utc).strftime("ablation_%Y%m%dT%H%M%S_%fZ"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "dataset": str(DATASET_PATH),
        "dataset_version": "octopus_capability_benchmark_v2_milestone2",
        "dataset_hash": sha256_file(DATASET_PATH),
        "tool_universe": str(INDEX_PATH / "tools.json"),
        "tool_universe_hash": sha256_file(INDEX_PATH / "tools.json"),
        "tool_count": len(tools),
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "tool_representation": "name + description",
        "octopus_max_tools": 16,
        "octopus_min_gap_percent": 2.0,
        "octopus_ttl": 8,
        "octopus_max_active_tools": 30,
        "records": records,
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{payload['experiment_id']}.json"
    with raw_path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    write_csv(REPORT_DIR / "ablation_summary.csv", aggregate_rows(records, tools, "overall"))
    write_csv(REPORT_DIR / "ablation_category_summary.csv", aggregate_rows(records, tools, "category"))
    write_csv(REPORT_DIR / "ablation_selection_distribution.csv", selection_distribution(records))
    write_csv(REPORT_DIR / "ablation_compound_intents.csv", compound_intent_rows(records))
    write_csv(REPORT_DIR / "ablation_multiturn_state.csv", state_rows(records, tasks_by_id))
    write_csv(REPORT_DIR / "ablation_failure_analysis.csv", failure_rows(records, tasks_by_id, len(tools)))
    write_csv(REPORT_DIR / "ablation_hidden_dependency_ranks.csv", hidden_dependency_rows(records, tasks_by_id))
    write_csv(REPORT_DIR / "ablation_single_capability_analysis.csv", single_capability_rows(records, tasks_by_id))
    return raw_path


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    print(f"Raw results: {run()}")
