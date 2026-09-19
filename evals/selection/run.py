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
from evals.methods.selection import (
    SelectionMethod,
    bounded_max_gap,
    ranked_score_knee,
    relative_threshold,
)
from octopus.index_store import load_tools
from octopus import Octopus


INDEX_PATH = Path("data/indexes/default")
SPLIT_PATH = Path("data/evals/selection_milestone4_split.json")
RAW_DIR = Path("results/raw")
REPORT_DIR = Path("results/reports")
THRESHOLD_GRID = [0.99, 0.98, 0.97, 0.95, 0.93, 0.90, 0.85, 0.80]


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]


def names(values) -> str:
    return ";".join(values)


def method_set(relative_alpha: float | None = None) -> list[SelectionMethod]:
    methods = [
        SelectionMethod("fixed_k_1", lambda ranking: (ranking[:1], {"cutoff": min(1, len(ranking)), "reason": "fixed_k"})),
        SelectionMethod("fixed_k_2", lambda ranking: (ranking[:2], {"cutoff": min(2, len(ranking)), "reason": "fixed_k"})),
        SelectionMethod("fixed_k_4", lambda ranking: (ranking[:4], {"cutoff": min(4, len(ranking)), "reason": "fixed_k"})),
        SelectionMethod("fixed_k_8", lambda ranking: (ranking[:8], {"cutoff": min(8, len(ranking)), "reason": "fixed_k"})),
        SelectionMethod("fixed_k_16", lambda ranking: (ranking[:16], {"cutoff": min(16, len(ranking)), "reason": "fixed_k"})),
        SelectionMethod("bounded_max_gap", bounded_max_gap),
        SelectionMethod("ranked_score_knee", ranked_score_knee),
    ]
    if relative_alpha is not None:
        methods.append(
            SelectionMethod(
                f"relative_threshold_{relative_alpha:.2f}",
                lambda ranking, alpha=relative_alpha: relative_threshold(ranking, alpha),
                {"alpha": relative_alpha},
            )
        )
    return methods


def task_record(task, method, tools, tokenizer, split: str) -> dict:
    method.reset()
    tool_map = {tool["name"]: tool for tool in tools}
    all_schema_tokens = count_schema_tokens(tools, tokenizer)
    turns = []
    union_ids = []
    for turn_number, turn in enumerate(task["turns"], start=1):
        result = method.retrieve(turn["query"], tools)
        selected_tools = [tool_map[tool_id] for tool_id in result.tool_ids]
        schema_tokens = count_schema_tokens(selected_tools, tokenizer)
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
                "schema_tokens": schema_tokens,
                "all_schema_tokens": all_schema_tokens,
                "schema_token_reduction": 1 - schema_tokens / all_schema_tokens,
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
    return {
        "task_id": task["id"],
        "category": task["category"],
        "split": split,
        "method": method.name,
        "turns": turns,
        "metrics": {
            "required_recall": union_metrics["required_recall"],
            "complete_required": union_metrics["complete_required"],
            "supporting_recall": union_metrics["supporting_recall"],
            "tool_count_union": len(union_ids),
            "context_reduction_union": union_metrics["context_reduction"],
        },
    }


def run_records(tasks, methods, tools, tokenizer, split: str) -> list[dict]:
    return [
        task_record(task, method, tools, tokenizer, split)
        for task in tasks
        for method in methods
    ]


def summarize(records: list[dict], category: str | None = None) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        if category is not None and record["category"] != category:
            continue
        grouped[(record["split"], record["method"])].append(record)
    rows = []
    for (split, method), selected_records in sorted(grouped.items()):
        turns = [turn for record in selected_records for turn in record["turns"]]
        counts = [turn["metrics"]["tool_count"] for turn in turns]
        rows.append(
            {
                "split": split,
                "method": method,
                "category": category or "all",
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
        )
    return rows


def selection_distribution(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        for turn in record["turns"]:
            for detail in turn["metadata"]["intent_details"]:
                grouped[(record["split"], record["method"])].append(detail["cutoff"])
    rows = []
    for (split, method), sizes in sorted(grouped.items()):
        rows.extend(
            [
                {"split": split, "method": method, "statistic": "mean", "value": statistics.mean(sizes)},
                {"split": split, "method": method, "statistic": "median", "value": statistics.median(sizes)},
                {"split": split, "method": method, "statistic": "p90", "value": percentile(sizes, 0.90)},
                {"split": split, "method": method, "statistic": "p95", "value": percentile(sizes, 0.95)},
                {"split": split, "method": method, "statistic": "max", "value": max(sizes)},
            ]
            + [
                {"split": split, "method": method, "statistic": f"k_{k}", "value": sizes.count(k)}
                for k in range(1, 17)
            ]
        )
    return rows


def score_context_around(details: dict) -> str:
    return json.dumps(
        [
            {
                "rank": item["rank"],
                "name": item["name"],
                "score": item["score"],
            }
            for item in details["scores_around_cutoff"]
        ],
        sort_keys=True,
    )


def failure_analysis(records: list[dict], tasks_by_id: dict) -> list[dict]:
    dynamic_records = [
        record
        for record in records
        if record["method"] == "bounded_max_gap"
        or record["method"] == "ranked_score_knee"
        or record["method"].startswith("relative_threshold_")
    ]
    rows = []
    for record in dynamic_records:
        task = tasks_by_id[record["task_id"]]
        for turn_record, task_turn in zip(record["turns"], task["turns"]):
            exposed = set(turn_record["tool_ids"])
            misses = [
                ("required", tool_id)
                for tool_id in set(task_turn.get("required_tools", [])) - exposed
            ] + [
                ("supporting", tool_id)
                for tool_id in set(task_turn.get("supporting_tools", [])) - exposed
            ]
            for tool_role, required_tool in misses:
                details = turn_record["metadata"]["intent_details"]
                matches = []
                for detail in details:
                    match = next((item for item in detail["ranking"] if item["name"] == required_tool), None)
                    if match:
                        matches.append((match["rank"], match["score"], detail))
                rank, score, nearest = min(matches, key=lambda item: item[0]) if matches else (None, None, None)
                selected = required_tool in set(turn_record["metadata"].get("selected_before_memory_ids", []))
                if selected:
                    reason = "state_or_ttl"
                elif rank is None or rank > 16:
                    reason = "ranking_failure"
                else:
                    reason = "selection_failure"
                rows.append(
                    {
                        "split": record["split"],
                        "method": record["method"],
                        "task_id": record["task_id"],
                        "category": record["category"],
                        "tool_role": tool_role,
                        "query": turn_record["query"],
                        "detected_intents": ";".join(detail["text"] for detail in details),
                        "required_tool": required_tool,
                        "dense_rank": rank,
                        "dense_score": score,
                        "selected_cutoff": nearest["cutoff"] if nearest else "",
                        "intent_cutoffs": json.dumps(
                            [
                                {"intent": detail["text"], "cutoff": detail["cutoff"]}
                                for detail in details
                            ],
                            sort_keys=True,
                        ),
                        "scores_around_cutoff": score_context_around(nearest) if nearest else "",
                        "reason": reason,
                    }
                )
    return rows


def single_analysis(records: list[dict], tasks_by_id: dict) -> list[dict]:
    rows = []
    for record in records:
        if record["category"] != "single_capability" or record["split"] != "held_out":
            continue
        task = tasks_by_id[record["task_id"]]
        required = set(task["required_tools"])
        turn = record["turns"][0]
        rows.append(
            {
                "task_id": record["task_id"],
                "method": record["method"],
                "query": task["query"],
                "required_tool": names(sorted(required)),
                "required_hits": names(sorted(required & set(turn["tool_ids"]))),
                "required_misses": names(sorted(required - set(turn["tool_ids"]))),
                "intent_details": json.dumps(
                    [
                        {
                            "text": detail["text"],
                            "cutoff": detail["cutoff"],
                            "selected": detail["selected_tool_ids"],
                        }
                        for detail in turn["metadata"]["intent_details"]
                    ],
                    sort_keys=True,
                ),
            }
        )
    return rows


def select_threshold(dev_records: list[dict]) -> tuple[float, list[dict]]:
    summaries = []
    for alpha in THRESHOLD_GRID:
        method_name = f"relative_threshold_{alpha:.2f}"
        method_records = [record for record in dev_records if record["method"] == method_name]
        turns = [turn for record in method_records for turn in record["turns"]]
        summaries.append(
            {
                "alpha": alpha,
                "complete_required_rate": statistics.mean(turn["metrics"]["complete_required"] for turn in turns),
                "mean_exposed_tools_per_turn": statistics.mean(turn["metrics"]["tool_count"] for turn in turns),
                "supporting_recall": statistics.mean(turn["metrics"]["supporting_recall"] for turn in turns),
            }
        )
    best = max(
        summaries,
        key=lambda row: (
            row["complete_required_rate"],
            -row["mean_exposed_tools_per_turn"],
            row["supporting_recall"],
        ),
    )
    return best["alpha"], summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run() -> Path:
    tools = load_tools(INDEX_PATH)
    tasks = load_tasks(DATASET_PATH, expected_count=50)
    validate_tasks(tasks, tools)
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    development_ids = set(split["development_task_ids"])
    held_out_ids = set(split["held_out_task_ids"])
    if development_ids | held_out_ids != {task["id"] for task in tasks}:
        raise ValueError("Selection split does not cover the frozen benchmark exactly.")
    tasks_by_id = {task["id"]: task for task in tasks}
    development_tasks = [task for task in tasks if task["id"] in development_ids]
    held_out_tasks = [task for task in tasks if task["id"] in held_out_ids]

    octopus = Octopus(index_path=INDEX_PATH)
    import asyncio

    asyncio.run(octopus.initialize())
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)

    calibration_methods = method_set()
    calibration_methods.append(
        SelectionMethod(
            "relative_threshold_0.00",
            lambda ranking, alpha=0.00: relative_threshold(ranking, alpha),
        )
    )
    # Generate one development record set for every preregistered threshold.
    for alpha in THRESHOLD_GRID:
        calibration_methods.append(
            SelectionMethod(
                f"relative_threshold_{alpha:.2f}",
                lambda ranking, alpha=alpha: relative_threshold(ranking, alpha),
            )
        )
    calibration_records = run_records(development_tasks, calibration_methods, tools, tokenizer, "development")
    selected_alpha, calibration_summary = select_threshold(calibration_records)

    methods = method_set(selected_alpha)
    records = run_records(development_tasks, methods, tools, tokenizer, "development")
    records.extend(run_records(held_out_tasks, methods, tools, tokenizer, "held_out"))
    commit = git_commit()
    payload = {
        "experiment_id": datetime.now(timezone.utc).strftime("selection_%Y%m%dT%H%M%S_%fZ"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "dataset": str(DATASET_PATH),
        "dataset_hash": sha256_file(DATASET_PATH),
        "split": str(SPLIT_PATH),
        "split_hash": sha256_file(SPLIT_PATH),
        "tool_universe": str(INDEX_PATH / "tools.json"),
        "tool_universe_hash": sha256_file(INDEX_PATH / "tools.json"),
        "tool_count": len(tools),
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "tool_representation": "name + description",
        "max_tools": 16,
        "min_gap_percent": 2.0,
        "ttl": 8,
        "active_cap": 30,
        "threshold_grid": THRESHOLD_GRID,
        "selected_relative_threshold_alpha": selected_alpha,
        "calibration_rule": "max development complete-required rate, then minimum mean exposed tools, then maximum supporting recall",
        "calibration_summary": calibration_summary,
        "machine": platform.platform(),
        "python": platform.python_version(),
        "records": records,
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{payload['experiment_id']}.json"
    with raw_path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    all_summary = summarize(records)
    heldout_summary = [row for row in all_summary if row["split"] == "held_out"]
    write_csv(REPORT_DIR / "selection_summary.csv", all_summary)
    write_csv(REPORT_DIR / "selection_heldout_summary.csv", heldout_summary)
    category_rows = []
    for category in sorted({task["category"] for task in tasks}):
        category_rows.extend(summarize(records, category))
    write_csv(REPORT_DIR / "selection_category_summary.csv", category_rows)
    write_csv(REPORT_DIR / "selection_k_distribution.csv", selection_distribution(records))
    write_csv(REPORT_DIR / "selection_recall_context.csv", heldout_summary)
    write_csv(REPORT_DIR / "selection_failure_analysis.csv", failure_analysis(records, tasks_by_id))
    write_csv(REPORT_DIR / "selection_single_capability_analysis.csv", single_analysis(records, tasks_by_id))
    return raw_path


if __name__ == "__main__":
    print(f"Raw results: {run()}")
