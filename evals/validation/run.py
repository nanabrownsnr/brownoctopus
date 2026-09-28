import csv
import asyncio
import json
import platform
import statistics
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from evals.common.metrics import score_context
from evals.common.token_count import serialize_tool
from evals.datasets.internal import sha256_file, validate_tasks
from evals.methods.selection import SelectionMethod, bounded_max_gap
from brown_octopus.index_store import load_tools
from brown_octopus import Octopus


DATASET_PATH = Path("data/evals/octopus_validation_benchmark_v1.json")
INDEX_PATH = Path("data/indexes/default")
RAW_DIR = Path("results/raw")
REPORT_DIR = Path("results/reports")


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))] if ordered else 0.0


def fixed_selector(k: int):
    def select(ranking):
        return ranking[:k], {"reason": "fixed_k", "cutoff": min(k, len(ranking)), "max_gap_cutoff": None}
    return select


def hybrid_selector(ranking):
    selected, metadata = bounded_max_gap(ranking)
    max_gap_k = metadata["cutoff"]
    cutoff = min(16, max(4, max_gap_k))
    return ranking[:cutoff], {
        "reason": "min_4_plus_bounded_max_gap",
        "cutoff": cutoff,
        "max_gap_cutoff": max_gap_k,
    }


def methods():
    return [
        SelectionMethod("fixed_k_4", fixed_selector(4)),
        SelectionMethod("fixed_k_8", fixed_selector(8)),
        SelectionMethod("bounded_max_gap", bounded_max_gap),
        SelectionMethod("min_4_plus_max_gap", hybrid_selector),
    ]


def evaluate_task(task, method, tools, token_counts, all_schema_tokens, split):
    if hasattr(method, "reset"):
        method.reset()
    tool_names = {tool["name"] for tool in tools}
    turns = []
    union = []
    for number, task_turn in enumerate(task["turns"], start=1):
        result = method.retrieve(task_turn["query"], tools)
        metrics = score_context(result.tool_ids, task_turn.get("required_tools", []), task_turn.get("supporting_tools", []), len(tools))
        turns.append({
            "turn": number,
            "query": task_turn["query"],
            "tool_ids": result.tool_ids,
            "schema_tokens": sum(token_counts[tool_id] for tool_id in result.tool_ids),
            "all_schema_tokens": all_schema_tokens,
            "schema_token_reduction": 1 - sum(token_counts[tool_id] for tool_id in result.tool_ids) / all_schema_tokens,
            "metrics": metrics,
            "metadata": result.metadata,
        })
        for tool_id in result.tool_ids:
            if tool_id not in union:
                union.append(tool_id)
    return {
        "task_id": task["id"],
        "category": task["category"],
        "split": split,
        "method": method.name,
        "turns": turns,
        "metrics": score_context(union, task["required_tools"], task["supporting_tools"], len(tool_names)),
    }


def all_records(tasks, tools, tokenizer):
    token_counts = {
        tool["name"]: len(tokenizer.encode(serialize_tool(tool), add_special_tokens=False))
        for tool in tools
    }
    all_schema_tokens = sum(token_counts.values())
    records = []
    for task in tasks:
        split = "validation"
        for method in methods():
            records.append(evaluate_task(task, method, tools, token_counts, all_schema_tokens, split))
    return records


def summarize(records, category=None):
    grouped = defaultdict(list)
    for record in records:
        if category is None or record["category"] == category:
            grouped[record["method"]].append(record)
    rows = []
    for method, selected in sorted(grouped.items()):
        turns = [turn for record in selected for turn in record["turns"]]
        counts = [turn["metrics"]["tool_count"] for turn in turns]
        rows.append({
            "split": "validation",
            "method": method,
            "category": category or "all",
            "task_count": len(selected),
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
        })
    return rows


def records_by_method(records):
    return {(record["task_id"], record["method"]): record for record in records}


def find_rank(details, tool_id):
    matches = []
    for detail in details:
        item = next((item for item in detail["ranking"] if item["name"] == tool_id), None)
        if item:
            matches.append((item["rank"], detail, item))
    return min(matches, key=lambda item: item[0]) if matches else (None, None, None)


def required_rank_analysis(records, tasks):
    by_method = records_by_method(records)
    rows = []
    for task in tasks:
        fixed4 = by_method[(task["id"], "fixed_k_4")]
        fixed8 = by_method[(task["id"], "fixed_k_8")]
        maxgap = by_method[(task["id"], "bounded_max_gap")]
        hybrid = by_method[(task["id"], "min_4_plus_max_gap")]
        for fixed_turn, max_turn, hybrid_turn, fixed8_turn, task_turn in zip(
            fixed4["turns"], maxgap["turns"], hybrid["turns"], fixed8["turns"], task["turns"]
        ):
            for tool_id in task_turn.get("required_tools", []):
                rank, detail, item = find_rank(fixed_turn["metadata"]["intent_details"], tool_id)
                _max_rank, max_detail, _max_item = find_rank(max_turn["metadata"]["intent_details"], tool_id)
                _hybrid_rank, hybrid_detail, _hybrid_item = find_rank(hybrid_turn["metadata"]["intent_details"], tool_id)
                rows.append({
                    "task_id": task["id"],
                    "category": task["category"],
                    "turn": fixed_turn["turn"],
                    "query": fixed_turn["query"],
                    "intent": detail["text"] if detail else "",
                    "required_tool": tool_id,
                    "required_rank": rank,
                    "required_score": item["score"] if item else None,
                    "max_gap_k": max_detail["cutoff"] if max_detail else None,
                    "fixed4_recovers": tool_id in set(fixed_turn["tool_ids"]),
                    "hybrid_recovers": tool_id in set(hybrid_turn["tool_ids"]),
                    "fixed8_recovers": tool_id in set(fixed8_turn["tool_ids"]),
                    "max_gap_recovers": tool_id in set(max_turn["tool_ids"]),
                })
    return rows


def expansion_analysis(records, tasks):
    by_method = records_by_method(records)
    rows = []
    tool_rows = []
    for task in tasks:
        hybrid = by_method[(task["id"], "min_4_plus_max_gap")]
        fixed4 = by_method[(task["id"], "fixed_k_4")]
        for hybrid_turn, fixed4_turn, task_turn in zip(hybrid["turns"], fixed4["turns"], task["turns"]):
            required = set(task_turn.get("required_tools", []))
            supporting = set(task_turn.get("supporting_tools", []))
            for detail in hybrid_turn["metadata"]["intent_details"]:
                if detail["cutoff"] <= 4:
                    continue
                additions = detail["ranking"][4:detail["cutoff"]]
                required_added = [item["name"] for item in additions if item["name"] in required]
                supporting_added = [item["name"] for item in additions if item["name"] in supporting]
                unlabeled_added = [item["name"] for item in additions if item["name"] not in required | supporting]
                rows.append({
                    "task_id": task["id"],
                    "category": task["category"],
                    "turn": hybrid_turn["turn"],
                    "query": hybrid_turn["query"],
                    "intent": detail["text"],
                    "max_gap_k": detail["selection"].get("max_gap_cutoff"),
                    "hybrid_k": detail["cutoff"],
                    "added_tool_count": len(additions),
                    "required_added": ";".join(required_added),
                    "supporting_added": ";".join(supporting_added),
                    "unlabeled_added": ";".join(unlabeled_added),
                    "additional_schema_tokens_vs_fixed4": hybrid_turn["schema_tokens"] - fixed4_turn["schema_tokens"],
                })
                for item in additions:
                    tool_rows.append({
                        "task_id": task["id"],
                        "category": task["category"],
                        "turn": hybrid_turn["turn"],
                        "query": hybrid_turn["query"],
                        "intent": detail["text"],
                        "rank": item["rank"],
                        "tool_id": item["name"],
                        "score": item["score"],
                        "label": "required" if item["name"] in required else "supporting" if item["name"] in supporting else "unlabeled",
                    })
    return rows, tool_rows


def k_distribution(records):
    rows = []
    for method in sorted({record["method"] for record in records}):
        sizes = [detail["cutoff"] for record in records if record["method"] == method for turn in record["turns"] for detail in turn["metadata"]["intent_details"]]
        rows.extend([
            {"method": method, "statistic": "mean", "value": statistics.mean(sizes)},
            {"method": method, "statistic": "median", "value": statistics.median(sizes)},
            {"method": method, "statistic": "p90", "value": percentile(sizes, .90)},
            {"method": method, "statistic": "p95", "value": percentile(sizes, .95)},
            {"method": method, "statistic": "max", "value": max(sizes)},
        ] + [{"method": method, "statistic": f"k_{k}", "value": sizes.count(k)} for k in range(1, 17)])
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run():
    payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    tools = load_tools(INDEX_PATH)
    tasks = payload["tasks"]
    validate_tasks(tasks, tools)
    octopus = Octopus(index_path=INDEX_PATH)
    asyncio.run(octopus.initialize())
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
    records = all_records(tasks, tools, tokenizer)
    production_v3 = __import__("evals.methods.octopus", fromlist=["OctopusMethod"]).OctopusMethod(octopus)
    token_counts = {
        tool["name"]: len(tokenizer.encode(serialize_tool(tool), add_special_tokens=False))
        for tool in tools
    }
    all_schema_tokens = sum(token_counts.values())
    production_records = [
        evaluate_task(task, production_v3, tools, token_counts, all_schema_tokens, "validation")
        for task in tasks
    ]
    summary = summarize(records)
    categories = []
    for category in sorted({task["category"] for task in tasks}):
        categories.extend(summarize(records, category))
    rank_rows = required_rank_analysis(records, tasks)
    expansion_rows, expansion_tools = expansion_analysis(records, tasks)
    distribution = k_distribution(records)
    hybrid_summary = next(row for row in summary if row["method"] == "min_4_plus_max_gap")
    v3_summary = next(row for row in summarize(production_records) if row["method"] == "brown_octopus_v3")
    compared_fields = (
        "required_recall",
        "complete_required_rate",
        "supporting_recall",
        "mean_exposed_tools_per_turn",
        "mean_schema_tokens_per_turn",
        "schema_token_reduction",
    )
    production_v3_matches_hybrid = all(
        abs(float(hybrid_summary[field]) - float(v3_summary[field])) < 1e-12
        for field in compared_fields
    )
    diagnostic = {
        "experiment_id": datetime.now(timezone.utc).strftime("validation_%Y%m%dT%H%M%S_%fZ"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "dataset": str(DATASET_PATH),
        "dataset_hash": sha256_file(DATASET_PATH),
        "tool_universe": str(INDEX_PATH / "tools.json"),
        "tool_universe_hash": sha256_file(INDEX_PATH / "tools.json"),
        "tool_count": len(tools),
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "tool_representation": "name + description",
        "intent_analyzer": "spaCy en_core_web_trf",
        "index_version": 1,
        "policy": "selected_k = max(4, bounded_max_gap_k)",
        "max_tools": 16,
        "min_gap_percent": 2.0,
        "ttl": 8,
        "active_cap": 30,
        "production_v3_matches_evaluation_hybrid": production_v3_matches_hybrid,
        "summary": summary,
        "production_v3_summary": summarize(production_records),
        "production_v3_records": production_records,
        "category_summary": categories,
        "required_rank_analysis": rank_rows,
        "expansion_summary": expansion_rows,
        "expansion_tools": expansion_tools,
        "k_distribution": distribution,
        "records": records,
    }
    raw_path = RAW_DIR / f"{diagnostic['experiment_id']}.json"
    with raw_path.open("x", encoding="utf-8") as file:
        json.dump(diagnostic, file, indent=2)
    write_csv(REPORT_DIR / "validation_summary.csv", summary)
    write_csv(REPORT_DIR / "validation_category_summary.csv", categories)
    write_csv(REPORT_DIR / "validation_required_rank_analysis.csv", rank_rows)
    write_csv(REPORT_DIR / "validation_expansion_summary.csv", expansion_rows)
    write_csv(REPORT_DIR / "validation_expansion_tools.csv", expansion_tools)
    write_csv(REPORT_DIR / "validation_k_distribution.csv", distribution)
    write_csv(REPORT_DIR / "validation_v3_regression.csv", summarize(production_records))
    print(f"Raw results: {raw_path}")


if __name__ == "__main__":
    run()
