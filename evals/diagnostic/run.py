import csv
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
from evals.datasets.internal import sha256_file
from octopus.active_tools import ACTIVE_TOOL_TTL, MAX_ACTIVE_TOOLS, update_active_tools
from octopus.index_store import load_tools


RAW_DIR = Path("results/raw")
REPORT_DIR = Path("results/reports")
INDEX_PATH = Path("data/indexes/default")
DATASET_PATH = Path("data/evals/octopus_capability_benchmark.json")
SOURCE_RAW = Path("results/raw/selection_20260919T195231_504359Z.json")
MAX_GAP_UPPER_BOUND = 16
SAFE_FLOOR = 4


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]


def names(values: list[str]) -> str:
    return ";".join(values)


def hybrid_cutoff(max_gap_cutoff: int, floor: int = SAFE_FLOOR, upper: int = MAX_GAP_UPPER_BOUND) -> int:
    return min(upper, max(floor, max_gap_cutoff))


def rank_bucket(rank: int | None) -> str:
    if rank is None:
        return "not_ranked"
    if rank in {5, 6, 7, 8}:
        return f"rank_{rank}"
    if 9 <= rank <= 16:
        return "rank_9_16"
    if rank > 16:
        return "rank_gt_16"
    return "rank_lt_5"


def latest_source() -> Path:
    if SOURCE_RAW.exists():
        return SOURCE_RAW
    candidates = sorted(RAW_DIR.glob("selection_*.json"))
    if not candidates:
        raise FileNotFoundError("No Milestone 4 selection raw result exists.")
    return candidates[-1]


def load_source() -> tuple[dict, dict[str, dict], dict[str, dict]]:
    source_path = latest_source()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    records = {
        (record["task_id"], record["method"]): record
        for record in payload["records"]
    }
    tasks_payload = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    tasks = {task["id"]: task for task in tasks_payload["tasks"]}
    return payload, records, tasks


def detail_for_tool(details: list[dict], tool_id: str) -> tuple[dict | None, dict | None]:
    matches = []
    for detail in details:
        match = next((item for item in detail["ranking"] if item["name"] == tool_id), None)
        if match is not None:
            matches.append((match["rank"], detail, match))
    if not matches:
        return None, None
    _, detail, match = min(matches, key=lambda item: item[0])
    return detail, match


def labels_for_tool(task_turn: dict, tool_id: str) -> str:
    if tool_id in set(task_turn.get("required_tools", [])):
        return "required"
    if tool_id in set(task_turn.get("supporting_tools", [])):
        return "supporting"
    return "unlabeled"


def reconstruct_hybrid(task: dict, maxgap_record: dict, tool_map: dict[str, dict], tokenizer) -> dict:
    active: dict[str, int] = {}
    turns = []
    for turn_number, (task_turn, source_turn) in enumerate(
        zip(task["turns"], maxgap_record["turns"]), start=1
    ):
        previous = set(active)
        selected_before_memory: list[str] = []
        seen = set()
        intent_details = []
        for detail in source_turn["metadata"]["intent_details"]:
            cutoff = hybrid_cutoff(detail["cutoff"])
            selected = detail["ranking"][:cutoff]
            selected_ids = [item["name"] for item in selected]
            for tool_id in selected_ids:
                if tool_id not in seen:
                    selected_before_memory.append(tool_id)
                    seen.add(tool_id)
            intent_details.append(
                {
                    "intent_number": detail["intent_number"],
                    "text": detail["text"],
                    "action": detail.get("action"),
                    "target": detail.get("target"),
                    "max_gap_cutoff": detail["cutoff"],
                    "hybrid_cutoff": cutoff,
                    "selected_tool_ids": selected_ids,
                    "ranking": detail["ranking"],
                }
            )

        active = update_active_tools(active, selected_before_memory, turn_number)
        exposed_ids = list(active)
        retained = sorted((previous & set(active)) - set(selected_before_memory))
        expired = sorted(previous - set(active))
        exposed_tools = [tool_map[tool_id] for tool_id in exposed_ids]
        metrics = score_context(
            exposed_ids,
            task_turn.get("required_tools", []),
            task_turn.get("supporting_tools", []),
            len(tool_map),
        )
        turns.append(
            {
                "turn": turn_number,
                "query": task_turn["query"],
                "tool_ids": exposed_ids,
                "selected_before_memory_ids": selected_before_memory,
                "retained_tool_ids": retained,
                "expired_tool_ids": expired,
                "intent_details": intent_details,
                "schema_tokens": count_schema_tokens(exposed_tools, tokenizer),
                "all_schema_tokens": count_schema_tokens(tool_map.values(), tokenizer),
                "metrics": metrics,
            }
        )
    union_ids = []
    for turn in turns:
        for tool_id in turn["tool_ids"]:
            if tool_id not in union_ids:
                union_ids.append(tool_id)
    return {
        "task_id": task["id"],
        "category": task["category"],
        "method": "min_4_plus_max_gap",
        "turns": turns,
        "metrics": score_context(union_ids, task["required_tools"], task["supporting_tools"], len(tool_map)),
    }


def hybrid_records(payload: dict, records: dict, tasks: dict, tools: list[dict], tokenizer) -> list[dict]:
    tool_map = {tool["name"]: tool for tool in tools}
    output = []
    for task_id, task in tasks.items():
        source = records[(task_id, "bounded_max_gap")]
        record = reconstruct_hybrid(task, source, tool_map, tokenizer)
        record["split"] = source["split"]
        output.append(record)
    return output


def summarize(records: list[dict], method_name: str | None = None, category: str | None = None) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        if method_name and record["method"] != method_name:
            continue
        if category and record["category"] != category:
            continue
        grouped[(record["split"], record["method"])].append(record)
    rows = []
    for (split, method), selected_records in sorted(grouped.items()):
        turns = [turn for record in selected_records for turn in record["turns"]]
        counts = [turn["metrics"]["tool_count"] for turn in turns]
        all_tokens = [turn["all_schema_tokens"] for turn in turns]
        selected_tokens = [turn["schema_tokens"] for turn in turns]
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
                "mean_schema_tokens_per_turn": statistics.mean(selected_tokens),
                "schema_token_reduction": statistics.mean(1 - selected / all for selected, all in zip(selected_tokens, all_tokens)),
            }
        )
    return rows


def expansion_rows(hybrid: list[dict], maxgap_records: dict, tasks: dict) -> tuple[list[dict], list[dict]]:
    summary_rows = []
    tool_rows = []
    for record in hybrid:
        task = tasks[record["task_id"]]
        source = maxgap_records[(record["task_id"], "bounded_max_gap")]
        for hybrid_turn, source_turn, task_turn in zip(record["turns"], source["turns"], task["turns"]):
            for hybrid_detail, source_detail in zip(
                hybrid_turn["intent_details"], source_turn["metadata"]["intent_details"]
            ):
                max_k = source_detail["cutoff"]
                if max_k <= SAFE_FLOOR:
                    continue
                additions = source_detail["ranking"][SAFE_FLOOR:hybrid_detail["hybrid_cutoff"]]
                required = [item["name"] for item in additions if labels_for_tool(task_turn, item["name"]) == "required"]
                supporting = [item["name"] for item in additions if labels_for_tool(task_turn, item["name"]) == "supporting"]
                unlabeled = [item["name"] for item in additions if labels_for_tool(task_turn, item["name"]) == "unlabeled"]
                fixed4_exposed = set(
                    next(
                        turn["tool_ids"]
                        for turn in maxgap_records[(record["task_id"], "fixed_k_4")]["turns"]
                        if turn["turn"] == hybrid_turn["turn"]
                    )
                )
                recovered_required = [tool_id for tool_id in required if tool_id not in fixed4_exposed]
                recovered_supporting = [tool_id for tool_id in supporting if tool_id not in fixed4_exposed]
                added_tools = [item["name"] for item in additions]
                summary_rows.append(
                    {
                        "split": record["split"],
                        "task_id": record["task_id"],
                        "category": record["category"],
                        "turn": hybrid_turn["turn"],
                        "query": hybrid_turn["query"],
                        "intent": hybrid_detail["text"],
                        "max_gap_k": max_k,
                        "hybrid_k": hybrid_detail["hybrid_cutoff"],
                        "added_tool_count": len(additions),
                        "added_tools": names(added_tools),
                        "required_added": names(required),
                        "supporting_added": names(supporting),
                        "unlabeled_added": names(unlabeled),
                        "recovered_required_from_fixed4": names(recovered_required),
                        "recovered_supporting_from_fixed4": names(recovered_supporting),
                        "additional_schema_tokens_vs_fixed4": hybrid_turn["schema_tokens"]
                        - next(turn["schema_tokens"] for turn in maxgap_records[(record["task_id"], "fixed_k_4")]["turns"] if turn["turn"] == hybrid_turn["turn"]),
                    }
                )
                for item in additions:
                    tool_rows.append(
                        {
                            "split": record["split"],
                            "task_id": record["task_id"],
                            "category": record["category"],
                            "turn": hybrid_turn["turn"],
                            "query": hybrid_turn["query"],
                            "intent": hybrid_detail["text"],
                            "rank": item["rank"],
                            "tool_id": item["name"],
                            "score": item["score"],
                            "label": labels_for_tool(task_turn, item["name"]),
                        }
                    )
    return summary_rows, tool_rows


def fixed4_failures(records: dict, tasks: dict) -> list[dict]:
    rows = []
    fixed4_records = [record for (task_id, method), record in records.items() if method == "fixed_k_4"]
    for record in fixed4_records:
        task = tasks[record["task_id"]]
        maxgap = records[(record["task_id"], "bounded_max_gap")]
        for fixed_turn, max_turn, task_turn in zip(record["turns"], maxgap["turns"], task["turns"]):
            for tool_id in set(task_turn.get("required_tools", [])) - set(fixed_turn["tool_ids"]):
                fixed_detail, match = detail_for_tool(fixed_turn["metadata"]["intent_details"], tool_id)
                max_detail, _ = detail_for_tool(max_turn["metadata"]["intent_details"], tool_id)
                rank = match["rank"] if match else None
                max_k = max_detail["cutoff"] if max_detail else None
                rows.append(
                    {
                        "split": record["split"],
                        "task_id": record["task_id"],
                        "category": record["category"],
                        "query": fixed_turn["query"],
                        "detected_intent": fixed_detail["text"] if fixed_detail else "",
                        "required_tool": tool_id,
                        "dense_rank": rank,
                        "dense_score": match["score"] if match else None,
                        "fixed_k_cutoff": 4,
                        "max_gap_cutoff": max_k,
                        "hybrid_cutoff": hybrid_cutoff(max_k) if max_k is not None else None,
                        "rank_bucket": rank_bucket(rank),
                        "hybrid_would_recover": rank is not None and max_k is not None and rank <= hybrid_cutoff(max_k),
                    }
                )
    return rows


def k_distribution(records: list[dict], source_records: dict) -> list[dict]:
    rows = []
    methods = {"fixed_k_4": None, "fixed_k_8": None, "bounded_max_gap": None, "min_4_plus_max_gap": None}
    for method in methods:
        sizes = []
        if method == "min_4_plus_max_gap":
            selected = records
            for record in selected:
                for detail in record["turns"]:
                    sizes.extend(item["hybrid_cutoff"] for item in detail["intent_details"])
        else:
            selected = [record for (task_id, record_method), record in source_records.items() if record_method == method]
            for record in selected:
                for turn in record["turns"]:
                    sizes.extend(detail["cutoff"] for detail in turn["metadata"]["intent_details"])
        by_split = defaultdict(list)
        if method == "min_4_plus_max_gap":
            for record in selected:
                for turn in record["turns"]:
                    by_split[record["split"]].extend(detail["hybrid_cutoff"] for detail in turn["intent_details"])
        else:
            for record in selected:
                for turn in record["turns"]:
                    by_split[record["split"]].extend(detail["cutoff"] for detail in turn["metadata"]["intent_details"])
        for split, split_sizes in sorted(by_split.items()):
            rows.extend(
                [{"split": split, "method": method, "statistic": "mean", "value": statistics.mean(split_sizes)},
                 {"split": split, "method": method, "statistic": "median", "value": statistics.median(split_sizes)},
                 {"split": split, "method": method, "statistic": "p90", "value": percentile(split_sizes, 0.90)},
                 {"split": split, "method": method, "statistic": "p95", "value": percentile(split_sizes, 0.95)},
                 {"split": split, "method": method, "statistic": "max", "value": max(split_sizes)}]
                + [{"split": split, "method": method, "statistic": f"k_{k}", "value": split_sizes.count(k)} for k in range(1, 17)]
            )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run() -> Path:
    payload, source_records, tasks = load_source()
    tools = load_tools(INDEX_PATH)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
    hybrid = hybrid_records(payload, source_records, tasks, tools, tokenizer)
    all_records = []
    for method in ("fixed_k_4", "fixed_k_8", "bounded_max_gap"):
        all_records.extend(record for (task_id, record_method), record in source_records.items() if record_method == method)
    all_records.extend(hybrid)
    failures = fixed4_failures(source_records, tasks)
    expansion_summary, expansion_tools = expansion_rows(hybrid, source_records, tasks)
    distributions = k_distribution(hybrid, source_records)
    summary = summarize(all_records)
    category_summary = []
    for category in sorted({task["category"] for task in tasks.values()}):
        category_summary.extend(summarize(all_records, category=category))

    expansion_intents = len(expansion_summary)
    expansion_required = sum(bool(row["recovered_required_from_fixed4"]) for row in expansion_summary)
    expansion_supporting = sum(bool(row["recovered_supporting_from_fixed4"]) for row in expansion_summary)
    expansion_only_unlabeled = sum(
        bool(row["unlabeled_added"]) and not row["required_added"] and not row["supporting_added"]
        for row in expansion_summary
    )
    expansion_token_cost = sum(row["additional_schema_tokens_vs_fixed4"] for row in expansion_summary)
    diagnostic = {
        "experiment_id": datetime.now(timezone.utc).strftime("hybrid_diagnostic_%Y%m%dT%H%M%S_%fZ"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "source_selection_raw": str(latest_source()),
        "source_selection_raw_hash": sha256_file(latest_source()),
        "dataset": str(DATASET_PATH),
        "dataset_hash": sha256_file(DATASET_PATH),
        "tool_universe": str(INDEX_PATH / "tools.json"),
        "tool_universe_hash": sha256_file(INDEX_PATH / "tools.json"),
        "tool_count": len(tools),
        "embedding_model": payload["embedding_model"],
        "tool_representation": payload["tool_representation"],
        "exploratory": True,
        "policy": "selected_k = max(4, bounded_max_gap_k)",
        "max_gap_upper_bound": MAX_GAP_UPPER_BOUND,
        "active_ttl": ACTIVE_TOOL_TTL,
        "active_cap": MAX_ACTIVE_TOOLS,
        "fixed4_failure_count": len(failures),
        "expansion_intent_count": expansion_intents,
        "expansion_intents_recovering_required": expansion_required,
        "expansion_intents_recovering_supporting": expansion_supporting,
        "expansion_intents_only_unlabeled": expansion_only_unlabeled,
        "expansion_additional_schema_tokens_sum": expansion_token_cost,
        "summary": summary,
        "category_summary": category_summary,
        "fixed4_failures": failures,
        "expansion_summary": expansion_summary,
        "expansion_tools": expansion_tools,
        "hybrid_records": hybrid,
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_DIR / f"{diagnostic['experiment_id']}.json"
    with raw_path.open("x", encoding="utf-8") as file:
        json.dump(diagnostic, file, indent=2)
    write_csv(REPORT_DIR / "hybrid_diagnostic_summary.csv", summary)
    write_csv(REPORT_DIR / "hybrid_diagnostic_category_summary.csv", category_summary)
    write_csv(REPORT_DIR / "hybrid_fixed4_failures.csv", failures)
    write_csv(REPORT_DIR / "hybrid_expansion_summary.csv", expansion_summary)
    write_csv(REPORT_DIR / "hybrid_expansion_tools.csv", expansion_tools)
    write_csv(REPORT_DIR / "hybrid_k_distribution.csv", distributions)
    write_csv(REPORT_DIR / "hybrid_recall_context.csv", [row for row in summary if row["split"] == "held_out"])
    return raw_path


if __name__ == "__main__":
    print(f"Raw results: {run()}")
