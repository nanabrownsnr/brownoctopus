import csv
import json
from pathlib import Path

from evals.datasets.internal import DATASET_PATH, load_tasks, validate_tasks
from octopus.index_store import load_tools


REPORT_DIR = Path("results/reports")


def find_milestone1_raw() -> Path:
    for path in sorted(Path("results/raw").glob("retrieval_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("dataset_version") == "octopus_capability_benchmark_v1_milestone1":
            return path
    raise FileNotFoundError("No Milestone 1 raw result was found.")


def _names(values):
    return ";".join(values)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    payload = json.loads(find_milestone1_raw().read_text(encoding="utf-8"))
    tasks = {task["id"]: task for task in load_tasks(DATASET_PATH)}
    tools = load_tools("data/indexes/default")
    validate_tasks(list(tasks.values()), tools)
    task_rows = []
    turn_rows = []
    for record in payload["records"]:
        task = tasks[record["task_id"]]
        retrieved = record["turns"]
        union_ids = []
        for turn in retrieved:
            for tool_id in turn["tool_ids"]:
                if tool_id not in union_ids:
                    union_ids.append(tool_id)
        required = set(task["required_tools"])
        supporting = set(task["supporting_tools"])
        task_rows.append(
            {
                "task_id": task["id"],
                "category": task["category"],
                "method": record["method"],
                "required_tool_ids": _names(task["required_tools"]),
                "supporting_tool_ids": _names(task["supporting_tools"]),
                "retrieved_tool_ids": _names(union_ids),
                "required_hits": _names(sorted(required & set(union_ids))),
                "required_misses": _names(sorted(required - set(union_ids))),
                "supporting_hits": _names(sorted(supporting & set(union_ids))),
                "supporting_misses": _names(sorted(supporting - set(union_ids))),
                "tool_count_union": len(union_ids),
                "context_reduction_union": 1 - len(union_ids) / len(tools),
            }
        )
        previous = set()
        for turn in retrieved:
            exposed = set(turn["tool_ids"])
            newly_retrieved = set(turn["metadata"].get("newly_retrieved_tool_ids", turn["tool_ids"]))
            active = set(turn["metadata"].get("active_state", {}).keys())
            retained = active - newly_retrieved if active else exposed - newly_retrieved
            required_turn = set(task["turns"][turn["turn"] - 1].get("required_tools", []))
            supporting_turn = set(task["turns"][turn["turn"] - 1].get("supporting_tools", []))
            turn_rows.append(
                {
                    "task_id": task["id"],
                    "method": record["method"],
                    "turn": turn["turn"],
                    "query": turn["query"],
                    "newly_retrieved_tools": _names(sorted(newly_retrieved)),
                    "active_or_exposed_tools": _names(sorted(active or exposed)),
                    "retained_tools": _names(sorted(retained)),
                    "required_hits": _names(sorted(required_turn & exposed)),
                    "required_misses": _names(sorted(required_turn - exposed)),
                    "supporting_hits": _names(sorted(supporting_turn & exposed)),
                    "supporting_misses": _names(sorted(supporting_turn - exposed)),
                    "tool_count_exposed": len(exposed),
                    "context_reduction_exposed": 1 - len(exposed) / len(tools),
                }
            )

    write_csv(REPORT_DIR / "milestone1_5_per_task.csv", task_rows)
    write_csv(REPORT_DIR / "milestone1_5_per_turn.csv", turn_rows)
    documentation = {
        "mean_tools_definition": "Milestone 1 summary used the mean unique tool union per task. This is retained as conversation-level mean_unique_tools_per_task, not the main per-turn exposure metric.",
        "main_tool_count_definition": "Milestone 2 uses the number of tool definitions exposed on each evaluated turn. Its aggregate is mean_exposed_tools_per_turn.",
        "dense_conversation_001": {
            "turn_1_exposed": 16,
            "turn_2_exposed": 16,
            "unique_union": 32,
        },
        "octopus_conversation_001": {
            "turn_1_exposed": 2,
            "turn_2_exposed": 13,
            "unique_union": 13,
        },
        "universe_count": len(tools),
        "canonical_universe_file": "data/indexes/default/tools.json",
    }
    (REPORT_DIR / "milestone1_5_validation.json").write_text(
        json.dumps(documentation, indent=2), encoding="utf-8"
    )
    print(f"Wrote {REPORT_DIR / 'milestone1_5_per_task.csv'}")
    print(f"Wrote {REPORT_DIR / 'milestone1_5_per_turn.csv'}")


if __name__ == "__main__":
    main()
