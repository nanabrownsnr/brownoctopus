import hashlib
import json
from pathlib import Path


DATASET_PATH = Path("data/evals/octopus_capability_benchmark.json")


def load_tasks(
    path: str | Path = DATASET_PATH,
    expected_count: int | None = None,
) -> list[dict]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload.get("tasks")
    if not isinstance(tasks, list) or (expected_count is not None and len(tasks) != expected_count):
        raise ValueError(f"Dataset must contain exactly {expected_count} tasks.")
    return tasks


def validate_tasks(tasks: list[dict], universe: list[dict]) -> None:
    universe_ids = {tool["name"] for tool in universe}
    seen_ids = set()
    for task in tasks:
        task_id = task.get("id")
        if not task_id or task_id in seen_ids:
            raise ValueError(f"Invalid or duplicate task id: {task_id!r}")
        seen_ids.add(task_id)
        if not task.get("category"):
            raise ValueError(f"Task {task_id} has no category.")
        turns = task.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ValueError(f"Task {task_id} must contain turns.")
        for field in ("required_tools", "supporting_tools"):
            ids = task.get(field, [])
            if not isinstance(ids, list) or len(ids) != len(set(ids)):
                raise ValueError(f"Task {task_id} has invalid {field}.")
            unknown = set(ids) - universe_ids
            if unknown:
                raise ValueError(
                    f"Task {task_id} references unknown {field}: {sorted(unknown)}"
                )
        for turn in turns:
            if not isinstance(turn.get("query"), str) or not turn["query"].strip():
                raise ValueError(f"Task {task_id} contains an invalid turn query.")
            for field in ("required_tools", "supporting_tools"):
                unknown = set(turn.get(field, [])) - universe_ids
                if unknown:
                    raise ValueError(
                        f"Task {task_id} turn references unknown {field}: {sorted(unknown)}"
                    )


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
