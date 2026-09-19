from collections.abc import Iterable


def required_recall(retrieved: Iterable[str], required: Iterable[str]) -> float:
    required_set = set(required)
    if not required_set:
        return 1.0
    return len(required_set & set(retrieved)) / len(required_set)


def complete_required(retrieved: Iterable[str], required: Iterable[str]) -> bool:
    return set(required) <= set(retrieved)


def supporting_recall(retrieved: Iterable[str], supporting: Iterable[str]) -> float:
    supporting_set = set(supporting)
    if not supporting_set:
        return 1.0
    return len(supporting_set & set(retrieved)) / len(supporting_set)


def context_reduction(selected_count: int, universe_count: int) -> float:
    if universe_count == 0:
        return 0.0
    return 1.0 - (selected_count / universe_count)


def score_context(
    tool_ids: Iterable[str],
    required: Iterable[str],
    supporting: Iterable[str],
    universe_count: int,
) -> dict[str, float | bool | int]:
    selected = list(tool_ids)
    return {
        "required_recall": required_recall(selected, required),
        "complete_required": complete_required(selected, required),
        "supporting_recall": supporting_recall(selected, supporting),
        "tool_count": len(selected),
        "context_reduction": context_reduction(len(selected), universe_count),
    }
