MAX_TOOLS = 16
MIN_GAP_PERCENT = 2.0


def select_bounded_max_gap(
    ranked_tools: list[dict],
    max_tools: int = MAX_TOOLS,
    min_gap_percent: float = MIN_GAP_PERCENT,
) -> list[dict]:
    if not ranked_tools:
        return []

    selected_window = ranked_tools[:max_tools]

    if len(ranked_tools) < 2:
        return selected_window

    peak_score = ranked_tools[0]["score"]

    if peak_score == 0:
        return selected_window

    comparison_window = ranked_tools[: max_tools + 1]

    gaps = []

    for index in range(len(comparison_window) - 1):
        current_score = comparison_window[index]["score"]

        next_score = comparison_window[index + 1]["score"]

        gap_percent = ((current_score - next_score) / peak_score) * 100

        gaps.append(
            {
                "cutoff": index + 1,
                "gap_percent": gap_percent,
            }
        )

    strongest_gap = max(
        gaps,
        key=lambda gap: gap["gap_percent"],
    )

    if strongest_gap["gap_percent"] < min_gap_percent:
        return selected_window

    return ranked_tools[: strongest_gap["cutoff"]]
