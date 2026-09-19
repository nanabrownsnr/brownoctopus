MAX_TOOLS = 16
MIN_TOOLS = 4
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


def select_min4_bounded_max_gap(
    ranked_tools: list[dict],
    min_tools: int = MIN_TOOLS,
    max_tools: int = MAX_TOOLS,
    min_gap_percent: float = MIN_GAP_PERCENT,
) -> list[dict]:
    """Select a bounded Max Gap prefix with a per-intent minimum floor.

    The existing Max Gap implementation remains the only source of the
    adaptive cutoff. The minimum is applied to each operational intent before
    the caller merges and deduplicates selections from multiple intents.
    """
    if not ranked_tools:
        return []

    max_gap_selected = select_bounded_max_gap(
        ranked_tools,
        max_tools=max_tools,
        min_gap_percent=min_gap_percent,
    )
    cutoff = min(max_tools, max(min_tools, len(max_gap_selected)))
    return ranked_tools[:cutoff]
