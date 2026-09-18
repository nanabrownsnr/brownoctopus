ACTIVE_TOOL_TTL = 8
MAX_ACTIVE_TOOLS = 30


def update_active_tools(
    active_tools: dict[str, int],
    retrieved_tools: list[str],
    turn: int,
) -> dict[str, int]:
    updated_tools = active_tools.copy()

    # Add newly retrieved tools and refresh
    # tools that were retrieved again.
    for tool_name in retrieved_tools:
        updated_tools[tool_name] = turn

    # Remove tools that have exceeded their TTL.
    updated_tools = {
        tool_name: last_retrieved_turn
        for tool_name, last_retrieved_turn in updated_tools.items()
        if turn - last_retrieved_turn <= ACTIVE_TOOL_TTL
    }

    # If the context is still too large,
    # keep the most recently retrieved tools.
    if len(updated_tools) > MAX_ACTIVE_TOOLS:
        updated_tools = dict(
            sorted(
                updated_tools.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:MAX_ACTIVE_TOOLS]
        )

    return updated_tools



