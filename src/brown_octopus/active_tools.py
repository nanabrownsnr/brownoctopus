ACTIVE_TOOL_TTL = 8
MAX_ACTIVE_TOOLS = 30


def update_active_tools(
    active_tools: dict[str, int],
    retrieved_tools: list[str],
    turn: int,
    ttl: int = ACTIVE_TOOL_TTL,
    max_active_tools: int = MAX_ACTIVE_TOOLS,
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
        if turn - last_retrieved_turn <= ttl
    }

    # If the context is still too large,
    # keep the most recently retrieved tools.
    if len(updated_tools) > max_active_tools:
        # When several capabilities were retrieved on the same turn, retain
        # the later entries as the most recently encountered ones. This keeps
        # the cap deterministic and evicts the oldest entry first.
        indexed = list(enumerate(updated_tools.items()))
        indexed.sort(key=lambda item: (item[1][1], item[0]), reverse=True)
        updated_tools = dict(item for _, item in indexed[:max_active_tools])

    return updated_tools



