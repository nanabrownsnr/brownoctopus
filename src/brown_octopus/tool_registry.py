TOOLS: list[dict] = []


def capability_id(tool: dict) -> str:
    """Return the stable identity used for state and deduplication.

    Older persisted indexes do not contain an explicit ID, so their existing
    names remain a backward-compatible fallback.
    """
    return str(tool.get("capability_id") or tool.get("id") or tool["name"])


def set_tools(tools: list[dict]) -> None:
    """Replace the current runtime tool universe."""
    global TOOLS
    TOOLS = tools.copy()


def get_tools(names) -> list[dict]:
    """Return definitions matching capability IDs or legacy exposed names."""
    tools_by_id = {capability_id(tool): tool for tool in TOOLS}
    tools_by_name = {tool["name"]: tool for tool in TOOLS}

    return [
        tools_by_id.get(name, tools_by_name.get(name))
        for name in names
        if tools_by_id.get(name, tools_by_name.get(name)) is not None
    ]


def get_all_tools() -> list[dict]:
    """Return all tools currently available to Octopus."""
    return TOOLS.copy()
