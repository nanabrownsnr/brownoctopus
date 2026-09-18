TOOLS: list[dict] = []


def set_tools(tools: list[dict]) -> None:
    """Replace the current runtime tool universe."""
    global TOOLS
    TOOLS = tools.copy()


def get_tools(names) -> list[dict]:
    """Return tool definitions matching the requested names."""
    tools_by_name = {tool["name"]: tool for tool in TOOLS}

    return [tools_by_name[name] for name in names if name in tools_by_name]


def get_all_tools() -> list[dict]:
    """Return all tools currently available to Octopus."""
    return TOOLS.copy()
