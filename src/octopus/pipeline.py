from octopus.active_tools import update_active_tools
from octopus.retriever import retrieve_tools
from octopus.tool_registry import get_tools


def process_turn(
    query: str,
    active_tools: dict[str, int],
    turn: int,
) -> dict:
    """
    Build the tool context for a single agent turn.

    The current query is used to retrieve relevant tools.
    Those tools are merged with the active tool working set,
    and the resulting tool definitions are returned for the
    agent to use.
    """
    retrieved_tools = retrieve_tools(query)

    retrieved_tool_names = [tool["name"] for tool in retrieved_tools]

    active_state = update_active_tools(
        active_tools=active_tools,
        retrieved_tools=retrieved_tool_names,
        turn=turn,
    )

    active_tools = get_tools(active_state.keys())

    return {
        "retrieved_tools": retrieved_tools,
        "active_state": active_state,
        "tools": active_tools,
    }
