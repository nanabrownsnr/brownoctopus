from pathlib import Path

from octopus.analyzer import initialize_analyzer
from octopus.mcp_discovery import discover_universe
from octopus.retriever import (
    initialize_retriever,
    refresh_index,
)
from octopus.sources import load_mcp_urls
from octopus.tool_registry import set_tools


DEFAULT_MCP_CATALOG = Path("data/mcps.json")


async def initialize(
    catalog_path: str | Path = DEFAULT_MCP_CATALOG,
) -> list[dict]:
    """
    Initialize Brown Octopus and prepare it for requests.

    Loads runtime models, discovers the available MCP tools,
    populates the registry, and builds the retrieval index.
    """
    initialize_analyzer()
    initialize_retriever()

    mcp_urls = load_mcp_urls(catalog_path)

    tools = await discover_universe(mcp_urls)

    set_tools(tools)
    refresh_index()

    return tools
