from fastmcp import Client


def normalize_name(name: str) -> str:
    """Normalize a name for use in a namespaced tool identifier."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


async def discover_tools(mcp_url: str) -> list[dict]:
    """
    Connect to an MCP server and discover its available tools.

    Tool definitions are namespaced using the MCP server name
    so tools from different servers cannot collide.
    """
    async with Client(mcp_url) as client:
        server_info = client.server_info
        tools = await client.list_tools()

    mcp_name = server_info.name
    normalized_mcp_name = normalize_name(mcp_name)

    return [
        {
            "name": f"{normalized_mcp_name}_{tool.name}",
            "mcp_name": mcp_name,
            "mcp_url": mcp_url,
            "tool_name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.input_schema,
        }
        for tool in tools
    ]


async def discover_universe(
    mcp_urls: list[str],
) -> list[dict]:
    """
    Discover tools from all configured MCP servers.

    A failure from one MCP does not prevent the remaining
    servers from being discovered.
    """
    all_tools = []

    for mcp_url in mcp_urls:
        try:
            tools = await discover_tools(mcp_url)
            all_tools.extend(tools)

            print(f"✓ {mcp_url}: " f"{len(tools)} tools")

        except Exception as error:
            print(f"✗ {mcp_url}: " f"{error}")

    unique_tools = {tool["name"]: tool for tool in all_tools}

    return list(unique_tools.values())
