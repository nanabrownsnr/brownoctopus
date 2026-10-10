import hashlib
import logging
from typing import Mapping


logger = logging.getLogger("brown_octopus.discovery")


def normalize_name(name: str) -> str:
    """Normalize a name for use in a namespaced tool identifier."""
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def make_capability_id(
    server_name: str,
    tool_name: str,
    server_url: str | None = None,
    server_id: str | None = None,
) -> str:
    stable_server_id = normalize_name(server_id or server_name)
    if server_id is None and server_url:
        stable_server_id = (
            f"{stable_server_id}-"
            f"{hashlib.sha256(server_url.encode()).hexdigest()[:12]}"
        )
    return f"{stable_server_id}_{normalize_name(tool_name)}"


def discovered_capability_id(tool: dict) -> str:
    return str(
        tool.get("capability_id")
        or tool.get("id")
        or (
            make_capability_id(
                tool["mcp_name"],
                tool["tool_name"],
                tool.get("mcp_url"),
            )
            if tool.get("mcp_name") and tool.get("tool_name")
            else tool["name"]
        )
    )


async def discover_tools(
    mcp_url: str,
    server_id: str | None = None,
    mcp_headers: Mapping[str, str] | None = None,
) -> list[dict]:
    """
    Connect to an MCP server and discover its available tools.

    Tool definitions are namespaced using the MCP server name
    so tools from different servers cannot collide.
    """
    try:
        from fastmcp import Client
    except ImportError as exc:
        raise RuntimeError(
            "MCP discovery requires FastMCP client support. "
            "Install the default brown-octopus dependencies or configure a "
            "different CapabilitySource."
        ) from exc

    client_kwargs = {"headers": dict(mcp_headers)} if mcp_headers else {}
    async with Client(mcp_url, **client_kwargs) as client:
        server_info = client.server_info
        tools = await client.list_tools()

    mcp_name = server_info.name
    normalized_mcp_name = normalize_name(mcp_name)

    return [
        {
            "capability_id": make_capability_id(
                mcp_name,
                tool.name,
                mcp_url,
                server_id=server_id,
            ),
            "source_id": server_id or mcp_url,
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

            logger.info(
                "capability_source_discovered",
                extra={"event_data": {"source_id": mcp_url, "tool_count": len(tools)}},
            )

        except Exception as error:
            logger.warning(
                "capability_source_failed",
                extra={"event_data": {"source_id": mcp_url, "error": str(error)}},
            )

    normalized_tools = []
    for tool in all_tools:
        normalized_tool = tool.copy()
        normalized_tool.setdefault(
            "capability_id",
            discovered_capability_id(normalized_tool),
        )
        normalized_tool.setdefault(
            "source_id",
            normalized_tool.get("mcp_url") or normalized_tool.get("mcp_name"),
        )
        normalized_tools.append(normalized_tool)

    unique_tools = {discovered_capability_id(tool): tool for tool in normalized_tools}

    return list(unique_tools.values())
