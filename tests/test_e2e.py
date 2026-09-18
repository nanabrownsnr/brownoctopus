import json

import pytest

from octopus.bootstrap import initialize
from octopus.pipeline import process_turn
from octopus.tool_registry import get_all_tools


WEB_SEARCH_MCP_URL = "https://twynity-dev.mcp.4th-ir.com/web-search/mcp"


@pytest.fixture(scope="module")
async def initialized_octopus(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("e2e")

    catalog_path = temp_dir / "mcps.json"

    catalog = {
        "items": [
            {
                "name": "Web Search",
                "url": WEB_SEARCH_MCP_URL,
            }
        ]
    }

    catalog_path.write_text(
        json.dumps(catalog),
        encoding="utf-8",
    )

    tools = await initialize(catalog_path)

    return tools


@pytest.mark.anyio
async def test_initialize_discovers_and_indexes_real_mcp(
    initialized_octopus,
):
    tools = initialized_octopus

    assert len(tools) >= 1

    web_search_tools = [tool for tool in tools if tool["tool_name"] == "web_search"]

    assert len(web_search_tools) == 1

    tool = web_search_tools[0]

    assert tool["mcp_name"]
    assert tool["mcp_url"] == WEB_SEARCH_MCP_URL
    assert tool["description"]
    assert tool["input_schema"]


@pytest.mark.anyio
async def test_initialized_tools_populate_registry(
    initialized_octopus,
):
    discovered_names = {tool["name"] for tool in initialized_octopus}

    registry_names = {tool["name"] for tool in get_all_tools()}

    assert discovered_names <= registry_names


@pytest.mark.anyio
async def test_discovered_mcp_tool_flows_through_pipeline(
    initialized_octopus,
):
    web_search_tool = next(
        tool for tool in initialized_octopus if tool["tool_name"] == "web_search"
    )

    result = process_turn(
        query="search the web for the latest AI news",
        active_tools={},
        turn=1,
    )

    tool_names = [tool["name"] for tool in result["tools"]]

    assert web_search_tool["name"] in tool_names
    assert web_search_tool["name"] in result["active_state"]


@pytest.mark.anyio
async def test_discovered_tool_survives_follow_up_turn(
    initialized_octopus,
):
    web_search_tool = next(
        tool for tool in initialized_octopus if tool["tool_name"] == "web_search"
    )

    turn_1 = process_turn(
        query="search the web for the latest Nvidia news",
        active_tools={},
        turn=1,
    )

    assert web_search_tool["name"] in turn_1["active_state"]

    turn_2 = process_turn(
        query="what about AMD?",
        active_tools=turn_1["active_state"],
        turn=2,
    )

    tool_names = [tool["name"] for tool in turn_2["tools"]]

    assert web_search_tool["name"] in tool_names
    assert web_search_tool["name"] in turn_2["active_state"]
