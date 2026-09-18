import pytest

from octopus.mcp_discovery import (
    discover_tools,
    discover_universe,
    normalize_name,
)


WEB_SEARCH_MCP_URL = "https://twynity-dev.mcp.4th-ir.com/web-search/mcp"


def test_normalize_name():
    assert normalize_name("Web Search Tools") == "web_search_tools"
    assert normalize_name("Office-Word MCP") == "office_word_mcp"


@pytest.mark.anyio
async def test_discover_tools_from_live_mcp():
    tools = await discover_tools(WEB_SEARCH_MCP_URL)

    assert len(tools) >= 1

    tool = next(tool for tool in tools if tool["tool_name"] == "web_search")

    assert tool["mcp_name"]
    assert tool["mcp_url"] == WEB_SEARCH_MCP_URL
    assert tool["tool_name"] == "web_search"
    assert tool["description"]
    assert tool["input_schema"]

    expected_name = f"{normalize_name(tool['mcp_name'])}_" f"{tool['tool_name']}"

    assert tool["name"] == expected_name


@pytest.mark.anyio
async def test_discover_universe_combines_tools_from_multiple_mcps(
    monkeypatch,
):
    async def fake_discover_tools(mcp_url):
        if "web-search" in mcp_url:
            return [
                {
                    "name": "web_search_tools_web_search",
                    "mcp_name": "Web Search Tools",
                    "mcp_url": mcp_url,
                    "tool_name": "web_search",
                    "description": "Search the web",
                    "input_schema": {},
                }
            ]

        return [
            {
                "name": "email_tools_send_email",
                "mcp_name": "Email Tools",
                "mcp_url": mcp_url,
                "tool_name": "send_email",
                "description": "Send an email",
                "input_schema": {},
            }
        ]

    monkeypatch.setattr(
        "octopus.mcp_discovery.discover_tools",
        fake_discover_tools,
    )

    tools = await discover_universe(
        [
            "https://example.com/web-search/mcp",
            "https://example.com/email/mcp",
        ]
    )

    assert [tool["name"] for tool in tools] == [
        "web_search_tools_web_search",
        "email_tools_send_email",
    ]


@pytest.mark.anyio
async def test_discover_universe_deduplicates_tools(
    monkeypatch,
):
    async def fake_discover_tools(mcp_url):
        return [
            {
                "name": "web_search_tools_web_search",
                "mcp_name": "Web Search Tools",
                "mcp_url": mcp_url,
                "tool_name": "web_search",
                "description": "Search the web",
                "input_schema": {},
            }
        ]

    monkeypatch.setattr(
        "octopus.mcp_discovery.discover_tools",
        fake_discover_tools,
    )

    tools = await discover_universe(
        [
            "https://example.com/server-one/mcp",
            "https://example.com/server-two/mcp",
        ]
    )

    assert len(tools) == 1
    assert tools[0]["name"] == "web_search_tools_web_search"


@pytest.mark.anyio
async def test_discover_universe_tolerates_failed_mcp(
    monkeypatch,
):
    async def fake_discover_tools(mcp_url):
        if "broken" in mcp_url:
            raise RuntimeError("MCP unavailable")

        return [
            {
                "name": "web_search_tools_web_search",
                "mcp_name": "Web Search Tools",
                "mcp_url": mcp_url,
                "tool_name": "web_search",
                "description": "Search the web",
                "input_schema": {},
            }
        ]

    monkeypatch.setattr(
        "octopus.mcp_discovery.discover_tools",
        fake_discover_tools,
    )

    tools = await discover_universe(
        [
            "https://example.com/broken/mcp",
            "https://example.com/working/mcp",
        ]
    )

    assert len(tools) == 1
    assert tools[0]["name"] == "web_search_tools_web_search"
