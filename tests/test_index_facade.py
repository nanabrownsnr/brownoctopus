import json

import pytest

from brown_octopus import (
    ApiCapabilitySource,
    CapabilityDiscoveryResult,
    JsonCapabilitySource,
    McpRegistrySource,
    McpServerSource,
    OctopusIndex,
)
from brown_octopus.mcp_discovery import make_capability_id
from brown_octopus.sources import LocalMcpCatalogSource


def test_capability_id_uses_stable_server_id_without_url_hash():
    assert make_capability_id(
        "Outlook",
        "send_email",
        "https://example.com/mcp",
        server_id="outlook-prod",
    ) == "outlook_prod_send_email"


def test_mcp_server_source_uses_source_id(monkeypatch):
    async def fake_discover(url, server_id=None):
        return [
            {
                "capability_id": f"{server_id}_send_email",
                "source_id": server_id,
                "name": "outlook_send_email",
            }
        ]

    monkeypatch.setattr("brown_octopus.sources.discover_tools", fake_discover)

    result = __import__("asyncio").run(
        McpServerSource(
            "https://example.com/mcp",
            source_id="outlook-prod",
        ).discover()
    )

    assert result.successful_sources == ["outlook-prod"]
    assert result.tools[0]["capability_id"] == "outlook-prod_send_email"


@pytest.mark.anyio
async def test_local_catalog_preserves_server_id(monkeypatch, tmp_path):
    path = tmp_path / "mcps.json"
    path.write_text(
        json.dumps(
            {
                "items": [
                    {
                        "id": "outlook-prod",
                        "name": "Outlook",
                        "url": "https://example.com/outlook/mcp",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    async def fake_discover(url, server_id=None):
        return [
            {
                "capability_id": f"{server_id}_send_email",
                "source_id": server_id,
                "name": "outlook_send_email",
            }
        ]

    monkeypatch.setattr("brown_octopus.sources.discover_tools", fake_discover)

    result = await LocalMcpCatalogSource(path).discover()

    assert result.successful_sources == ["outlook-prod"]
    assert result.tools[0]["capability_id"] == "outlook-prod_send_email"


@pytest.mark.anyio
async def test_json_source_reads_normalized_capabilities(tmp_path):
    path = tmp_path / "capabilities.json"
    path.write_text(
        json.dumps(
            {
                "capabilities": [
                    {
                        "capability_id": "crm_search",
                        "source_id": "crm",
                        "name": "search_customers",
                        "description": "Search customers",
                        "input_schema": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = await JsonCapabilitySource(path).discover()

    assert result.tools[0]["capability_id"] == "crm_search"
    assert result.successful_sources[0].startswith("file:")


def test_api_source_normalizes_mapped_fields():
    source = ApiCapabilitySource(
        "https://registry.example/capabilities",
        items_path="data.items",
        fields={
            "capability_id": "id",
            "source_id": "server.id",
            "name": "tool_name",
            "description": "summary",
            "input_schema": "schema",
        },
    )

    assert source._normalize(
        {
            "id": "crm_search",
            "server": {"id": "crm"},
            "tool_name": "search_customers",
            "summary": "Search customers",
            "schema": {"type": "object"},
        }
    ) == {
        "capability_id": "crm_search",
        "source_id": "crm",
        "name": "search_customers",
        "description": "Search customers",
        "input_schema": {"type": "object"},
    } 


def test_api_source_preserves_execution_metadata():
    source = ApiCapabilitySource("https://registry.example/capabilities")

    tool = source._normalize(
        {
            "capability_id": "outlook:send_email",
            "source_id": "outlook-prod",
            "name": "outlook_send_email",
            "description": "Send an email",
            "input_schema": {},
            "mcp_url": "https://example.com/outlook/mcp",
            "tool_name": "send_email",
        }
    )

    assert tool["mcp_url"] == "https://example.com/outlook/mcp"
    assert tool["tool_name"] == "send_email"


@pytest.mark.anyio
async def test_mcp_registry_source_paginates_and_discovers_tools():
    pages = {
        1: {
            "items": [
                {"id": "web", "name": "Web Search", "url": "https://web"},
                {"id": "mail", "name": "Mail", "url": "https://mail"},
            ]
        },
        2: {
            "items": [
                {"id": "calendar", "name": "Calendar", "url": "https://calendar"}
            ]
        },
    }
    calls = []

    async def discover_tools_for_server(url, server_id=None):
        calls.append((url, server_id))
        return [
            {
                "capability_id": f"{server_id}:tool",
                "name": f"{server_id}_tool",
                "description": "test",
                "input_schema": {},
                "mcp_url": url,
                "tool_name": "tool",
            }
        ]

    source = McpRegistrySource(
        "https://registry.example/servers",
        page_size=2,
        tool_discoverer=discover_tools_for_server,
    )
    source._request_page = lambda page, headers, cursor=None: (pages[page], 200)

    result = await source.discover()

    assert [item[1] for item in calls] == ["web", "mail", "calendar"]
    assert len(result.tools) == 3
    assert result.failed_sources == {}


@pytest.mark.anyio
async def test_mcp_registry_source_follows_cursor_pagination():
    requests = []
    responses = {
        None: (
            {
                "items": [
                    {"id": "first", "url": "https://first"},
                ],
                "next_cursor": "cursor-2",
            },
            200,
        ),
        "cursor-2": (
            {
                "items": [
                    {"id": "second", "url": "https://second"},
                ],
                "next_cursor": None,
            },
            200,
        ),
    }

    async def discover_tools_for_server(url, server_id=None):
        return [
            {
                "capability_id": f"{server_id}:tool",
                "name": f"{server_id}_tool",
                "description": "test",
                "input_schema": {},
            }
        ]

    source = McpRegistrySource(
        "https://registry.example/servers",
        page_size=10,
        tool_discoverer=discover_tools_for_server,
    )

    def request_page(page, headers, cursor=None):
        requests.append((page, cursor))
        return responses[cursor]

    source._request_page = request_page
    result = await source.discover()

    assert requests == [(1, None), (1, "cursor-2")]
    assert [tool["capability_id"] for tool in result.tools] == [
        "first:tool",
        "second:tool",
    ]


@pytest.mark.anyio
async def test_mcp_registry_source_refreshes_headers_once_after_auth_failure():
    requests = []
    refreshes = []

    async def refresh_headers():
        refreshes.append(True)
        return {"Authorization": "Bearer refreshed"}

    source = McpRegistrySource(
        "https://registry.example/servers",
        headers={"Authorization": "Bearer expired"},
        refresh_headers=refresh_headers,
    )

    def request_page(page, headers, cursor=None):
        requests.append(dict(headers))
        if len(requests) == 1:
            return None, 401
        return {"items": []}, 200

    source._request_page = request_page
    result = await source.discover()

    assert len(refreshes) == 1
    assert requests == [
        {"Authorization": "Bearer expired"},
        {"Authorization": "Bearer refreshed"},
    ]
    assert result.failed_sources == {}
    assert result.successful_sources == [source.source_id]


@pytest.mark.anyio
async def test_mcp_registry_source_preserves_partial_failures():
    async def discover_tools_for_server(url, server_id=None):
        if server_id == "offline":
            raise RuntimeError("provider unavailable")
        return [
            {
                "capability_id": f"{server_id}:tool",
                "name": f"{server_id}_tool",
                "description": "test",
                "input_schema": {},
            }
        ]

    source = McpRegistrySource(
        "https://registry.example/servers",
        tool_discoverer=discover_tools_for_server,
    )
    source._request_page = lambda page, headers, cursor=None: (
        {
            "items": [
                {"id": "online", "url": "https://online"},
                {"id": "offline", "url": "https://offline"},
            ]
        },
        200,
    )

    result = await source.discover()

    assert result.tools[0]["capability_id"] == "online:tool"
    assert result.failed_sources["offline"] == "provider unavailable"
    assert result.authoritative is False


def test_index_facade_exposes_registry_constructor():
    index = OctopusIndex.from_mcp_registry(
        "https://registry.example/servers",
        index_path="data/indexes/registry-test",
        headers={"Authorization": "Bearer secret"},
        cursor_param="next",
    )

    assert isinstance(index.sources[0], McpRegistrySource)
    assert index.sources[0].headers["Authorization"] == "Bearer secret"
    assert index.sources[0].cursor_param == "next"


@pytest.mark.anyio
async def test_index_facade_adds_and_removes_sources(monkeypatch, tmp_path):
    index = OctopusIndex.from_sources([], index_path=tmp_path / "index")
    calls = []

    index._octopus.index_store.exists = lambda: True

    async def fake_add_sources(sources):
        calls.append(
            [getattr(source, "source_id", None) for source in sources]
        )
        return CapabilityDiscoveryResult()

    async def fake_update():
        calls.append(
            [getattr(source, "source_id", None) for source in index.source.sources]
        )
        return CapabilityDiscoveryResult()

    monkeypatch.setattr(index._octopus, "add_sources", fake_add_sources)
    monkeypatch.setattr(index._octopus, "update", fake_update)
    source = McpServerSource("https://example.com/mcp", source_id="crm")

    await index.add(source)
    assert calls == [["crm"]]

    await index.remove_source("crm")
    assert calls == [["crm"], []]


@pytest.mark.anyio
async def test_index_facade_add_accepts_multiple_sources_in_one_update(
    monkeypatch,
    tmp_path,
):
    index = OctopusIndex.from_sources([], index_path=tmp_path / "index")
    index._octopus.index_store.exists = lambda: True
    calls = []

    async def fake_add_sources(sources):
        calls.append([source.source_id for source in sources])
        return CapabilityDiscoveryResult()

    monkeypatch.setattr(index._octopus, "add_sources", fake_add_sources)

    await index.add(
        McpServerSource("https://example.com/a", source_id="a"),
        McpServerSource("https://example.com/b", source_id="b"),
    )

    assert calls == [["a", "b"]]
    assert [source.source_id for source in index.sources] == ["a", "b"]


@pytest.mark.anyio
async def test_index_facade_add_requires_an_existing_index(tmp_path):
    index = OctopusIndex.from_sources([], index_path=tmp_path / "missing")

    with pytest.raises(RuntimeError, match="index exists"):
        await index.add(McpServerSource("https://example.com/mcp"))


@pytest.mark.anyio
async def test_create_does_not_replace_existing_snapshot_by_default(monkeypatch, tmp_path):
    index = OctopusIndex.from_sources([], index_path=tmp_path / "index")
    existing_tools = [{"capability_id": "existing", "name": "existing"}]
    update_calls = []

    index._octopus.index_store.exists = lambda: True
    index._octopus.index_store.load_tools = lambda: existing_tools

    async def fake_update():
        update_calls.append(True)
        return None

    monkeypatch.setattr(index, "update", fake_update)

    report = await index.create()

    assert report.tool_count == 1
    assert update_calls == []


@pytest.mark.anyio
async def test_create_replace_rebuilds_existing_snapshot(monkeypatch, tmp_path):
    index = OctopusIndex.from_sources([], index_path=tmp_path / "index")
    update_calls = []

    index._octopus.index_store.exists = lambda: True

    async def fake_update():
        update_calls.append(True)
        return "updated"

    monkeypatch.setattr(index, "update", fake_update)

    assert await index.create(replace=True) == "updated"
    assert update_calls == [True]


def test_index_reset_removes_persisted_directory(tmp_path):
    index_path = tmp_path / "index"
    index_path.mkdir()
    (index_path / "marker").write_text("x", encoding="utf-8")

    index = OctopusIndex.from_sources([], index_path=index_path)
    index.reset()

    assert not index_path.exists()
