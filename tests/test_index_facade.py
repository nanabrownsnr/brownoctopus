import json

import pytest

from brown_octopus import (
    ApiCapabilitySource,
    CapabilityDiscoveryResult,
    JsonCapabilitySource,
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


@pytest.mark.anyio
async def test_index_facade_adds_and_removes_sources(monkeypatch, tmp_path):
    index = OctopusIndex.from_sources([], index_path=tmp_path / "index")
    calls = []

    async def fake_update():
        await index.source.discover()
        return CapabilityDiscoveryResult()

    async def fake_discover():
        calls.append(
            [getattr(source, "source_id", None) for source in index.source.sources]
        )
        return CapabilityDiscoveryResult()

    monkeypatch.setattr(index._octopus, "update", fake_update)
    monkeypatch.setattr(index.source, "discover", fake_discover)
    source = McpServerSource("https://example.com/mcp", source_id="crm")

    await index.add(source)
    assert calls == [["crm"]]

    await index.remove_source("crm")
    assert calls == [["crm"], []]


def test_index_reset_removes_persisted_directory(tmp_path):
    index_path = tmp_path / "index"
    index_path.mkdir()
    (index_path / "marker").write_text("x", encoding="utf-8")

    index = OctopusIndex.from_sources([], index_path=index_path)
    index.reset()

    assert not index_path.exists()
