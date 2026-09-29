import json
import asyncio
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from brown_octopus.mcp_discovery import discover_tools
from brown_octopus.contracts import CapabilityDiscoveryResult, CapabilitySource


@dataclass(frozen=True)
class CapabilityUpdateReport:
    added: list[str]
    changed: list[str]
    removed: list[str]
    failed_sources: dict[str, str]
    tool_count: int
    authoritative: bool = True


class LocalMcpCatalogSource:
    """Default source backed by the existing local MCP URL catalog."""

    def __init__(self, catalog_path: str | Path = "data/mcps.json") -> None:
        self.catalog_path = Path(catalog_path)

    async def discover(self) -> CapabilityDiscoveryResult:
        result = CapabilityDiscoveryResult()
        for server in load_mcp_servers(self.catalog_path):
            url = server["url"]
            source_id = str(server.get("id") or url)
            try:
                tools = await discover_tools(url, server_id=source_id)
            except Exception as exc:
                result.failed_sources[source_id] = str(exc)
                continue
            result.successful_sources.append(source_id)
            result.tools.extend(tools)
        return result


class McpServerSource:
    """Discover capabilities from one HTTP MCP server."""

    def __init__(self, url: str, source_id: str | None = None) -> None:
        self.url = url
        self.source_id = source_id or url

    async def discover(self) -> CapabilityDiscoveryResult:
        result = CapabilityDiscoveryResult()
        try:
            result.tools.extend(
                await discover_tools(self.url, server_id=self.source_id)
            )
            result.successful_sources.append(self.source_id)
        except Exception as exc:
            result.failed_sources[self.source_id] = str(exc)
        return result


class JsonCapabilitySource:
    """Read an already-normalized capability list from a JSON file."""

    def __init__(
        self,
        path: str | Path,
        *,
        items_path: str = "capabilities",
    ) -> None:
        self.path = Path(path)
        self.items_path = items_path
        self.source_id = f"file:{self.path}"

    async def discover(self) -> CapabilityDiscoveryResult:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        records = _nested_value(payload, self.items_path)
        if not isinstance(records, list):
            raise ValueError(
                f"Capability JSON field '{self.items_path}' must contain a list."
            )
        tools = [record for record in records if isinstance(record, dict)]
        return CapabilityDiscoveryResult(
            tools=tools,
            successful_sources=[self.source_id],
            authoritative=True,
        )


class ApiCapabilitySource:
    """Read normalized capabilities from a JSON HTTP API response.

    The source uses the standard library so it does not add a package-manager
    dependency. ``fields`` maps canonical capability fields to response paths.
    """

    def __init__(
        self,
        url: str,
        *,
        items_path: str = "items",
        fields: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float = 15.0,
        source_id: str | None = None,
    ) -> None:
        self.url = url
        self.items_path = items_path
        self.fields = dict(fields or {})
        self.headers = dict(headers or {})
        self.timeout = timeout
        self.source_id = source_id or url

    async def discover(self) -> CapabilityDiscoveryResult:
        payload = await asyncio.to_thread(self._request)
        records = _nested_value(payload, self.items_path)
        if not isinstance(records, list):
            raise ValueError(
                f"Capability API field '{self.items_path}' must contain a list."
            )

        tools = []
        for record in records:
            if not isinstance(record, dict):
                continue
            tools.append(self._normalize(record))

        return CapabilityDiscoveryResult(
            tools=tools,
            successful_sources=[self.source_id],
            authoritative=True,
        )

    def _request(self) -> Any:
        request = urllib.request.Request(self.url, headers=self.headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _normalize(self, record: dict) -> dict:
        required = {
            "capability_id",
            "source_id",
            "name",
            "description",
            "input_schema",
        }
        tool = {
            field: _nested_value(record, path)
            for field, path in self.fields.items()
            if field in required
        }
        tool.setdefault("source_id", self.source_id)
        tool.setdefault("description", "")
        tool.setdefault("input_schema", {})
        if "capability_id" not in tool or "name" not in tool:
            raise ValueError(
                "Capability API records must map capability_id and name."
            )
        return tool


class CompositeCapabilitySource:
    """Combine multiple sources into one authoritative snapshot."""

    def __init__(self, sources: Sequence[CapabilitySource] = ()) -> None:
        self.sources = list(sources)
        self.suppressed_capabilities: set[str] = set()

    async def discover(self) -> CapabilityDiscoveryResult:
        combined = CapabilityDiscoveryResult()
        for source in self.sources:
            discovered = await source.discover()
            if isinstance(discovered, list):
                discovered = CapabilityDiscoveryResult(tools=discovered)
            combined.tools.extend(discovered.tools)
            combined.successful_sources.extend(discovered.successful_sources)
            combined.failed_sources.update(discovered.failed_sources)
            combined.authoritative = (
                combined.authoritative and discovered.authoritative
            )

        combined.tools = [
            tool
            for tool in combined.tools
            if str(tool.get("capability_id") or tool.get("id") or tool.get("name"))
            not in self.suppressed_capabilities
        ]
        if combined.failed_sources:
            combined.authoritative = False
        return combined


def _nested_value(payload: Any, path: str) -> Any:
    value = payload
    if not path:
        return value
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def load_mcp_urls(path: str | Path) -> list[str]:
    """
    Load MCP server URLs from a JSON catalog.

    Expected format:

    {
        "items": [
            {"url": "https://example.com/mcp"}
        ]
    }
    """
    catalog_path = Path(path)

    with catalog_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        catalog = json.load(file)

    items = catalog.get("items")

    if not isinstance(items, list):
        raise ValueError("MCP catalog must contain an 'items' list.")

    return [item["url"] for item in items if item.get("url")]


def load_mcp_servers(path: str | Path) -> list[dict[str, Any]]:
    """Load MCP records while preserving stable catalog server IDs."""
    catalog_path = Path(path)
    with catalog_path.open("r", encoding="utf-8") as file:
        catalog = json.load(file)

    items = catalog.get("items")
    if not isinstance(items, list):
        raise ValueError("MCP catalog must contain an 'items' list.")

    return [
        item
        for item in items
        if isinstance(item, dict) and item.get("url")
    ]
