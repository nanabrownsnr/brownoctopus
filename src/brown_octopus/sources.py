import json
import asyncio
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from inspect import isawaitable
from typing import Any, Awaitable, Callable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

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
            "mcp_url",
            "mcp_name",
            "tool_name",
        }
        tool = {
            field: _nested_value(record, path)
            for field, path in self.fields.items()
            if field in required
        }
        for field in required:
            if (
                field in record
                and field not in tool
                and field not in self.fields
                and field not in self.fields.values()
            ):
                tool[field] = record[field]
        tool.setdefault("source_id", self.source_id)
        tool.setdefault("description", "")
        tool.setdefault("input_schema", {})
        if "capability_id" not in tool or "name" not in tool:
            raise ValueError(
                "Capability API records must map capability_id and name."
            )
        return tool


class McpRegistrySource:
    """Discover MCP tools from a registry that returns MCP server records.

    This is intentionally separate from :class:`ApiCapabilitySource`:
    ``ApiCapabilitySource`` consumes already-normalized capability records,
    while this source fetches server records and discovers tools from each
    server URL.

    ``refresh_headers`` may be supplied by the host for an expired registry
    token. It is called at most once after a 401/403 response. Header values
    are never included in errors or logs.
    """

    def __init__(
        self,
        url: str,
        *,
        items_path: str = "items",
        server_url_field: str = "url",
        server_id_field: str = "id",
        page_param: str = "page",
        limit_param: str = "limit",
        cursor_param: str = "cursor",
        next_cursor_paths: Sequence[str] = (
            "next_cursor",
            "nextCursor",
            "pagination.next_cursor",
            "pagination.nextCursor",
            "meta.next_cursor",
            "meta.nextCursor",
        ),
        page_size: int = 100,
        max_pages: int = 1000,
        headers: Mapping[str, str] | None = None,
        refresh_headers: Callable[
            [], Mapping[str, str] | Awaitable[Mapping[str, str]]
        ]
        | None = None,
        timeout: float = 15.0,
        source_id: str | None = None,
        tool_discoverer: Callable[..., Awaitable[list[dict]]] | None = None,
    ) -> None:
        if page_size <= 0 or max_pages <= 0:
            raise ValueError("page_size and max_pages must be positive.")
        self.url = url
        self.items_path = items_path
        self.server_url_field = server_url_field
        self.server_id_field = server_id_field
        self.page_param = page_param
        self.limit_param = limit_param
        self.cursor_param = cursor_param
        self.next_cursor_paths = tuple(next_cursor_paths)
        self.page_size = page_size
        self.max_pages = max_pages
        self.headers = dict(headers or {})
        self.refresh_headers = refresh_headers
        self.timeout = timeout
        self.source_id = source_id or url
        self.tool_discoverer = tool_discoverer or discover_tools

    async def discover(self) -> CapabilityDiscoveryResult:
        result = CapabilityDiscoveryResult()
        headers = dict(self.headers)
        refreshed = False
        page = 1
        cursor: str | None = None
        seen_cursors: set[str] = set()

        while page <= self.max_pages:
            try:
                payload, status_code = await asyncio.to_thread(
                    self._request_page,
                    page,
                    headers,
                    cursor,
                )
                if status_code in {401, 403} and self.refresh_headers and not refreshed:
                    refreshed = True
                    refreshed_headers = self.refresh_headers()
                    if isawaitable(refreshed_headers):
                        refreshed_headers = await refreshed_headers
                    headers = dict(refreshed_headers)
                    continue
                if status_code in {401, 403}:
                    raise RuntimeError(
                        f"registry authorization failed (HTTP {status_code})"
                    )
                records, next_cursor, has_more = self._records_from_page(payload)
            except Exception as exc:
                result.failed_sources[self.source_id] = str(exc)
                result.authoritative = False
                return result

            for record in records:
                await self._discover_server(record, result)

            if not has_more:
                break
            if next_cursor is not None:
                if next_cursor in seen_cursors:
                    result.failed_sources[self.source_id] = (
                        "registry returned a repeated pagination cursor"
                    )
                    result.authoritative = False
                    return result
                seen_cursors.add(next_cursor)
                cursor = next_cursor
            else:
                page += 1
        else:
            result.failed_sources[self.source_id] = (
                f"registry exceeded max_pages={self.max_pages}"
            )
            result.authoritative = False

        if result.failed_sources:
            result.authoritative = False
        if not result.failed_sources:
            result.successful_sources.append(self.source_id)
        return result

    async def _discover_server(
        self,
        record: dict[str, Any],
        result: CapabilityDiscoveryResult,
    ) -> None:
        mcp_url = record.get(self.server_url_field)
        if not isinstance(mcp_url, str) or not mcp_url:
            return
        server_id = str(record.get(self.server_id_field) or mcp_url)
        try:
            tools = await self.tool_discoverer(mcp_url, server_id=server_id)
            for tool in tools:
                tool.setdefault("source_id", server_id)
                tool.setdefault("mcp_url", mcp_url)
                if record.get("name"):
                    tool.setdefault("mcp_name", record["name"])
                result.tools.append(tool)
            result.successful_sources.append(server_id)
        except Exception as exc:
            result.failed_sources[server_id] = str(exc)

    def _request_page(
        self,
        page: int,
        headers: Mapping[str, str],
        cursor: str | None = None,
    ) -> tuple[Any, int]:
        values: dict[str, Any] = {self.limit_param: self.page_size}
        if cursor is not None:
            values[self.cursor_param] = cursor
        else:
            values[self.page_param] = page
        request_url = _with_query(self.url, values)
        request = urllib.request.Request(request_url, headers=dict(headers))
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8")), response.status
        except urllib.error.HTTPError as exc:
            if exc.code in {401, 403}:
                return None, exc.code
            raise

    def _records_from_page(
        self,
        payload: Any,
    ) -> tuple[list[dict], str | None, bool]:
        records = payload if isinstance(payload, list) else _nested_value(
            payload, self.items_path
        )
        if not isinstance(records, list):
            raise ValueError(
                f"MCP registry field '{self.items_path}' must contain a list."
            )
        records = [record for record in records if isinstance(record, dict)]
        if isinstance(payload, list):
            return records, None, len(records) >= self.page_size
        next_cursor = next(
            (
                _nested_value(payload, path)
                for path in self.next_cursor_paths
                if _nested_value(payload, path) not in (None, "")
            ),
            None,
        )
        if next_cursor is not None:
            return records, str(next_cursor), True
        next_value = (
            _nested_value(payload, "next")
            or _nested_value(payload, "next_url")
            or _nested_value(payload, "links.next")
            or _nested_value(payload, "pagination.next")
        )
        if next_value:
            return records, None, True
        return records, None, len(records) >= self.page_size


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


def _with_query(url: str, values: Mapping[str, Any]) -> str:
    """Add or replace query parameters without dropping existing filters."""
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query.update({key: str(value) for key, value in values.items()})
    return urlunsplit(
        (parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment)
    )


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
