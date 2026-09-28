import json
from dataclasses import dataclass
from pathlib import Path

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
        urls = load_mcp_urls(self.catalog_path)
        result = CapabilityDiscoveryResult()
        for url in urls:
            try:
                tools = await discover_tools(url)
            except Exception as exc:
                result.failed_sources[url] = str(exc)
                continue
            result.successful_sources.append(url)
            result.tools.extend(tools)
        return result


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
