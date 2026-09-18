import json
from pathlib import Path


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
