"""Request-scoped capability filtering helpers."""

from collections.abc import Iterable


def normalize_mcp_url(url: str) -> str:
    """Normalize harmless URL formatting differences for exact matching."""
    return url.strip().rstrip("/")


def normalize_allowed_mcp_urls(
    allowed_mcp_urls: Iterable[str] | None,
) -> set[str] | None:
    if allowed_mcp_urls is None:
        return None
    return {
        normalize_mcp_url(url)
        for url in allowed_mcp_urls
        if isinstance(url, str) and url.strip()
    }


def tool_is_allowed(
    tool: dict,
    allowed_mcp_urls: set[str] | None,
) -> bool:
    """Return whether a tool is in the request's MCP URL scope."""
    if allowed_mcp_urls is None:
        return True

    mcp_url = tool.get("mcp_url")
    return isinstance(mcp_url, str) and normalize_mcp_url(mcp_url) in allowed_mcp_urls
