import json

import pytest

from octopus.sources import load_mcp_urls


def test_load_mcp_urls_from_json(tmp_path):
    catalog = {
        "items": [
            {
                "name": "Web Search",
                "url": "https://example.com/web-search/mcp",
            },
            {
                "name": "GitHub",
                "url": "https://example.com/github/mcp",
            },
        ]
    }

    path = tmp_path / "mcps.json"
    path.write_text(
        json.dumps(catalog),
        encoding="utf-8",
    )

    urls = load_mcp_urls(path)

    assert urls == [
        "https://example.com/web-search/mcp",
        "https://example.com/github/mcp",
    ]


def test_load_mcp_urls_ignores_items_without_url(tmp_path):
    catalog = {
        "items": [
            {
                "name": "Web Search",
                "url": "https://example.com/web-search/mcp",
            },
            {
                "name": "Missing URL",
            },
        ]
    }

    path = tmp_path / "mcps.json"
    path.write_text(
        json.dumps(catalog),
        encoding="utf-8",
    )

    urls = load_mcp_urls(path)

    assert urls == [
        "https://example.com/web-search/mcp",
    ]


def test_load_mcp_urls_allows_empty_items_list(tmp_path):
    catalog = {
        "items": [],
    }

    path = tmp_path / "mcps.json"
    path.write_text(
        json.dumps(catalog),
        encoding="utf-8",
    )

    assert load_mcp_urls(path) == []


def test_load_mcp_urls_rejects_missing_items_list(tmp_path):
    path = tmp_path / "mcps.json"
    path.write_text(
        json.dumps({}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="'items' list",
    ):
        load_mcp_urls(path)


def test_load_mcp_urls_rejects_invalid_items_type(tmp_path):
    catalog = {
        "items": {
            "name": "Web Search",
            "url": "https://example.com/web-search/mcp",
        }
    }

    path = tmp_path / "mcps.json"
    path.write_text(
        json.dumps(catalog),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="'items' list",
    ):
        load_mcp_urls(path)
