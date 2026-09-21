from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ToolRetExample:
    query_id: str
    query: str
    instruction: str
    labels: dict[str, int]
    category: str
    dataset: str


def adapt_toolret_tool(row: dict[str, Any]) -> dict[str, Any]:
    """Convert one released ToolRet tool while retaining its exact ID.

    ``name`` is the canonical adapter identifier used by the existing Octopus
    index API.  The original display name is retained separately so the index
    builder can embed the required name+description representation without
    replacing IDs with fuzzy name matching.
    """
    document = row.get("documentation", row.get("doc", row))
    if isinstance(document, str):
        document = json.loads(document)
    tool_id = str(row["id"])
    return {
        "id": tool_id,
        "name": tool_id,
        "toolret_name": str(document.get("name", "")),
        "description": str(document.get("description", "") or ""),
        "input_schema": document.get("parameters", {}),
    }


def tool_embedding_text(tool: dict[str, Any]) -> str:
    return f"{tool.get('toolret_name', tool['name'])}: {tool.get('description') or ''}"


def load_toolret_tools(root: str | Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load the official parquet tool shards for Web, Code, and Custom."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("ToolRet loading requires pyarrow") from exc

    root = Path(root)
    tools: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    for category in ("web", "code", "customized"):
        path = root / "tools" / category / "tools-00000-of-00001.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        sources[category] = str(path)
        for row in parquet.read_table(path).to_pylist():
            tools.append(adapt_toolret_tool(row))
    ids = [tool["id"] for tool in tools]
    if len(ids) != len(set(ids)):
        raise ValueError("ToolRet contains duplicate canonical tool IDs")
    return tools, sources


def load_toolret_queries(
    root: str | Path,
    datasets: tuple[str, ...] = ("apibank", "craft-math-algebra", "appbench"),
) -> tuple[list[ToolRetExample], dict[str, str]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("ToolRet loading requires pyarrow") from exc

    root = Path(root)
    examples: list[ToolRetExample] = []
    sources: dict[str, str] = {}
    for dataset in datasets:
        path = root / "queries" / dataset / "queries-00000-of-00001.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        sources[dataset] = str(path)
        for row in parquet.read_table(path).to_pylist():
            labels = {
                str(label["id"]): int(label.get("relevance", 0))
                for label in json.loads(row["labels"])
            }
            examples.append(
                ToolRetExample(
                    query_id=str(row["id"]),
                    query=str(row["query"]),
                    instruction=str(row.get("instruction", "") or ""),
                    labels=labels,
                    category=str(row.get("category", "")),
                    dataset=dataset,
                )
            )
    return examples, sources
