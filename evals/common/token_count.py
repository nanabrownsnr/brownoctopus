import json
from collections.abc import Iterable
from typing import Any


def serialize_tool(tool: dict[str, Any]) -> str:
    """Serialize exactly the definition supplied to an agent."""
    return json.dumps(tool, sort_keys=True, separators=(",", ":"))


def count_schema_tokens(tools: Iterable[dict[str, Any]], tokenizer) -> int:
    return sum(
        len(tokenizer.encode(serialize_tool(tool), add_special_tokens=False))
        for tool in tools
    )
