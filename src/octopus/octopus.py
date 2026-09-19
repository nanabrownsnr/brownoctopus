from pathlib import Path

from octopus.analyzer import initialize_analyzer
from octopus.index_store import (
    load_embeddings,
    load_tools,
    save_embeddings,
    load_metadata,
    save_metadata,
    save_tools,
)
from octopus.mcp_discovery import discover_universe
from octopus.pipeline import process_turn
from octopus.retriever import (
    initialize_retriever,
    load_index,
    refresh_index,
    retrieve_tools,
)
from octopus.sources import load_mcp_urls
from octopus.tool_registry import set_tools


class Octopus:
    def __init__(
        self,
        index_path: str | Path = "data/indexes/default",
        catalog_path: str | Path = "data/mcps.json",
    ):
        self.index_path = Path(index_path)
        self.catalog_path = Path(catalog_path)

        self.turn = 0
        self.active_tools = {}

    async def initialize(self) -> list[dict]:
        initialize_analyzer()
        initialize_retriever()

        tools_path = self.index_path / "tools.json"
        embeddings_path = self.index_path / "embeddings.pt"
        metadata_path = self.index_path / "metadata.json"

        if (
            not tools_path.exists()
            or not embeddings_path.exists()
            or not metadata_path.exists()
        ):
            raise RuntimeError(
                f"Octopus index not found at '{self.index_path}'. "
                "Run `await octopus.update()` to build the index first."
            )

        metadata = load_metadata(self.index_path)

        if metadata.get("version") != 1:
            raise RuntimeError(
                "Unsupported Octopus index version: " f"{metadata.get('version')}"
            )

        tools = load_tools(self.index_path)

        embeddings = load_embeddings(self.index_path)

        set_tools(tools)

        load_index(
            tools=tools,
            embeddings=embeddings,
        )

        return tools

    def retrieve(self, query: str) -> list[dict]:
        return retrieve_tools(query)

    def process(self, query: str) -> dict:
        self.turn += 1

        result = process_turn(
            query=query,
            active_tools=self.active_tools,
            turn=self.turn,
        )

        self.active_tools = result["active_state"]

        return result

    def reset(self) -> None:
        self.turn = 0
        self.active_tools = {}

    async def update(self) -> list[dict]:
        initialize_analyzer()
        initialize_retriever()

        mcp_urls = load_mcp_urls(self.catalog_path)

        tools = await discover_universe(mcp_urls)

        set_tools(tools)

        embeddings = refresh_index()

        save_tools(
            self.index_path,
            tools,
        )

        save_embeddings(
            self.index_path,
            embeddings,
        )

        save_metadata(
            self.index_path,
            {
                "version": 1,
                "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
                "tool_count": len(tools),
            },
        )

        return tools
