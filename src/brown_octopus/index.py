"""High-level capability-index setup facade."""

from pathlib import Path
from collections.abc import Sequence
import warnings

from brown_octopus.contracts import CapabilitySource
from brown_octopus.embedding import EmbeddingProvider
from brown_octopus.index_store import IndexStore
from brown_octopus.octopus import Octopus
from brown_octopus.sources import (
    ApiCapabilitySource,
    CompositeCapabilitySource,
    JsonCapabilitySource,
    McpRegistrySource,
    McpServerSource,
)


class OctopusIndex:
    """Create and maintain a persisted Brown Octopus capability index.

    This class hides source composition and index lifecycle details. The
    returned ``Octopus`` runtime remains available through ``runtime()``.
    """

    def __init__(
        self,
        sources: Sequence[CapabilitySource] | None = None,
        *,
        index_path: str | Path = "data/indexes/default",
        session_store=None,
        config=None,
        embedding_provider: EmbeddingProvider | None = None,
        index_store: IndexStore | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.sources = list(sources or [])
        self.source = CompositeCapabilitySource(self.sources)
        self._octopus = Octopus(
            index_path=self.index_path,
            capability_source=self.source,
            session_store=session_store,
            config=config,
            embedding_provider=embedding_provider,
            index_store=index_store,
        )

    def _sync_sources(self) -> None:
        """Keep the composite source aligned with the public source list."""
        self.source.sources = list(self.sources)

    @classmethod
    def from_sources(
        cls,
        sources: Sequence[CapabilitySource],
        **kwargs,
    ) -> "OctopusIndex":
        return cls(sources, **kwargs)

    @classmethod
    def from_mcp_server(
        cls,
        url: str,
        *,
        source_id: str | None = None,
        **kwargs,
    ) -> "OctopusIndex":
        return cls(
            [McpServerSource(url, source_id=source_id)],
            **kwargs,
        )

    @classmethod
    def from_mcp_registry(
        cls,
        url: str,
        **kwargs,
    ) -> "OctopusIndex":
        """Build an index from a registry that returns MCP server records."""
        source_kwargs = {
            key: kwargs.pop(key)
            for key in tuple(kwargs)
            if key
            in {
                "items_path",
                "server_url_field",
                "server_id_field",
                "page_param",
                "limit_param",
                "cursor_param",
                "next_cursor_paths",
                "page_size",
                "max_pages",
                "headers",
                "refresh_headers",
                "timeout",
                "source_id",
                "tool_discoverer",
            }
        }
        return cls([McpRegistrySource(url, **source_kwargs)], **kwargs)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        items_path: str = "capabilities",
        **kwargs,
    ) -> "OctopusIndex":
        return cls(
            [JsonCapabilitySource(path, items_path=items_path)],
            **kwargs,
        )

    @classmethod
    def from_api(
        cls,
        url: str,
        *,
        items_path: str = "items",
        fields: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 15.0,
        source_id: str | None = None,
        **kwargs,
    ) -> "OctopusIndex":
        return cls(
            [
                ApiCapabilitySource(
                    url,
                    items_path=items_path,
                    fields=fields,
                    headers=headers,
                    timeout=timeout,
                    source_id=source_id,
                )
            ],
            **kwargs,
        )

    async def create(self):
        """Create the initial index, or synchronize an existing one."""
        return await self.update()

    async def update(self):
        """Synchronize all configured sources with the persisted index."""
        return await self._octopus.update()

    async def add(self, source: CapabilitySource):
        """Add a source and synchronize the complete configured source set."""
        self.sources.append(source)
        self._sync_sources()
        try:
            return await self.update()
        except Exception:
            self.sources.pop()
            self._sync_sources()
            raise

    async def remove_source(self, source_id: str):
        """Remove a configured source and synchronize the remaining sources."""
        original = list(self.sources)
        remaining = [
            source
            for source in self.sources
            if getattr(source, "source_id", None) != source_id
        ]
        if len(remaining) == len(original):
            raise ValueError(f"No configured source has ID '{source_id}'.")
        self.sources[:] = remaining
        self._sync_sources()
        try:
            return await self.update()
        except Exception:
            self.sources[:] = original
            self._sync_sources()
            raise

    async def remove_capability(self, capability_id: str):
        """Suppress one capability from future snapshots from these sources.

        .. deprecated::
           Prefer :meth:`remove_source` or update the authoritative source.
           This method remains temporarily for compatibility with existing
           applications.
        """
        warnings.warn(
            "OctopusIndex.remove_capability() is deprecated; use "
            "remove_source() or update the authoritative CapabilitySource.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.source.suppressed_capabilities.add(capability_id)
        try:
            return await self.update()
        except Exception:
            self.source.suppressed_capabilities.discard(capability_id)
            raise

    def runtime(self) -> Octopus:
        """Return the runtime engine using this index's persisted path."""
        return self._octopus

    @property
    def octopus(self) -> Octopus:
        """Access the runtime engine for advanced integrations."""
        return self._octopus

    def reset(self) -> None:
        """Remove the persisted index and clear the in-memory registry.

        A subsequent ``create()`` or ``update()`` is required before runtime
        retrieval can be initialized again.
        """
        self._octopus.reset_index()
        self.source.suppressed_capabilities.clear()
