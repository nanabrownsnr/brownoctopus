"""High-level capability-index setup facade."""

from pathlib import Path
from collections.abc import Sequence

from brown_octopus.contracts import CapabilitySource
from brown_octopus.octopus import Octopus
from brown_octopus.sources import (
    ApiCapabilitySource,
    CompositeCapabilitySource,
    JsonCapabilitySource,
    McpServerSource,
)


class OctopusIndex:
    """Create and maintain a persisted Brown Octopus capability index.

    This class hides source composition and index lifecycle details. The
    returned ``Octopus`` runtime remains available through ``runtime()``.
    """

    def __init__(
        self,
        sources: Sequence[CapabilitySource],
        *,
        index_path: str | Path = "data/indexes/default",
        session_store=None,
        config=None,
    ) -> None:
        self.index_path = Path(index_path)
        self.sources = list(sources)
        self.source = CompositeCapabilitySource(self.sources)
        self._octopus = Octopus(
            index_path=self.index_path,
            capability_source=self.source,
            session_store=session_store,
            config=config,
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
        """Suppress one capability from future snapshots from these sources."""
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
