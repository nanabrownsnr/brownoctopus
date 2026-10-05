from importlib.metadata import PackageNotFoundError, version as package_version

from brown_octopus.octopus import Octopus
from brown_octopus.index import OctopusIndex
from brown_octopus.device import detect_compute_device
from brown_octopus.embedding import (
    EmbeddingProvider,
    HttpEmbeddingProvider,
    LocalEmbeddingProvider,
)
from brown_octopus.config import OctopusConfig
from brown_octopus.contracts import RetrievalResult
from brown_octopus.contracts import (
    CapabilityDiscoveryResult,
    CapabilitySource,
    SessionState,
    SessionStore,
)
from brown_octopus.session_store import InMemorySessionStore
from brown_octopus.sources import (
    ApiCapabilitySource,
    CapabilityUpdateReport,
    JsonCapabilitySource,
    LocalMcpCatalogSource,
    McpRegistrySource,
    McpServerSource,
)

try:
    __version__ = package_version("brown-octopus")
except PackageNotFoundError:
    # Keep source-checkout imports useful before the project is installed.
    __version__ = "0.4.20"

__all__ = [
    "Octopus",
    "OctopusIndex",
    "detect_compute_device",
    "EmbeddingProvider",
    "HttpEmbeddingProvider",
    "LocalEmbeddingProvider",
    "OctopusConfig",
    "RetrievalResult",
    "CapabilitySource",
    "SessionState",
    "SessionStore",
    "InMemorySessionStore",
    "CapabilityDiscoveryResult",
    "CapabilityUpdateReport",
    "LocalMcpCatalogSource",
    "McpServerSource",
    "McpRegistrySource",
    "JsonCapabilitySource",
    "ApiCapabilitySource",
    "__version__",
]
