class OctopusError(RuntimeError):
    """Base class for runtime Octopus failures."""


class ConfigurationError(OctopusError):
    """The service is not configured for the requested operation."""


class StrategyUnavailableError(OctopusError):
    """The configured selection strategy cannot be used."""


class IndexNotReadyError(OctopusError):
    """The persisted capability index is missing or not ready."""


class MissingIndexError(IndexNotReadyError):
    """The configured capability index does not exist or is incomplete."""


class InitializationError(OctopusError):
    """Brown Octopus could not load its required local runtime resources."""


class ModelInitializationError(InitializationError):
    """A required local NLP or embedding model could not be loaded."""


class IndexLoadError(InitializationError):
    """A persisted capability index could not be read."""


class IncompatibleIndexError(InitializationError):
    """A persisted capability index uses an unsupported format."""


class CapabilityUpdateError(OctopusError):
    """A capability snapshot could not be prepared or installed safely."""
