"""Cross-platform locations and runtime loading for Brown Octopus models."""

from __future__ import annotations

import os
import sys
from importlib.metadata import entry_points
from pathlib import Path


SPACY_MODEL_NAME = "en_core_web_trf"
SPACY_MODEL_VERSION = "3.8.0"
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"


def model_directory() -> Path:
    """Return the persistent Brown Octopus model directory.

    ``BROWN_OCTOPUS_MODEL_DIR`` is intended for containers, shared volumes,
    and enterprise deployments. The default follows the platform's normal
    per-user data conventions without depending on the consumer project.
    """
    override = os.environ.get("BROWN_OCTOPUS_MODEL_DIR")
    if override:
        return Path(override).expanduser()

    if sys.platform == "win32":
        root = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
        return Path(root) / "Brown Octopus" / "models"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Brown Octopus" / "models"

    root = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    return Path(root) / "brown-octopus" / "models"


def spacy_model_path() -> Path:
    """Return the canonical managed spaCy model path."""
    # The model wheel installs a package wrapper plus its actual spaCy model
    # data directory. ``spacy.load`` needs the latter because it contains
    # config.cfg.
    return model_directory() / SPACY_MODEL_NAME / f"{SPACY_MODEL_NAME}-{SPACY_MODEL_VERSION}"


def prepare_spacy_runtime() -> Path | None:
    """Make managed spaCy plugin packages importable and return model path.

    The managed directory contains the model and its curated-transformer
    plugin distributions. It is added to ``sys.path`` only when present.
    ``None`` means the managed model is absent and callers may use the legacy
    installed-package fallback.
    """
    path = spacy_model_path()
    if not path.is_dir():
        return None
    root = str(model_directory())
    if root not in sys.path:
        sys.path.insert(0, root)
    return path


def register_managed_spacy_plugins() -> None:
    """Register bundled spaCy plugin factories in the current interpreter."""
    if prepare_spacy_runtime() is None:
        return
    try:
        import spacy_curated_transformers.models  # noqa: F401
        import spacy_curated_transformers.tokenization  # noqa: F401
        import spacy_curated_transformers.pipeline.transformer  # noqa: F401
        from spacy.util import registry

        for entry_point in entry_points(group="spacy_architectures"):
            if not entry_point.name.startswith("spacy-curated-transformers."):
                continue
            if entry_point.name not in registry.architectures:
                registry.architectures(entry_point.name)(entry_point.load())
    except ImportError:
        pass


def resolve_spacy_model() -> str:
    """Resolve the model source used by runtime initialization.

    Managed storage is preferred. The installed package fallback preserves
    pre-release environments that already prepared the model the old way.
    """
    managed = prepare_spacy_runtime()
    if managed is not None:
        return str(managed)
    return SPACY_MODEL_NAME
