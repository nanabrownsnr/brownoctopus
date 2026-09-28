import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path

import torch


def _active_index_path(index_path: str | Path) -> Path:
    """Resolve the atomically published snapshot, falling back to legacy files."""
    root = Path(index_path)
    pointer = root / "current.json"
    if pointer.exists():
        with pointer.open("r", encoding="utf-8") as file:
            snapshot = json.load(file)["snapshot"]
        return root / snapshot
    return root


def save_tools(
    index_path: str | Path,
    tools: list[dict],
) -> None:
    index_path = Path(index_path)
    index_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    tools_path = index_path / "tools.json"

    with tools_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            tools,
            file,
            indent=2,
        )


def load_tools(
    index_path: str | Path,
) -> list[dict]:
    tools_path = _active_index_path(index_path) / "tools.json"

    with tools_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_embeddings(
    index_path: str | Path,
    embeddings,
) -> None:
    index_path = Path(index_path)
    index_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_path = index_path / "embeddings.pt"

    torch.save(
        embeddings,
        embeddings_path,
    )


def load_embeddings(
    index_path: str | Path,
):
    embeddings_path = _active_index_path(index_path) / "embeddings.pt"

    return torch.load(
        embeddings_path,
        weights_only=True,
    )


def save_metadata(
    index_path: str | Path,
    metadata: dict,
) -> None:
    index_path = Path(index_path)
    index_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = index_path / "metadata.json"

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )


def load_metadata(
    index_path: str | Path,
) -> dict:
    metadata_path = _active_index_path(index_path) / "metadata.json"

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_snapshot_atomic(
    index_path: str | Path,
    tools: list[dict],
    embeddings,
    metadata: dict,
) -> None:
    """Write a complete index snapshot before replacing the live files."""
    target = Path(index_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    snapshots = target / "snapshots"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot_name = f"snapshot-{uuid.uuid4().hex}"
    temporary = Path(tempfile.mkdtemp(prefix=".pending-", dir=snapshots))
    published = snapshots / snapshot_name
    try:
        save_tools(temporary, tools)
        save_embeddings(temporary, embeddings)
        save_metadata(temporary, metadata)
        os.replace(temporary, published)
        pointer = target / "current.json"
        pointer_temp = target / f".current-{uuid.uuid4().hex}.json"
        with pointer_temp.open("w", encoding="utf-8") as file:
            json.dump({"snapshot": f"snapshots/{snapshot_name}"}, file)
        os.replace(pointer_temp, pointer)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
