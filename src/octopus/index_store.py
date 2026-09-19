import json
from pathlib import Path

import torch


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
    tools_path = Path(index_path) / "tools.json"

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
    embeddings_path = Path(index_path) / "embeddings.pt"

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
    metadata_path = Path(index_path) / "metadata.json"

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)
