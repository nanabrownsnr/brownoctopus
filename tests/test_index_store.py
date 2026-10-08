import json

from brown_octopus.index_store import load_tools
import torch
from brown_octopus.index_store import (
    load_embeddings,
    load_tools,
    save_embeddings,
    save_metadata,
    save_tools,
)


def test_save_and_load_tools(tmp_path):
    tools = [
        {
            "name": "web_search",
            "description": "Search the web",
        },
        {
            "name": "send_email",
            "description": "Send an email",
        },
    ]

    save_tools(
        tmp_path,
        tools,
    )

    loaded_tools = load_tools(tmp_path)

    assert loaded_tools == tools


def test_save_and_load_embeddings(tmp_path):
    embeddings = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ]
    )

    save_embeddings(
        tmp_path,
        embeddings,
    )

    loaded_embeddings = load_embeddings(tmp_path)

    assert torch.equal(
        loaded_embeddings,
        embeddings,
    )


def test_load_embeddings_maps_to_cpu_when_cuda_is_unavailable(monkeypatch, tmp_path):
    captured = {}

    def fake_load(source, **options):
        captured.update(options)
        return torch.tensor([[1.0, 0.0]])

    monkeypatch.setattr("brown_octopus.index_store.torch.load", fake_load)
    monkeypatch.setattr(
        "brown_octopus.index_store.torch.cuda.is_available",
        lambda: False,
    )

    load_embeddings(tmp_path)

    assert captured["weights_only"] is True
    assert captured["map_location"] == torch.device("cpu")


def test_load_tools_from_index(tmp_path):
    tools = [
        {
            "name": "web_search_tools_web_search",
            "description": "Search the web",
        },
        {
            "name": "outlook_mail_management_send_email",
            "description": "Send an email",
        },
    ]

    tools_path = tmp_path / "tools.json"

    tools_path.write_text(
        json.dumps(tools),
        encoding="utf-8",
    )

    loaded_tools = load_tools(tmp_path)

    assert loaded_tools == tools


def test_save_and_load_metadata(tmp_path):
    from brown_octopus.index_store import (
        load_metadata,
        save_metadata,
    )

    metadata = {
        "version": 1,
        "embedding_model": "Qwen/Qwen3-Embedding-0.6B",
        "tool_count": 122,
    }

    save_metadata(
        tmp_path,
        metadata,
    )

    loaded_metadata = load_metadata(tmp_path)

    assert loaded_metadata == metadata
