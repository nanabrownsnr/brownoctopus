import brown_octopus.retriever as retriever
import pytest
import torch

from brown_octopus import Octopus, OctopusIndex
from brown_octopus.index_store import save_snapshot_atomic


def test_empty_index_returns_no_candidates(monkeypatch):
    monkeypatch.setattr(retriever, "model", object())
    monkeypatch.setattr(retriever, "embedding_provider", None)
    monkeypatch.setattr(retriever, "TOOLS", [])
    monkeypatch.setattr(retriever, "tool_embeddings", None)

    assert retriever.rank_tools("reply to an email") == []


def test_missing_embeddings_for_nonempty_index_still_raises(monkeypatch):
    monkeypatch.setattr(retriever, "model", object())
    monkeypatch.setattr(retriever, "embedding_provider", None)
    monkeypatch.setattr(retriever, "TOOLS", [{"name": "reply"}])
    monkeypatch.setattr(retriever, "tool_embeddings", None)

    try:
        retriever.rank_tools("reply to an email")
    except RuntimeError as exc:
        assert str(exc) == "Tool index is not loaded."
    else:
        raise AssertionError("A non-empty index without embeddings must fail")


@pytest.mark.anyio
async def test_runtime_initializes_without_a_persisted_index(monkeypatch, tmp_path):
    monkeypatch.setattr("brown_octopus.octopus.initialize_analyzer", lambda: None)
    monkeypatch.setattr("brown_octopus.octopus.initialize_retriever", lambda: None)

    octopus = Octopus(index_path=tmp_path / "not-created-yet")
    await octopus.initialize()
    octopus.analyzer.analyze = lambda query: []

    result = octopus.retrieve_result(
        "Reply to an email",
        session_id="empty-session",
    )

    assert result.tools == []
    assert result.retrieved_tools == []
    assert result.tool_ids == []
    assert result.turn == 1


def test_index_can_be_opened_without_sources(tmp_path):
    index = OctopusIndex(index_path=tmp_path / "not-created-yet")

    assert index.sources == []
    assert index.runtime() is index.octopus


def test_remove_capability_is_not_public():
    index = OctopusIndex()

    assert not hasattr(index, "remove_capability")


@pytest.mark.anyio
async def test_remove_source_filters_persisted_tools_by_mcp_url(
    monkeypatch,
    tmp_path,
):
    index_path = tmp_path / "index"
    index = OctopusIndex(index_path=index_path)
    first_url = "https://example.com/first/mcp"
    second_url = "https://example.com/second/mcp"
    tools = [
        {
            "capability_id": "first_tool",
            "source_id": first_url,
            "mcp_url": first_url,
            "name": "first_tool",
        },
        {
            "capability_id": "second_tool",
            "source_id": second_url,
            "mcp_url": second_url,
            "name": "second_tool",
        },
    ]
    save_snapshot_atomic(
        index_path,
        tools,
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        {"tool_count": 2},
    )
    monkeypatch.setattr(index._octopus, "_build_pipeline", lambda: None)

    report = await index.remove_source(first_url)

    assert report.removed == ["first_tool"]
    assert report.tool_count == 1
    assert index._octopus.index_store.load_tools() == [tools[1]]
