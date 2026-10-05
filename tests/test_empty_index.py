import brown_octopus.retriever as retriever
import pytest

from brown_octopus import Octopus, OctopusIndex


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


@pytest.mark.anyio
async def test_remove_capability_is_compatibility_deprecated(monkeypatch, tmp_path):
    index = OctopusIndex(index_path=tmp_path / "index")

    async def fake_update():
        return None

    monkeypatch.setattr(index._octopus, "update", fake_update)

    with pytest.warns(DeprecationWarning, match="remove_capability"):
        await index.remove_capability("legacy-capability")
