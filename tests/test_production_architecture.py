import pytest

from brown_octopus.config import OctopusConfig
from brown_octopus.context_manager import ActiveCapabilityContext
from brown_octopus.octopus import Octopus
from brown_octopus.pipeline import CapabilityPipeline
from brown_octopus.strategies import BoundedMaxGapSelectionStrategy, JevSelectionStrategy


class FakeAnalyzer:
    def analyze(self, query):
        return [{"text": query, "action": "request"}]


class FakeRetriever:
    def retrieve(self, query, intent):
        return [
            {"name": "one", "description": "first", "rank": 1, "score": 0.9},
            {"name": "two", "description": "second", "rank": 2, "score": 0.8},
        ]


class FakeSelector:
    name = "fake"

    def select(self, query, intent, candidates):
        return candidates[:1]


def test_pipeline_components_are_replaceable():
    context = ActiveCapabilityContext(lambda names: [{"name": name} for name in names])
    pipeline = CapabilityPipeline(FakeAnalyzer(), FakeRetriever(), FakeSelector(), context)

    result = pipeline.process("find something", 1)

    assert result["strategy"] == "fake"
    assert result["tool_ids"] == ["one"]
    assert result["per_intent"][0]["candidate_count"] == 2


def test_jev_strategy_maps_document_indices_and_scores():
    class FakeReranker:
        def relevance_rerank(self, query, documents, **kwargs):
            assert query == "find something"
            assert len(documents) == 2
            assert kwargs["threshold"] == 0.2
            return {"results": [{"document_index": 1, "score": 0.91}]}

    strategy = JevSelectionStrategy(reranker=FakeReranker())
    selected = strategy.select(
        "request",
        {"text": "find something"},
        FakeRetriever().retrieve("", {}),
    )

    assert [tool["name"] for tool in selected] == ["two"]
    assert selected[0]["jev_score"] == 0.91


def test_production_defaults_are_configurable_and_not_selector_parameters():
    config = OctopusConfig.from_env()
    assert config.ttl == 8
    assert config.active_cap == 30


def test_octopus_builds_frozen_v3_by_default():
    octopus = Octopus()
    octopus._build_pipeline()

    assert isinstance(octopus.selection_strategy, BoundedMaxGapSelectionStrategy)
    assert octopus.selection_strategy.name == "brown_octopus_v3"


def test_context_state_isolated_between_instances():
    lookup = lambda names: [{"name": name} for name in names]
    first = ActiveCapabilityContext(lookup)
    second = ActiveCapabilityContext(lookup)

    first.update([{"name": "first"}], 1)

    assert first.active_state == {"first": 1}
    assert second.active_state == {}


def test_jev_missing_dependency_is_clear(monkeypatch):
    strategy = JevSelectionStrategy()
    monkeypatch.setitem(__import__("sys").modules, "jev_reranker", None)
    with pytest.raises(Exception, match="Jev is not installed"):
        strategy.select("query", {"text": "query"}, [{"name": "tool"}])
