import json
from pathlib import Path

import pytest

from evals.common.metrics import (
    complete_required,
    context_reduction,
    required_recall,
    supporting_recall,
)
from evals.common.token_count import count_schema_tokens, serialize_tool
from evals.datasets.internal import load_tasks, validate_tasks
from evals.methods.all_tools import AllTools
from evals.methods.bm25 import BM25
from evals.methods.ablation import EvaluationOctopusMethod
from evals.methods.selection import ranked_score_knee, relative_threshold
from evals.diagnostic.run import hybrid_cutoff, rank_bucket
from evals.validation.run import hybrid_selector


def test_metrics_keep_required_and_supporting_capabilities_separate():
    retrieved = ["required_a", "supporting_a"]
    assert required_recall(retrieved, ["required_a", "required_b"]) == 0.5
    assert supporting_recall(retrieved, ["supporting_a", "supporting_b"]) == 0.5
    assert complete_required(retrieved, ["required_a"]) is True
    assert context_reduction(11, 122) == pytest.approx(1 - 11 / 122)


def test_all_returns_entire_universe():
    tools = [{"name": "a"}, {"name": "b"}]
    result = AllTools().retrieve("anything", tools)
    assert result.tool_ids == ["a", "b"]
    assert result.scores is None


def test_internal_dataset_has_fifty_tasks_and_known_ids():
    tools = json.loads(
        Path("data/indexes/default/tools.json").read_text(encoding="utf-8")
    )
    tasks = load_tasks(expected_count=50)
    validate_tasks(tasks, tools)
    assert {task["category"] for task in tasks} == {
        "single_capability",
        "capability_family",
        "compound",
        "hidden_dependency",
        "multi_turn",
    }


def test_dataset_rejects_unknown_tool_ids():
    tasks = load_tasks(expected_count=50)
    tools = [{"name": "known"}]
    with pytest.raises(ValueError, match="unknown"):
        validate_tasks(tasks[:1], tools)


def test_dataset_has_planned_category_distribution():
    counts = {}
    for task in load_tasks(expected_count=50):
        counts[task["category"]] = counts.get(task["category"], 0) + 1
    assert counts == {
        "single_capability": 10,
        "capability_family": 10,
        "compound": 10,
        "hidden_dependency": 10,
        "multi_turn": 10,
    }


def test_bm25_returns_configured_k():
    tools = [
        {"name": "search_web", "description": "search the web"},
        {"name": "send_email", "description": "send an email"},
        {"name": "create_document", "description": "create a document"},
    ]
    method = BM25(2)
    method.initialize(tools)
    result = method.retrieve("search the web", tools)
    assert method.name == "bm25_2"
    assert len(result.tool_ids) == 2


def test_schema_token_counter_uses_exact_serialization():
    class FakeTokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return text.split()

    tool = {"name": "tool", "description": "description"}
    assert serialize_tool(tool) == '{"description":"description","name":"tool"}'
    assert count_schema_tokens([tool], FakeTokenizer()) == 1


def test_fixed_k_ablation_uses_requested_k_and_no_memory_does_not_retain(monkeypatch):
    ranked = [
        {"name": f"tool_{index}", "score": 1.0 - index / 100}
        for index in range(1, 6)
    ]
    monkeypatch.setattr("evals.methods.ablation._cached_intents", lambda query: [{"text": query}])
    monkeypatch.setattr("evals.methods.ablation._cached_ranking", lambda query: ranked)
    monkeypatch.setattr(
        "evals.methods.ablation.get_tools",
        lambda names: [{"name": name} for name in names],
    )

    fixed = EvaluationOctopusMethod("fixed", "fixed_k", 2)
    first = fixed.retrieve("first", [{"name": item["name"]} for item in ranked])
    assert first.tool_ids == ["tool_1", "tool_2"]

    no_memory = EvaluationOctopusMethod("no_memory", "no_memory")
    first = no_memory.retrieve("first", [{"name": item["name"]} for item in ranked])
    second = no_memory.retrieve("second", [{"name": item["name"]} for item in ranked])
    assert first.metadata["retained_tool_ids"] == []
    assert second.metadata["retained_tool_ids"] == []


def test_relative_threshold_selects_prefix_from_top_score():
    ranked = [
        {"name": "tool_1", "rank": 1, "score": 1.0},
        {"name": "tool_2", "rank": 2, "score": 0.97},
        {"name": "tool_3", "rank": 3, "score": 0.80},
    ]
    selected, metadata = relative_threshold(ranked, 0.95)
    assert [tool["name"] for tool in selected] == ["tool_1", "tool_2"]
    assert metadata["cutoff"] == 2


def test_ranked_score_knee_returns_bounded_prefix():
    ranked = [
        {"name": "tool_1", "rank": 1, "score": 1.00},
        {"name": "tool_2", "rank": 2, "score": 0.99},
        {"name": "tool_3", "rank": 3, "score": 0.50},
        {"name": "tool_4", "rank": 4, "score": 0.49},
    ]
    selected, metadata = ranked_score_knee(ranked)
    assert 1 <= metadata["cutoff"] <= 4
    assert [tool["name"] for tool in selected] == [
        tool["name"] for tool in ranked[: metadata["cutoff"]]
    ]


def test_hybrid_selector_has_fixed_four_floor_and_sixteen_ceiling():
    assert hybrid_cutoff(1) == 4
    assert hybrid_cutoff(4) == 4
    assert hybrid_cutoff(9) == 9
    assert hybrid_cutoff(20) == 16


def test_fixed_four_failure_rank_buckets_are_explicit():
    assert rank_bucket(5) == "rank_5"
    assert rank_bucket(8) == "rank_8"
    assert rank_bucket(9) == "rank_9_16"
    assert rank_bucket(16) == "rank_9_16"
    assert rank_bucket(17) == "rank_gt_16"


def test_validation_benchmark_is_independent_and_balanced():
    development = load_tasks("data/evals/octopus_capability_benchmark.json", expected_count=50)
    validation = load_tasks("data/evals/octopus_validation_benchmark_v1.json", expected_count=100)
    assert not ({task["query"] for task in development} & {task["query"] for task in validation})
    assert {task["category"] for task in validation} == {
        "single_capability",
        "capability_family",
        "compound",
        "hidden_dependency",
        "multi_turn",
    }
    assert all(sum(task["category"] == category for task in validation) == 20 for category in {
        "single_capability", "capability_family", "compound", "hidden_dependency", "multi_turn"
    })


def test_validation_hybrid_selector_preserves_max_gap_and_floor(monkeypatch):
    ranked = [
        {"name": f"tool_{index}", "rank": index, "score": 1 - index / 100}
        for index in range(1, 10)
    ]
    monkeypatch.setattr(
        "evals.validation.run.bounded_max_gap",
        lambda ranking: (ranking[:2], {"cutoff": 2, "reason": "bounded_max_gap"}),
    )
    selected, metadata = hybrid_selector(ranked)
    assert len(selected) == 4
    assert metadata["max_gap_cutoff"] == 2
