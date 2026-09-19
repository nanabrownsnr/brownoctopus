from octopus.pipeline import process_turn
from octopus.selection import select_min4_bounded_max_gap


def ranked_tools(count, gap_after=None):
    gap_after = gap_after or count
    return [
        {
            "name": f"tool_{index}",
            "score": (0.90 - index * 0.001) if index < gap_after else (0.50 - index * 0.001),
        }
        for index in range(count)
    ]


def test_v3_simple_intent_uses_four_tool_floor(monkeypatch):
    intents = [{"text": "inspect the project"}]
    ranking = ranked_tools(8, gap_after=1)
    monkeypatch.setattr("octopus.retriever.analyze_intents", lambda query: intents)
    monkeypatch.setattr("octopus.retriever.rank_tools", lambda query: ranking)
    selected = __import__("octopus.retriever", fromlist=["retrieve_tools"]).retrieve_tools("inspect")
    assert [tool["name"] for tool in selected] == [f"tool_{index}" for index in range(4)]


def test_v3_compound_intents_each_get_floor_before_dedup(monkeypatch):
    intents = [{"text": "first action"}, {"text": "second action"}]
    rankings = {
        "first action": ranked_tools(5, gap_after=1),
        "second action": [
            {"name": "tool_4", "score": 0.91},
            {"name": "second_1", "score": 0.89},
            {"name": "second_2", "score": 0.88},
            {"name": "second_3", "score": 0.87},
        ],
    }
    monkeypatch.setattr("octopus.retriever.analyze_intents", lambda query: intents)
    monkeypatch.setattr("octopus.retriever.rank_tools", lambda query: rankings[query])
    selected = __import__("octopus.retriever", fromlist=["retrieve_tools"]).retrieve_tools("compound")
    assert [tool["name"] for tool in selected] == [
        "tool_0", "tool_1", "tool_2", "tool_3", "tool_4",
        "second_1", "second_2", "second_3",
    ]


def test_v3_memory_carries_capability_forward_and_ttl_expires(monkeypatch):
    retrieved = iter([[{"name": "carried"}], [], []])
    monkeypatch.setattr("octopus.pipeline.retrieve_tools", lambda query: next(retrieved))
    first = process_turn("first", {}, 1)
    second = process_turn("follow up", first["active_state"], 1 + 8)
    expired = process_turn("later", second["active_state"], 1 + 8 + 1)
    assert "carried" in first["active_state"]
    assert "carried" in second["active_state"]
    assert "carried" not in expired["active_state"]


def test_v3_global_active_cap_evicts_oldest_capabilities(monkeypatch):
    monkeypatch.setattr(
        "octopus.pipeline.retrieve_tools",
        lambda query: [{"name": "new_a"}, {"name": "new_b"}, {"name": "new_c"}, {"name": "new_d"}],
    )
    active = {f"old_{index}": 10 for index in range(1, 31)}
    active["old_1"] = 6
    active["old_2"] = 7
    result = process_turn("refresh", active, 11)
    assert len(result["active_state"]) == 30
    assert "old_1" not in result["active_state"]
    assert "old_2" not in result["active_state"]
    assert all(name in result["active_state"] for name in ("new_a", "new_b", "new_c", "new_d"))


def test_v3_selector_is_independently_callable():
    assert len(select_min4_bounded_max_gap(ranked_tools(10, gap_after=7))) == 7
