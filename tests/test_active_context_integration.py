from concurrent.futures import ThreadPoolExecutor

from brown_octopus.context_manager import ActiveCapabilityContext
from brown_octopus.octopus import Octopus
from brown_octopus.pipeline import CapabilityPipeline
from brown_octopus.selection import select_min4_bounded_max_gap
from brown_octopus.session_store import InMemorySessionStore
from brown_octopus.strategies import BoundedMaxGapSelectionStrategy


TOOLS = [
    {"capability_id": "weather:search", "name": "web_search", "description": "Search weather"},
    {"capability_id": "weather:forecast", "name": "weather_forecast", "description": "Get forecast"},
    {"capability_id": "ghana:search", "name": "ghana_search", "description": "Search Ghana"},
]


def lookup(ids):
    by_id = {tool["capability_id"]: tool for tool in TOOLS}
    return [by_id[identity] for identity in ids if identity in by_id]


class Analyzer:
    def analyze(self, query):
        return [{"text": query}]


class Retriever:
    def retrieve(self, query, intent):
        if query == "search for weather in USA today":
            return [dict(TOOLS[0], rank=1, score=0.9)]
        if query == "wbu ghana":
            return [dict(TOOLS[2], rank=1, score=0.9)]
        return []


class Selector:
    name = "brown_octopus_v3"

    def select(self, query, intent, candidates):
        return candidates


def make_pipeline():
    context = ActiveCapabilityContext(lookup, ttl=8, active_cap=30)
    return CapabilityPipeline(Analyzer(), Retriever(), Selector(), context)


def make_engine():
    engine = Octopus()
    engine.pipeline = CapabilityPipeline(Analyzer(), Retriever(), Selector())
    engine.sessions = InMemorySessionStore()
    return engine


def test_nonmatching_followup_keeps_capability_active_within_ttl():
    pipeline = make_pipeline()

    first = pipeline.process("search for weather in USA today", 1)
    second = pipeline.process("wbu ghana", 2)

    assert [tool["capability_id"] for tool in first["retrieved_tools"]] == ["weather:search"]
    assert [tool["capability_id"] for tool in second["retrieved_tools"]] == ["ghana:search"]
    assert {tool["capability_id"] for tool in second["active_tools"]} == {
        "weather:search",
        "ghana:search",
    }


def test_active_capability_survives_nonmatching_turns_and_expires_after_ttl():
    pipeline = make_pipeline()
    pipeline.process("search for weather in USA today", 1)

    at_expiry = pipeline.process("unrelated", 9)
    after_expiry = pipeline.process("unrelated", 10)

    assert "weather:search" in at_expiry["active_state"]
    assert "weather:search" not in after_expiry["active_state"]


def test_reretrieval_refreshes_ttl():
    pipeline = make_pipeline()
    pipeline.process("search for weather in USA today", 1)
    pipeline.process("search for weather in USA today", 9)
    result = pipeline.process("unrelated", 17)

    assert "weather:search" in result["active_state"]


def test_deduplication_uses_capability_id_not_readable_name():
    same_name = [
        {"capability_id": "server-a:search", "name": "search"},
        {"capability_id": "server-b:search", "name": "search"},
    ]
    context = ActiveCapabilityContext(lambda ids: same_name, ttl=8, active_cap=30)
    result = context.update(same_name, 1)

    assert result.tool_ids == ["server-a:search", "server-b:search"]


def test_active_cap_never_exceeds_thirty_and_evicts_oldest():
    tools = [
        {"capability_id": f"server:tool-{index}", "name": f"tool_{index}"}
        for index in range(31)
    ]
    context = ActiveCapabilityContext(
        lambda ids: [tool for tool in tools if tool["capability_id"] in ids],
        ttl=8,
        active_cap=30,
    )
    result = context.update(tools, 1)

    assert len(result.tools) == 30
    assert "server:tool-0" not in result.active_state


def test_reset_clears_authoritative_context_state():
    pipeline = make_pipeline()
    pipeline.process("search for weather in USA today", 1)
    pipeline.reset()

    assert pipeline.context_manager.active_state == {}


def test_retrieval_result_exposes_active_context_and_current_retrieval():
    pipeline = make_pipeline()
    octopus = Octopus()
    octopus.pipeline = pipeline

    result = octopus.retrieve_result("search for weather in USA today")

    assert [tool["capability_id"] for tool in result.tools] == ["weather:search"]
    assert result.tool_ids == ["weather:search"]
    assert [tool["capability_id"] for tool in result.metadata["retrieved_tools"]] == [
        "weather:search"
    ]


def test_sessions_have_independent_turns_and_active_state():
    engine = make_engine()

    engine.retrieve_result("search for weather in USA today", session_id="a")
    engine.retrieve_result("wbu ghana", session_id="b")

    session_a = engine.get_session("a")
    session_b = engine.get_session("b")
    assert session_a["turn"] == 1
    assert session_b["turn"] == 1
    assert set(session_a["active_state"]) == {"weather:search"}
    assert set(session_b["active_state"]) == {"ghana:search"}


def test_activity_in_one_session_does_not_refresh_another_session_ttl():
    engine = make_engine()
    engine.retrieve_result("search for weather in USA today", session_id="a")

    for turn in range(1, 9):
        engine.retrieve_result("wbu ghana", session_id="b")

    for _ in range(2, 11):
        engine.retrieve_result("unrelated", session_id="a")
    assert "weather:search" not in engine.get_session("a")["active_state"]


def test_reset_and_delete_only_affect_requested_session():
    engine = make_engine()
    engine.retrieve_result("search for weather in USA today", session_id="a")
    engine.retrieve_result("wbu ghana", session_id="b")

    engine.reset_session("a")
    assert engine.get_session("a")["turn"] == 0
    assert engine.get_session("a")["active_state"] == {}
    assert engine.get_session("b")["active_state"]

    engine.delete_session("a")
    assert engine.get_session("a") is None
    assert engine.get_session("b") is not None


def test_default_session_id_is_backward_compatible():
    engine = make_engine()
    result = engine.retrieve_result("search for weather in USA today")

    assert result.session_id == "default"
    assert result.turn == 1
    assert engine.get_session()["turn"] == 1


def test_same_session_concurrent_requests_are_serialized():
    engine = make_engine()

    def request(_):
        return engine.retrieve_result("search for weather in USA today", "same")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(request, range(12)))

    turns = sorted(result.turn for result in results)
    assert turns == list(range(1, 13))
    assert engine.get_session("same")["turn"] == 12


def test_different_sessions_can_process_without_sharing_state():
    engine = make_engine()

    def request(args):
        session_id, query = args
        return engine.retrieve_result(query, session_id)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(request, [("a", "search for weather in USA today"), ("b", "wbu ghana")]))

    assert {result.session_id for result in results} == {"a", "b"}
    assert all(result.turn == 1 for result in results)


def test_current_turn_tools_precede_retained_tools():
    engine = make_engine()
    engine.retrieve_result("search for weather in USA today", session_id="ordered")
    result = engine.retrieve_result("wbu ghana", session_id="ordered")

    assert [tool["capability_id"] for tool in result.tools] == [
        "ghana:search",
        "weather:search",
    ]


def test_frozen_v3_selector_behavior_is_unchanged():
    ranking = [
        {"name": f"tool_{index}", "score": 0.9 - index * 0.001}
        for index in range(8)
    ]
    ranking[1]["score"] = 0.7

    assert len(select_min4_bounded_max_gap(ranking)) == 4
    assert len(BoundedMaxGapSelectionStrategy().select("query", {}, ranking)) == 4
