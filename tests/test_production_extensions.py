import asyncio
from copy import deepcopy
from threading import RLock

from brown_octopus import (
    CapabilityDiscoveryResult,
    InMemorySessionStore,
    Octopus,
    SessionState,
)
from brown_octopus.pipeline import CapabilityPipeline
from brown_octopus.tool_registry import set_tools
from brown_octopus.index_store import load_tools, save_tools


TOOLS = [
    {"capability_id": "weather:search", "name": "web_search", "description": "Search weather"},
    {"capability_id": "ghana:search", "name": "ghana_search", "description": "Search Ghana"},
]


class Analyzer:
    def analyze(self, query):
        return [{"text": query}]


class Retriever:
    def retrieve(self, query, intent):
        if query == "search for weather in USA today":
            return [dict(TOOLS[0], rank=1, score=0.9)]
        if query == "wbu ghana":
            return [dict(TOOLS[1], rank=1, score=0.9)]
        return []


class Selector:
    name = "brown_octopus_v3"

    def select(self, query, intent, candidates):
        return candidates


class FakeSource:
    def __init__(self, result):
        self.result = result

    async def discover(self):
        return self.result


class SerializableStore:
    """Test store persisting only the public SessionState fields."""

    def __init__(self):
        self.records = {}
        self.locks = {}

    def _get(self, session_id):
        if session_id not in self.records:
            self.records[session_id] = SessionState(session_id)
            self.locks[session_id] = RLock()
        return self.records[session_id], self.locks[session_id]

    def get(self, session_id):
        state, lock = self._get(session_id)
        with lock:
            return deepcopy(state)

    def mutate(self, session_id, operation):
        state, lock = self._get(session_id)
        with lock:
            return operation(state)

    def reset(self, session_id):
        def clear(state):
            state.turn = 0
            state.active_state = {}
            return deepcopy(state)

        return self.mutate(session_id, clear)

    def delete(self, session_id):
        self.records.pop(session_id, None)
        self.locks.pop(session_id, None)

    def snapshot(self, session_id):
        if session_id not in self.records:
            return None
        state = self.get(session_id)
        return {
            "session_id": state.session_id,
            "turn": state.turn,
            "active_state": dict(state.active_state),
        }


def test_custom_capability_source_and_session_store_are_injectable(monkeypatch):
    source = FakeSource(CapabilityDiscoveryResult())
    store = InMemorySessionStore()
    engine = Octopus(capability_source=source, session_store=store)

    assert engine.capability_source is source
    assert engine.sessions is store


def test_custom_serializable_store_can_continue_with_a_second_engine():
    set_tools(TOOLS)
    store = SerializableStore()

    first = Octopus(session_store=store)
    first.pipeline = CapabilityPipeline(Analyzer(), Retriever(), Selector())
    second = Octopus(session_store=store)
    second.pipeline = CapabilityPipeline(Analyzer(), Retriever(), Selector())

    first.retrieve_result("search for weather in USA today", session_id="shared")
    result = second.retrieve_result("wbu ghana", session_id="shared")

    assert result.turn == 2
    assert {tool["capability_id"] for tool in result.tools} == {
        "weather:search",
        "ghana:search",
    }
    assert store.records["shared"].turn == 2
    assert set(store.records["shared"].active_state) == {
        "weather:search",
        "ghana:search",
    }


def test_failed_source_preserves_previous_capabilities(tmp_path, monkeypatch):
    previous = [
        {
            "capability_id": "offline:tool",
            "source_id": "twynity:outlook-123",
            "name": "offline_tool",
            "description": "Previously available",
            "mcp_url": "https://offline.example/mcp",
        },
        {
            "capability_id": "online:old",
            "name": "old_tool",
            "description": "Removed authoritatively",
            "mcp_url": "https://online.example/mcp",
        },
    ]
    index_path = tmp_path / "index"
    save_tools(index_path, previous)

    current = [
        {
            "capability_id": "online:new",
            "source_id": "twynity:crm-456",
            "name": "new_tool",
            "description": "Added now",
            "mcp_url": "https://online.example/mcp",
        }
    ]
    engine = Octopus(
        index_path=index_path,
        capability_source=FakeSource(
            CapabilityDiscoveryResult(
                tools=current,
                successful_sources=["twynity:crm-456"],
                failed_sources={"twynity:outlook-123": "timeout"},
            )
        ),
    )
    monkeypatch.setattr("brown_octopus.octopus.initialize_analyzer", lambda: None)
    monkeypatch.setattr("brown_octopus.octopus.initialize_retriever", lambda: None)
    monkeypatch.setattr("brown_octopus.octopus.build_embeddings", lambda tools: "embeddings")
    monkeypatch.setattr("brown_octopus.octopus.save_snapshot_atomic", lambda *args: None)
    monkeypatch.setattr("brown_octopus.octopus.load_index", lambda **kwargs: None)

    report = asyncio.run(engine.update())

    assert report.added == ["online:new"]
    assert report.removed == ["online:old"]
    assert report.failed_sources == {"twynity:outlook-123": "timeout"}


def test_snapshot_pointer_is_published_after_complete_snapshot(tmp_path):
    from brown_octopus.index_store import save_snapshot_atomic

    tools = [{"capability_id": "server:tool", "name": "tool", "description": "x"}]
    save_snapshot_atomic(tmp_path / "index", tools, None, {"version": 1})

    assert load_tools(tmp_path / "index") == tools
    assert (tmp_path / "index" / "current.json").exists()
