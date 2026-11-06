import pytest
from brown_octopus import Octopus
from brown_octopus.config import OctopusConfig
from brown_octopus.pipeline import CapabilityPipeline
from brown_octopus.context_manager import ActiveCapabilityContext
from brown_octopus.index_store import _deserialize_embeddings


def test_config_controls_real_selection_and_rebuild(monkeypatch):
    monkeypatch.setenv("OCTOPUS_MAX_TOOLS", "32")
    candidates = [{"name": str(i), "score": 1.0} for i in range(40)]
    for config in (OctopusConfig(max_tools=32), OctopusConfig.from_env()):
        runtime = Octopus(config=config)
        for _ in range(2):
            runtime._build_pipeline()
            assert len(runtime.selection_strategy.select("", {}, candidates)) == 32
            adaptive = [dict(tool, score=1.0 if i < 8 else 0.1) for i, tool in enumerate(candidates)]
            assert len(runtime.selection_strategy.select("", {}, adaptive)) == 8


@pytest.mark.parametrize("kwargs", [{"max_tools": 0}, {"min_tools": 17}, {"max_returned_tools": 0}, {"min_gap_percent": float("nan")}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Octopus(config=OctopusConfig(**kwargs))


def test_null_embedding_payload_is_valid_empty_value():
    assert _deserialize_embeddings(None) is None
    with pytest.raises(Exception):
        _deserialize_embeddings(b"corrupt")


def test_response_limit_environment(monkeypatch):
    monkeypatch.setenv("OCTOPUS_MAX_RETURNED_TOOLS", "7")
    runtime = Octopus()
    runtime._build_pipeline()
    assert runtime.pipeline.max_returned_tools == 7


def test_response_cap_preserves_session_and_covers_intents():
    tools = [{"name": str(i)} for i in range(6)]
    class Analyzer:
        def analyze(self, query):
            return [{"text": "a"}, {"text": "b"}]
    class Retriever:
        def retrieve(self, query, intent):
            return tools[:3] if intent["text"] == "a" else tools[3:]
    class Selector:
        def select(self, query, intent, candidates):
            return candidates
    context = ActiveCapabilityContext(lambda ids: [tool for tool in tools if tool["name"] in ids], active_cap=10)
    pipeline = CapabilityPipeline(Analyzer(), Retriever(), Selector(), context, max_returned_tools=2)
    result = pipeline.retrieve("query", turn=1)
    assert [tool["name"] for tool in result["tools"]] == ["0", "3"]
    assert len(context.active_state) == 6
    assert result["retrieval_diagnostics"]["response_truncated"] is True


@pytest.mark.anyio
async def test_empty_postgres_first_add_readd_and_reopen(monkeypatch):
    import torch
    from brown_octopus import OctopusIndex, CapabilityDiscoveryResult
    from brown_octopus.index_store import PostgresIndexStore, _serialize_embeddings
    class Store(PostgresIndexStore):
        snapshot = None
        def exists(self):
            return self.snapshot is not None
        def _load(self, field):
            return self.snapshot[field]
        def save_snapshot_atomic(self, tools, embeddings, metadata):
            self.snapshot = dict(tools=tools, embeddings=None if embeddings is None else _serialize_embeddings(embeddings), metadata=metadata)
    class Provider:
        model_id = "test"
        dimension = 2
    class Source:
        source_id = "test"
        async def discover(self):
            return CapabilityDiscoveryResult(tools=[{"name": "one", "capability_id": "one"}])
    encoded = []
    def embed(tools, **kwargs):
        encoded.append(len(tools))
        return torch.ones((len(tools), 2))
    monkeypatch.setattr("brown_octopus.octopus.initialize_analyzer", lambda: None)
    monkeypatch.setattr("brown_octopus.octopus.initialize_retriever", lambda *a: None)
    monkeypatch.setattr("brown_octopus.octopus.build_embeddings", embed)
    store = Store("unused")
    index = OctopusIndex(index_store=store, embedding_provider=Provider(), config=OctopusConfig(max_tools=32))
    await index.create()
    assert store.load_embeddings() is None
    await index.add(Source())
    await index.add(Source())
    assert encoded == [1]
    assert index.octopus.selection_strategy.max_tools == 32
    reopened = OctopusIndex(index_store=store, embedding_provider=Provider())
    await reopened.runtime().initialize()
    assert len(store.load_tools()) == 1
    assert store.load_embeddings().shape == (1, 2)


def test_database_null_contract_and_corruption(monkeypatch):
    from brown_octopus.index_store import PostgresIndexStore, MongoIndexStore
    pg = PostgresIndexStore("unused")
    monkeypatch.setattr(pg, "_load", lambda field: None if field == "embeddings" else [])
    assert pg.load_embeddings() is None
    monkeypatch.setattr(pg, "load_tools", lambda: [{"name": "one"}])
    with pytest.raises(ValueError, match="no embedding matrix"):
        pg.load_embeddings()
    mongo = MongoIndexStore("unused")
    monkeypatch.setattr(mongo, "_load", lambda: {"tools": [], "embeddings": None})
    assert mongo.load_embeddings() is None
