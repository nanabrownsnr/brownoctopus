"""Disposable local backend checks for incremental index additions.

Run explicitly with ``-m external`` while the local test services are running.
These tests are not part of the normal offline suite.
"""

import uuid

import pytest
import torch

from brown_octopus import CapabilityDiscoveryResult, Octopus
from brown_octopus.index_store import MongoIndexStore, PostgresIndexStore


class Provider:
    model_id = "incremental-test-provider"
    dimension = 2


class Source:
    async def discover(self):
        return CapabilityDiscoveryResult(
            tools=[
                {
                    "capability_id": "new-capability",
                    "name": "new_tool",
                    "description": "new capability",
                }
            ]
        )


def _patch_runtime(monkeypatch):
    encoded = []

    def build_embeddings(tools, embedding_provider=None):
        encoded.append([tool["capability_id"] for tool in tools])
        return torch.tensor([[0.0, 1.0] for _ in tools])

    monkeypatch.setattr("brown_octopus.octopus.initialize_analyzer", lambda: None)
    monkeypatch.setattr(
        "brown_octopus.octopus.initialize_retriever",
        lambda *args: None,
    )
    monkeypatch.setattr(
        "brown_octopus.octopus.build_embeddings",
        build_embeddings,
    )
    monkeypatch.setattr("brown_octopus.octopus.set_tools", lambda tools: None)
    monkeypatch.setattr("brown_octopus.octopus.load_index", lambda **kwargs: None)
    return encoded


def _seed(store):
    store.save_snapshot_atomic(
        [
            {
                "capability_id": "existing-capability",
                "name": "existing_tool",
                "description": "existing capability",
            }
        ],
        torch.tensor([[1.0, 0.0]]),
        {
            "version": 1,
            "index_format_version": 1,
            "embedding_provider": {
                "type": "Provider",
                "model_id": Provider.model_id,
                "dimension": Provider.dimension,
            },
        },
    )


@pytest.mark.external
@pytest.mark.anyio
async def test_incremental_add_postgres_preserves_existing_capabilities(
    monkeypatch,
):
    store = PostgresIndexStore(
        "postgresql://postgres:brown-octopus-test@127.0.0.1:55432/brown_octopus",
        index_name=f"incremental-{uuid.uuid4().hex}",
    )
    _seed(store)
    encoded = _patch_runtime(monkeypatch)
    octopus = Octopus(index_store=store, embedding_provider=Provider())
    octopus._build_pipeline = lambda: None

    try:
        report = await octopus.add_sources([Source()])
        tools = store.load_tools()
        assert encoded == [["new-capability"]]
        assert report.added == ["new-capability"]
        assert {tool["capability_id"] for tool in tools} == {
            "existing-capability",
            "new-capability",
        }
    finally:
        store.reset()


@pytest.mark.external
@pytest.mark.anyio
async def test_incremental_add_mongo_preserves_existing_capabilities(
    monkeypatch,
):
    pytest.importorskip("pymongo")
    store = MongoIndexStore(
        "mongodb://127.0.0.1:32768",
        index_name=f"incremental-{uuid.uuid4().hex}",
    )
    _seed(store)
    encoded = _patch_runtime(monkeypatch)
    octopus = Octopus(index_store=store, embedding_provider=Provider())
    octopus._build_pipeline = lambda: None

    try:
        report = await octopus.add_sources([Source()])
        tools = store.load_tools()
        assert encoded == [["new-capability"]]
        assert report.added == ["new-capability"]
        assert {tool["capability_id"] for tool in tools} == {
            "existing-capability",
            "new-capability",
        }
    finally:
        store.reset()
