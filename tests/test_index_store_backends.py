import json

import torch

from brown_octopus.index_store import (
    IndexStore,
    LocalIndexStore,
    MongoIndexStore,
    PostgresIndexStore,
)


class MemoryIndexStore:
    def __init__(self):
        self.snapshot = None

    def exists(self):
        return self.snapshot is not None

    def load_tools(self):
        return self.snapshot["tools"]

    def load_embeddings(self):
        return self.snapshot["embeddings"]

    def load_metadata(self):
        return self.snapshot["metadata"]

    def save_snapshot_atomic(self, tools, embeddings, metadata):
        self.snapshot = {
            "tools": tools,
            "embeddings": embeddings,
            "metadata": metadata,
        }

    def reset(self):
        self.snapshot = None


def test_custom_index_store_matches_public_contract():
    store = MemoryIndexStore()
    assert isinstance(store, IndexStore)
    assert store.exists() is False

    tools = [{"capability_id": "source:tool", "name": "tool"}]
    embeddings = torch.tensor([[1.0, 0.0]])
    metadata = {"version": 1, "tool_count": 1}
    store.save_snapshot_atomic(tools, embeddings, metadata)

    assert store.exists() is True
    assert store.load_tools() == tools
    assert torch.equal(store.load_embeddings(), embeddings)
    assert store.load_metadata() == metadata

    store.reset()
    assert store.exists() is False


def test_local_index_store_remains_the_default_file_backend(tmp_path):
    store = LocalIndexStore(tmp_path / "index")
    assert store.exists() is False

    store.save_snapshot_atomic(
        [{"capability_id": "source:tool", "name": "tool"}],
        torch.tensor([[1.0, 0.0]]),
        {"version": 1},
    )

    assert store.exists() is True
    assert store.load_tools()[0]["name"] == "tool"
    assert torch.equal(store.load_embeddings(), torch.tensor([[1.0, 0.0]]))
    assert store.load_metadata() == {"version": 1}


def test_database_stores_do_not_import_optional_clients_at_construction():
    postgres = PostgresIndexStore("postgresql://localhost/brown_octopus")
    mongo = MongoIndexStore("mongodb://localhost:27017")

    assert postgres.index_name == "default"
    assert mongo.database_name == "brown_octopus"
    assert mongo.collection_name == "index_snapshots"


def test_serialized_snapshot_payload_is_json_safe():
    tools = [{"capability_id": "source:tool", "name": "tool"}]
    assert json.loads(json.dumps(tools)) == tools
