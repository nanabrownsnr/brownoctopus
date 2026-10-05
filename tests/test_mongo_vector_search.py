from brown_octopus.index_store import MongoIndexStore


class FakeCollection:
    def __init__(self):
        self.documents = []
        self.pipeline = None

    def find_one(self, query):
        return next((doc for doc in self.documents if doc.get("_id") == query["_id"]), None)

    def replace_one(self, query, document, upsert=False):
        self.documents = [
            item for item in self.documents
            if item.get("_id") != document.get("_id")
        ]
        self.documents.append(document)

    def aggregate(self, pipeline):
        self.pipeline = pipeline
        return [{"tool": {"capability_id": "search"}, "score": 0.9}]


class FakeClient:
    def __init__(self, collection):
        self.collection = collection

    def __getitem__(self, name):
        if name == "brown_octopus":
            return self
        if name == "index_snapshots":
            return self.collection
        raise KeyError(name)

    def close(self):
        pass


def test_mongo_vector_search_builds_atlas_pipeline():
    collection = FakeCollection()
    collection.documents.append(
        {"_id": "current:default", "snapshot_id": "snapshot-1"}
    )
    store = MongoIndexStore(
        "mongodb://unused",
        vector_search_index="brown-octopus-vectors",
        client_factory=lambda _: FakeClient(collection),
    )

    result = store.search_vectors(
        [0.1, 0.2],
        limit=20,
        allowed_mcp_urls=["https://search.example/mcp"],
    )

    assert result[0]["capability_id"] == "search"
    vector_stage = collection.pipeline[0]["$vectorSearch"]
    assert vector_stage["index"] == "brown-octopus-vectors"
    assert vector_stage["queryVector"] == [0.1, 0.2]
    assert vector_stage["filter"]["mcp_url"] == {
        "$in": ["https://search.example/mcp"]
    }


def test_mongo_snapshot_store_defaults_to_local_retrieval_mode():
    store = MongoIndexStore("mongodb://unused")
    assert store.supports_native_vector_search is False
