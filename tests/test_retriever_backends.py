from brown_octopus.retriever import (
    LocalCandidateRetriever,
    QwenCandidateRetriever,
    VectorSearchCandidateRetriever,
)


class FakeProvider:
    model_id = "test-embedding"

    def embed(self, texts):
        return [0.1, 0.2, 0.3]


class FakeVectorIndex:
    def __init__(self):
        self.calls = []

    def search_vectors(self, query_embedding, *, limit=None, allowed_mcp_urls=None):
        self.calls.append((query_embedding, limit, allowed_mcp_urls))
        return [{"name": "search", "rank": 1, "score": 0.91}]


def test_default_qwen_retriever_remains_local_implementation():
    assert issubclass(QwenCandidateRetriever, LocalCandidateRetriever)
    assert QwenCandidateRetriever.name == "qwen_dense"
    assert QwenCandidateRetriever is LocalCandidateRetriever


def test_vector_search_retriever_preserves_candidate_contract():
    index = FakeVectorIndex()
    retriever = VectorSearchCandidateRetriever(
        index,
        FakeProvider(),
        candidate_k=20,
    )

    result = retriever.retrieve(
        "original query",
        {"retrieval_text": "search web"},
        allowed_mcp_urls=["https://search.example/mcp"],
    )

    assert result == [{"name": "search", "rank": 1, "score": 0.91}]
    assert index.calls == [
        ([0.1, 0.2, 0.3], 20, ["https://search.example/mcp"])
    ]
