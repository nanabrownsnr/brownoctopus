import torch

from brown_octopus.retriever import QwenCandidateRetriever, rank_tools
import brown_octopus.retriever as retriever


class FakeEmbeddingProvider:
    model_id = "test-embedding"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        if isinstance(texts, list):
            return torch.tensor([[1.0, 0.0] for _ in texts])
        return torch.tensor([1.0, 0.0])

    def healthcheck(self):
        return None


def test_candidate_retriever_accepts_an_injected_embedding_provider(monkeypatch):
    provider = FakeEmbeddingProvider()
    tools = [
        {"name": "first", "description": "first", "capability_id": "first"},
        {"name": "second", "description": "second", "capability_id": "second"},
    ]
    monkeypatch.setattr(retriever, "model", None)
    monkeypatch.setattr(retriever, "embedding_provider", None)
    monkeypatch.setattr(retriever, "TOOLS", tools)
    monkeypatch.setattr(
        retriever,
        "tool_embeddings",
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )

    ranked = QwenCandidateRetriever(embedding_provider=provider).retrieve(
        "query",
        {"retrieval_text": "capability query"},
    )

    assert [tool["name"] for tool in ranked] == ["first", "second"]
    assert provider.calls == ["capability query"]


def test_empty_legacy_index_returns_no_candidates(monkeypatch):
    monkeypatch.setattr(retriever, "model", object())
    monkeypatch.setattr(retriever, "embedding_provider", None)
    monkeypatch.setattr(retriever, "TOOLS", [])
    monkeypatch.setattr(retriever, "tool_embeddings", None)

    assert rank_tools("query") == []
