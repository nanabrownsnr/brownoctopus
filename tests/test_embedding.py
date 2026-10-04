import sys
import types

from brown_octopus.embedding import LocalEmbeddingProvider


def test_local_provider_preserves_default_qwen_model_and_encoding_contract(monkeypatch):
    calls = {}

    class FakeModel:
        def encode(self, texts, **kwargs):
            calls["texts"] = texts
            calls["kwargs"] = kwargs
            return "embeddings"

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            calls["model_name"] = model_name
            calls["init_kwargs"] = kwargs
            self.model = FakeModel()

        def encode(self, texts, **kwargs):
            return self.model.encode(texts, **kwargs)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    provider = LocalEmbeddingProvider(device="cpu")

    assert provider.model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert provider.embed(["tool description"]) == "embeddings"
    assert calls["model_name"] == "Qwen/Qwen3-Embedding-0.6B"
    assert calls["init_kwargs"]["local_files_only"] is True
    assert calls["init_kwargs"]["device"] == "cpu"
    assert calls["kwargs"] == {
        "convert_to_tensor": True,
        "normalize_embeddings": True,
    }


def test_local_provider_reuses_loaded_model(monkeypatch):
    loads = []

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            loads.append(1)

        def encode(self, texts, **kwargs):
            return texts

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    provider = LocalEmbeddingProvider(device="cpu")
    provider.healthcheck()
    provider.healthcheck()

    assert len(loads) == 1


def test_local_provider_accepts_a_non_qwen_model_name(monkeypatch):
    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            calls["model_name"] = model_name
            calls["kwargs"] = kwargs

        def encode(self, texts, **kwargs):
            return texts

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    provider = LocalEmbeddingProvider("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    assert provider.embed("query") == "query"
    assert provider.model_id == "sentence-transformers/all-MiniLM-L6-v2"
    assert calls["model_name"] == "sentence-transformers/all-MiniLM-L6-v2"


def test_local_provider_accepts_a_model_path(monkeypatch, tmp_path):
    captured = {}

    class FakeSentenceTransformer:
        def __init__(self, model_name, **kwargs):
            captured["model_name"] = model_name

        def encode(self, texts, **kwargs):
            return texts

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )

    model_path = tmp_path / "embedding-model"
    provider = LocalEmbeddingProvider(model_path=model_path, device="cpu")
    provider.healthcheck()
    assert captured["model_name"] == str(model_path)
