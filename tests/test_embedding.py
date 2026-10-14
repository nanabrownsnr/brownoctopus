import sys
import types

from brown_octopus.embedding import LocalEmbeddingProvider


def test_local_provider_preserves_default_qwen_model_and_encoding_contract(monkeypatch):
    # CI may enable explicit model downloads for integration tests. This unit
    # test verifies the provider's normal offline-by-default behavior.
    monkeypatch.delenv("BROWN_OCTOPUS_ALLOW_MODEL_DOWNLOAD", raising=False)
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
        "batch_size": 8,
        "show_progress_bar": False,
        "convert_to_tensor": True,
        "normalize_embeddings": True,
    }


def test_local_provider_reads_embedding_resource_controls_from_environment(
    monkeypatch,
):
    monkeypatch.setenv("OCTOPUS_EMBEDDING_BATCH_SIZE", "3")
    monkeypatch.setenv("OCTOPUS_EMBEDDING_MAX_SEQ_LENGTH", "256")
    monkeypatch.setenv("OCTOPUS_EMBEDDING_DEVICE", "cuda:1")

    provider = LocalEmbeddingProvider()

    assert provider.batch_size == 3
    assert provider.max_seq_length == 256
    assert provider.device is None


def test_local_provider_retries_on_cuda_oom_on_cpu(monkeypatch):
    calls = []

    class FakeModel:
        max_seq_length = None

        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs.copy()))
            if len(calls) == 1:
                raise RuntimeError("CUDA out of memory")
            return "cpu-embeddings"

        def to(self, device):
            calls.append(("to", device))

    provider = LocalEmbeddingProvider(device="cuda", batch_size=4)
    provider.model = FakeModel()

    # The fallback only needs to identify the error and move the model. The
    # real torch CUDA cache is not involved in this unit test.
    monkeypatch.setattr(provider, "_move_to_cpu_after_oom", lambda: setattr(provider, "device", "cpu"))

    assert provider.embed(["long tool description"]) == "cpu-embeddings"
    assert calls[1][1]["batch_size"] == 1
    assert provider.device == "cpu"


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
