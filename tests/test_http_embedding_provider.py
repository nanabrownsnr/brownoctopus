import json

import torch

from brown_octopus.embedding import HttpEmbeddingProvider


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_http_embedding_provider_posts_openai_compatible_request(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.data.decode())
        captured["timeout"] = timeout
        return FakeResponse({
            "data": [
                {"index": 0, "embedding": [3.0, 4.0]},
                {"index": 1, "embedding": [4.0, 3.0]},
            ]
        })

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = HttpEmbeddingProvider(
        "https://embed.example/v1/embeddings",
        "qwen-embed",
        api_key="secret-token",
        timeout=12.0,
    )

    result = provider.embed(["first", "second"])

    assert captured["url"] == "https://embed.example/v1/embeddings"
    assert captured["body"] == {
        "model": "qwen-embed",
        "input": ["first", "second"],
    }
    assert captured["headers"]["Authorization"] == "Bearer secret-token"
    assert captured["timeout"] == 12.0
    assert torch.allclose(result[0], torch.tensor([0.6, 0.8]))
    assert torch.allclose(result[1], torch.tensor([0.8, 0.6]))


def test_http_embedding_provider_supports_single_input(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeResponse({
            "data": [{"embedding": [0.0, 2.0]}],
        }),
    )

    result = HttpEmbeddingProvider(
        "https://embed.example/v1/embeddings",
        "qwen-embed",
    ).embed("query")

    assert result.shape == (2,)
    assert torch.allclose(result, torch.tensor([0.0, 1.0]))


def test_http_embedding_provider_accepts_ollama_embeddings_response(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeResponse({
            "embeddings": [[3.0, 4.0]],
        }),
    )

    result = HttpEmbeddingProvider(
        "http://localhost:11434/api/embed",
        "qwen3-embedding:0.6b",
    ).embed("query")

    assert torch.allclose(result, torch.tensor([0.6, 0.8]))


def test_http_embedding_provider_rejects_incomplete_response(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: FakeResponse({"data": []}),
    )

    provider = HttpEmbeddingProvider(
        "https://embed.example/v1/embeddings",
        "qwen-embed",
    )

    try:
        provider.embed(["first"])
    except RuntimeError as exc:
        assert "one data item per input" in str(exc)
    else:
        raise AssertionError("invalid API responses must fail clearly")
