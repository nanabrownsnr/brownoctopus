import sys
import types

import pytest

from brown_octopus import device


def test_detect_device_prefers_cuda(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True)
        ),
        xpu=types.SimpleNamespace(is_available=lambda: True),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert device.detect_compute_device() == "cuda"


def test_detect_device_uses_mps_when_cuda_is_unavailable(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: True)
        ),
        xpu=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert device.detect_compute_device() == "mps"


def test_detect_device_falls_back_to_cpu(monkeypatch):
    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False),
        backends=types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        ),
        xpu=types.SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert device.detect_compute_device() == "cpu"


def test_retriever_passes_detected_device_to_sentence_transformers(monkeypatch):
    import brown_octopus.retriever as retriever

    calls = {}

    class FakeSentenceTransformer:
        def __init__(self, *args, **kwargs):
            calls.update(kwargs)

    fake_module = types.SimpleNamespace(
        SentenceTransformer=FakeSentenceTransformer
    )
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(retriever, "model", None)
    monkeypatch.setattr(retriever, "detect_compute_device", lambda: "cuda")
    monkeypatch.delenv("BROWN_OCTOPUS_ALLOW_MODEL_DOWNLOAD", raising=False)

    retriever.initialize_retriever()

    assert calls["device"] == "cuda"
    assert calls["local_files_only"] is True


@pytest.mark.parametrize("available", [True, False])
def test_analyzer_requests_spacy_gpu_preference(monkeypatch, available):
    import brown_octopus.analyzer as analyzer

    calls = []
    monkeypatch.setattr(analyzer, "nlp", None)
    monkeypatch.setattr(analyzer, "register_managed_spacy_plugins", lambda: None)
    monkeypatch.setattr(
        analyzer.spacy,
        "prefer_gpu",
        lambda: calls.append(True) or available,
    )
    monkeypatch.setattr(analyzer.spacy, "load", lambda _: "fake-nlp")

    analyzer.initialize_analyzer()

    assert calls == [True]
    assert analyzer.nlp == "fake-nlp"
