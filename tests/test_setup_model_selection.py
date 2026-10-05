from brown_octopus import setup_models


def test_setup_models_without_selector_prepares_both(monkeypatch):
    calls = []
    monkeypatch.setattr(setup_models, "_prepare_spaCy_model", lambda: calls.append("spacy"))
    monkeypatch.setattr(setup_models, "_prepare_embedding_model", lambda: calls.append("embedding"))

    setup_models.main()

    assert calls == ["spacy", "embedding"]


def test_setup_models_spacy_selector(monkeypatch):
    calls = []
    monkeypatch.setattr(setup_models, "_prepare_spaCy_model", lambda: calls.append("spacy"))
    monkeypatch.setattr(setup_models, "_prepare_embedding_model", lambda: calls.append("embedding"))

    setup_models.main(spacy=True)

    assert calls == ["spacy"]


def test_setup_models_embedding_selector(monkeypatch):
    calls = []
    monkeypatch.setattr(setup_models, "_prepare_spaCy_model", lambda: calls.append("spacy"))
    monkeypatch.setattr(setup_models, "_prepare_embedding_model", lambda: calls.append("embedding"))

    setup_models.main(embedding=True)

    assert calls == ["embedding"]
