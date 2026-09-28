from brown_octopus import model_store
from brown_octopus import setup_models
import subprocess


def test_model_directory_honors_environment_override(monkeypatch, tmp_path):
    override = tmp_path / "managed-models"
    monkeypatch.setenv("BROWN_OCTOPUS_MODEL_DIR", str(override))

    assert model_store.model_directory() == override
    assert model_store.spacy_model_path() == (
        override / "en_core_web_trf" / "en_core_web_trf-3.8.0"
    )


def test_managed_model_is_preferred_over_installed_package(monkeypatch, tmp_path):
    managed = tmp_path / "en_core_web_trf" / "en_core_web_trf-3.8.0"
    managed.mkdir(parents=True)
    monkeypatch.setenv("BROWN_OCTOPUS_MODEL_DIR", str(tmp_path))

    assert model_store.resolve_spacy_model() == str(managed)
    assert str(tmp_path) in __import__("sys").path


def test_missing_managed_model_keeps_legacy_fallback(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWN_OCTOPUS_MODEL_DIR", str(tmp_path))

    assert model_store.resolve_spacy_model() == model_store.SPACY_MODEL_NAME


def test_setup_is_idempotent_when_managed_model_is_ready(monkeypatch, tmp_path):
    managed = tmp_path / "en_core_web_trf" / "en_core_web_trf-3.8.0"
    managed.mkdir(parents=True)
    monkeypatch.setenv("BROWN_OCTOPUS_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(setup_models, "_spaCy_model_is_ready", lambda: True)

    def unexpected_run(*args, **kwargs):
        raise AssertionError("idempotent setup must not invoke pip")

    monkeypatch.setattr(setup_models.subprocess, "run", unexpected_run)
    setup_models._prepare_spaCy_model()


def test_setup_targets_managed_directory_when_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWN_OCTOPUS_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(setup_models, "_spaCy_model_is_ready", lambda: True)
    monkeypatch.setattr(
        setup_models,
        "_verify_spacy_model",
        lambda: subprocess.CompletedProcess([], 0, "verified", ""),
    )

    calls = []
    monkeypatch.setattr(
        setup_models,
        "_install_managed_spacy_assets",
        lambda: calls.append(tmp_path),
    )
    setup_models._prepare_spaCy_model()

    assert calls == [tmp_path]


def test_setup_does_not_require_pip_module(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWN_OCTOPUS_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(setup_models, "_spaCy_model_is_ready", lambda: True)
    monkeypatch.setattr(setup_models, "_install_managed_spacy_assets", lambda: None)
    monkeypatch.setattr(
        setup_models,
        "_verify_spacy_model",
        lambda: subprocess.CompletedProcess([], 0, "verified", ""),
    )

    def fail_if_subprocess_is_used(*args, **kwargs):
        raise AssertionError("managed model bootstrap must not invoke python -m pip")

    monkeypatch.setattr(setup_models.subprocess, "run", fail_if_subprocess_is_used)
    setup_models._prepare_spaCy_model()


def test_verification_failure_preserves_subprocess_diagnostics(monkeypatch, tmp_path):
    monkeypatch.setenv("BROWN_OCTOPUS_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(setup_models, "_spaCy_model_is_ready", lambda: False)
    monkeypatch.setattr(setup_models, "_install_managed_spacy_assets", lambda: None)
    monkeypatch.setattr(
        setup_models,
        "_verify_spacy_model",
        lambda: subprocess.CompletedProcess(
            ["python", "-c", "probe"],
            1,
            "",
            "Traceback (most recent call last):\nModuleNotFoundError: curated_tokenizers",
        ),
    )

    import pytest

    with pytest.raises(RuntimeError, match="ModuleNotFoundError: curated_tokenizers") as error:
        setup_models._prepare_spaCy_model()
    assert "Exit code: 1" in str(error.value)
    assert "Model directory:" in str(error.value)


def test_wheel_resolution_rejects_incompatible_first_artifact(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b"{}"

    from packaging.tags import sys_tags

    compatible_tag = str(next(iter(sys_tags())))
    payload = {
        "urls": [
            {
                "packagetype": "bdist_wheel",
                "filename": "curated_tokenizers-0.0.10-cp312-cp312-win_amd64.whl",
                "url": "https://example.invalid/incompatible.whl",
            },
            {
                "packagetype": "bdist_wheel",
                "filename": f"curated_tokenizers-0.0.10-{compatible_tag}.whl",
                "url": "https://example.invalid/compatible.whl",
            },
        ]
    }

    import json

    Response.read = lambda self: json.dumps(payload).encode()
    monkeypatch.setattr(setup_models.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    assert setup_models._pypi_wheel_url("curated-tokenizers", "0.0.10").endswith(
        "compatible.whl"
    )


def test_runtime_resolution_does_not_require_model_package_import(monkeypatch, tmp_path):
    managed = tmp_path / "en_core_web_trf"
    managed.mkdir(parents=True)
    monkeypatch.setenv("BROWN_OCTOPUS_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(model_store, "prepare_spacy_runtime", lambda: managed)

    assert model_store.resolve_spacy_model() == str(managed)
