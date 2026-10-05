"""Explicitly bootstrap Brown Octopus's external model assets."""

import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from packaging.tags import sys_tags
from packaging.utils import InvalidWheelFilename, parse_wheel_filename

from brown_octopus.model_store import (
    EMBEDDING_MODEL_NAME,
    SPACY_MODEL_NAME,
    SPACY_MODEL_VERSION,
    model_directory,
    spacy_model_path,
)


SPACY_MODEL_URL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    f"{SPACY_MODEL_NAME}-{SPACY_MODEL_VERSION}/"
    f"{SPACY_MODEL_NAME}-{SPACY_MODEL_VERSION}-py3-none-any.whl"
)
SPACY_PLUGIN_PACKAGES = (
    ("spacy-curated-transformers", "0.3.1"),
    ("curated-transformers", "0.1.1"),
    ("curated-tokenizers", "0.0.10"),
)
SPACY_VERIFY_PROBE = (
    "from brown_octopus.analyzer import initialize_analyzer; "
    "initialize_analyzer(); print('spaCy model verified')"
)


def _run_python(code: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()[-1:]
        raise RuntimeError(
            "Model bootstrap subprocess failed. "
            + (detail[0] if detail else "Check the model installation and retry.")
        )


def _verify_spacy_model() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", SPACY_VERIFY_PROBE],
        check=False,
        capture_output=True,
        text=True,
    )


def _spaCy_model_is_ready() -> bool:
    return _verify_spacy_model().returncode == 0


def _format_verification_failure(result: subprocess.CompletedProcess[str]) -> str:
    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    details = [
        "Brown Octopus could not verify the managed spaCy model in a fresh Python process.",
        f"Verification command: {sys.executable} -c {SPACY_VERIFY_PROBE!r}",
        f"Exit code: {result.returncode}",
    ]
    if stdout:
        details.extend(["stdout:", stdout])
    if stderr:
        details.extend(["stderr:", stderr])
    details.extend([
        f"Model directory: {model_directory()}",
        "Run: brown-octopus doctor",
    ])
    return "\n".join(details)


def _pypi_wheel_url(project: str, version: str) -> str:
    """Resolve a pinned wheel without invoking pip, uv, or another installer."""
    api_url = f"https://pypi.org/pypi/{project}/{version}/json"
    with urllib.request.urlopen(api_url, timeout=60) as response:
        payload = json.load(response)
    candidates = [
        file
        for file in payload.get("urls", [])
        if file.get("packagetype") == "bdist_wheel"
    ]
    supported_tags = set(sys_tags())
    compatible = []
    rejected = []
    for file in candidates:
        filename = file.get("filename", "")
        try:
            _, _, _, tags = parse_wheel_filename(filename)
        except InvalidWheelFilename:
            rejected.append(filename)
            continue
        if tags & supported_tags:
            compatible.append(file)
        else:
            rejected.append(filename)

    # Prefer a universal compatible wheel, then preserve the repository's
    # deterministic PyPI ordering for compatible platform wheels.
    compatible.sort(
        key=lambda file: not file.get("filename", "").endswith(
            ("-py3-none-any.whl", "-py2.py3-none-any.whl")
        )
    )
    selected = compatible[0] if compatible else None
    if selected is None or not selected.get("url"):
        platform = f"{sys.implementation.name} {sys.version.split()[0]}"
        raise RuntimeError(
            f"No compatible wheel found for {project}=={version} on {platform}. "
            f"Available wheel files: {', '.join(rejected) or 'none'}."
        )
    print(
        f"Selected wheel for {project}=={version}: {selected['filename']}\n"
        f"URL: {selected['url']}"
    )
    return selected["url"]


def _extract_wheel(url: str, destination) -> None:
    """Download and safely extract one wheel into managed model storage."""
    destination.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=".brown-octopus-model-", dir=destination.parent)
    )
    wheel_path = staging / "asset.whl"
    try:
        with urllib.request.urlopen(url, timeout=300) as response, wheel_path.open("wb") as output:
            shutil.copyfileobj(response, output)

        with zipfile.ZipFile(wheel_path) as archive:
            root = staging.resolve()
            for member in archive.infolist():
                member_path = (staging / member.filename).resolve()
                if root not in member_path.parents and member_path != root:
                    raise RuntimeError(f"Unsafe model wheel path: {member.filename}")
            archive.extractall(staging)

        for child in staging.iterdir():
            if child.name == "asset.whl":
                continue
            target = destination / child.name
            if child.is_dir():
                shutil.copytree(child, target, dirs_exist_ok=True)
            else:
                shutil.copy2(child, target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _install_managed_spacy_assets() -> None:
    destination = model_directory()
    print(f"Installing spaCy model: {SPACY_MODEL_NAME} into {destination}")
    print(f"URL: {SPACY_MODEL_URL}")
    _extract_wheel(SPACY_MODEL_URL, destination)
    for project, version in SPACY_PLUGIN_PACKAGES:
        print(f"Preparing spaCy runtime plugin: {project}=={version}")
        _extract_wheel(_pypi_wheel_url(project, version), destination)


def _prepare_spaCy_model() -> None:
    if spacy_model_path().is_dir() and _spaCy_model_is_ready():
        print(f"spaCy model already prepared: {SPACY_MODEL_NAME}")
        return

    _install_managed_spacy_assets()

    # Verify in a new process so imports/entry points from the installation
    # cannot be masked by the current interpreter's import state.
    verification = _verify_spacy_model()
    if verification.returncode:
        raise RuntimeError(_format_verification_failure(verification))


def _prepare_embedding_model() -> None:
    probe = (
        "from sentence_transformers import SentenceTransformer; "
        f"model = SentenceTransformer({EMBEDDING_MODEL_NAME!r}, "
        "trust_remote_code=True, local_files_only=True); "
        "print('embedding model already prepared:', model.device)"
    )
    if subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0:
        return

    print(f"Downloading embedding model: {EMBEDDING_MODEL_NAME}")
    download = (
        "from sentence_transformers import SentenceTransformer; "
        f"model = SentenceTransformer({EMBEDDING_MODEL_NAME!r}, "
        "trust_remote_code=True); "
        "print('embedding model prepared:', model.device)"
    )
    _run_python(download)


def main(*, spacy: bool = False, embedding: bool = False) -> None:
    """Download and validate selected models without a package manager.

    With no selector, preserve the original behavior and prepare both runtime
    models. The selectors are intentionally mutually exclusive at the CLI.
    """
    prepare_spacy = spacy or not (spacy or embedding)
    prepare_embedding = embedding or not (spacy or embedding)
    if prepare_spacy:
        _prepare_spaCy_model()
    if prepare_embedding:
        _prepare_embedding_model()
    print("Brown Octopus model bootstrap complete.")


if __name__ == "__main__":
    main()
