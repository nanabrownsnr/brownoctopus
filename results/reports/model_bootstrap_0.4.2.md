# Brown Octopus 0.4.2 model-bootstrap release blocker

## Root cause

0.4.1 invoked `sys.executable -m pip` to install the managed spaCy model and
curated-transformer packages. uv-managed environments may intentionally omit
the `pip` module, so bootstrap failed before downloading any assets.

## Fix

0.4.2 no longer invokes pip, uv, or another package manager for spaCy model
bootstrap. It uses the Python standard library to:

1. Resolve the pinned spaCy model wheel URL.
2. Resolve pinned pure-Python plugin wheels through the PyPI JSON API.
3. Download wheels with `urllib.request`.
4. Validate wheel paths and extract them into Brown Octopus-managed model
   storage with `zipfile`.
5. Verify the model in a fresh Python subprocess.

The Qwen model continues to use its existing Hugging Face cache behavior. No
runtime downloads were added to `initialize()`.

## No-pip regression

The installed test environment confirmed:

```text
python -m pip --version
    No module named pip
    exit code 1
```

With the managed assets present, the same environment successfully ran:

```text
brown-octopus setup-models
    spaCy model already prepared: en_core_web_trf
    Brown Octopus model bootstrap complete.

brown-octopus doctor
    [OK] en_core_web_trf
    [OK] Qwen/Qwen3-Embedding-0.6B
    [OK] Brown Octopus is ready.

initialize_analyzer()
    Analyzer OK
```

The environment's legacy `en-core-web-trf`, `spacy-curated-transformers`,
`curated-transformers`, and `curated-tokenizers` distributions were removed
before this validation. The runtime used only the managed model directory.

The focused regression suite also verifies that bootstrap does not invoke
`python -m pip`.

## Tests

```text
Focused model-bootstrap tests: 7 passed
Installed-wheel core suite: 155 passed, 5 external tests deselected
```

The five external tests require the live MCP endpoint and remain separately
marked. No retrieval tests were skipped.

## Build artifacts

- `dist/brown_octopus-0.4.2-py3-none-any.whl`
- `dist/brown_octopus-0.4.2.tar.gz`

## Retrieval confirmation

This change is limited to model bootstrap and managed model loading. Qwen
retrieval behavior, `retrieval_text`, intent decomposition, Min-4, Bounded Max
Gap, TTL 8, active cap 30, sessions, capability sources, index behavior, and
result contracts were not changed.
