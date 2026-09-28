# Brown Octopus 0.4.1 model-bootstrap release blocker

## Root cause

`setup-models` previously installed `en-core-web-trf` and its curated-transformer
runtime packages directly into the consumer virtual environment. They were not
declared dependencies of the consumer project, so `uv sync` correctly removed
them while reconciling the environment.

## Resolution

Brown Octopus now stores spaCy model assets outside the consumer dependency
lifecycle:

- Windows: `%LOCALAPPDATA%\Brown Octopus\models`
- macOS: `~/Library/Application Support/Brown Octopus/models`
- Linux: `$XDG_DATA_HOME/brown-octopus/models`, or `~/.local/share/brown-octopus/models`
- Override: `BROWN_OCTOPUS_MODEL_DIR`

The managed model is resolved from:

```text
<model-directory>/en_core_web_trf/en_core_web_trf-3.8.0/
```

The managed directory also contains the curated-transformer runtime packages
needed by the model. Runtime initialization adds that directory to the import
path and registers the bundled spaCy plugin/architectures before loading the
model.

Resolution order:

1. `BROWN_OCTOPUS_MODEL_DIR`, when set.
2. The platform-specific Brown Octopus user model directory.
3. An already-installed `en_core_web_trf` package as a pre-release fallback.

New `setup-models` runs always prepare the managed location. Normal
initialization never downloads models.

## CLI validation

With the managed model already present, the second/idempotent setup run
reported:

```text
spaCy model already prepared: en_core_web_trf
Brown Octopus model bootstrap complete.
```

Doctor reported:

```text
Models
  [OK] en_core_web_trf
  [OK] Qwen/Qwen3-Embedding-0.6B

Status
  [OK] Brown Octopus is ready.
```

## Persistence regression

The consumer regression was simulated in a clean installed-wheel environment:

1. Brown Octopus 0.4.1 was installed from the built wheel.
2. The managed model directory was populated.
3. The environment's `en-core-web-trf`, `spacy-curated-transformers`,
   `curated-transformers`, and `curated-tokenizers` distributions were removed,
   simulating dependency synchronization cleanup.
4. `brown-octopus doctor` still reported ready.
5. `initialize_analyzer()` printed `Analyzer OK`.

This proves runtime no longer depends on those packages remaining registered in
the consumer environment. The actual external `octopus-agent` checkout was not
available in this workspace, so that exact project was not modified or run.

## Tests

Focused model/CLI tests:

```text
9 passed
```

Installed-wheel core suite under the managed-model/no-installed-spaCy-package
simulation:

```text
154 passed, 5 external tests deselected
```

The five external tests require the live MCP endpoint and remain separately
marked; no retrieval or model-bootstrap tests were skipped.

## Build

- `dist/brown_octopus-0.4.1-py3-none-any.whl`
- `dist/brown_octopus-0.4.1.tar.gz`

## Retrieval regression confirmation

No retrieval behavior or architecture was changed. Qwen retrieval,
`retrieval_text`, Min-4, Bounded Max Gap, TTL 8, active cap 30, sessions,
capability sources, session stores, index updates, and result contracts remain
unchanged.
