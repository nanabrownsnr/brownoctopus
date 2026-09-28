# Brown Octopus 0.4.0 production-readiness report

Date: 2026-09-27

## Package

- Distribution: `brown-octopus`
- Python package: `brown_octopus`
- Version: `0.4.0`
- Wheel: `dist/brown_octopus-0.4.0-py3-none-any.whl`
- Source distribution: `dist/brown_octopus-0.4.0.tar.gz`
- Tested Python: 3.13.15 on Windows
- CI matrix configured: Python 3.13 on Windows and Linux

The wheel metadata was inspected and reports `Name: brown-octopus` and
`Version: 0.4.0`. The wheel contains package code only; no test data, result
files, `.env` files, indexes, or local caches were found.

## CLI evidence

The installed wheel was used for these checks.

`brown-octopus --help`:

```text
usage: brown-octopus COMMAND

Brown Octopus - Capability context management for AI agents.

positional arguments:
  {setup-models,doctor,inspect,version}
    setup-models        Prepare the models required by Octopus.
    doctor              Check whether Brown Octopus is ready.
    inspect             Inspect the local index and runtime configuration.
    version             Display version information.

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
```

`brown-octopus --version` and `brown-octopus version` both return:

```text
brown-octopus 0.4.0
```

`doctor` and `inspect --json` both completed successfully in the prepared
environment. Doctor reported the local spaCy model, Qwen cache, and 122-tool
index as available. Help/version/inspect do not initialize the ML models.

## Tests

Installed-wheel core suite:

```text
148 passed, 5 deselected, 2 warnings
```

Command:

```powershell
.clean-env\Scripts\pytest.exe -q --basetemp .pytest-tmp-wheel -m "not external"
```

The five deselected tests are explicitly marked `external`; they require the
live development MCP endpoint and are acceptance tests rather than offline
package tests. When attempted in this environment, the endpoint failed with
`Client failed to connect: All connection attempts failed`.

The original development environment still has filesystem permission failures
inside installed ML packages. A clean environment using the built wheel passed
the core suite, so the prior `_multiarray_umath` access-denied failure is
diagnosed as corruption/ACL damage in the old environment, not a demonstrated
Brown Octopus test failure. No tests were skipped to hide a NumPy failure.

The local source environment also has an unrelated permission failure while
loading `transformers` files, confirming that it should not be used as the
release validation environment.

## Clean wheel validation

The actual wheel was installed into `.clean-env`, not imported from the source
checkout:

```powershell
uv venv .clean-env --python 3.13
uv pip install --python .clean-env\Scripts\python.exe dist\brown_octopus-0.4.0-py3-none-any.whl
.clean-env\Scripts\brown-octopus.exe --help
.clean-env\Scripts\brown-octopus.exe --version
.clean-env\Scripts\brown-octopus.exe version
.clean-env\Scripts\brown-octopus.exe inspect --json
.clean-env\Scripts\brown-octopus.exe doctor
.clean-env\Scripts\python.exe -c "from brown_octopus import Octopus; print(Octopus)"
```

All import, CLI, wheel, and prepared-model checks passed. `setup-models` is
explicit and idempotent; normal `initialize()` does not download models.

The installed-wheel runtime smoke also completed:

```text
initialize(): 122 capabilities
retrieve_result("search the web for latest Nvidia news"): 4 retrieved, 4 active
```

## Production contracts

- `CapabilitySource` remains injectable and host-controlled through
  `await octopus.update()`.
- `SessionStore` remains injectable; `InMemorySessionStore` is the default.
- Session state is isolated per session and custom stores must make `mutate()`
  atomic across workers.
- Capability index publication remains snapshot-based and atomic.
- Failed/non-authoritative capability discovery preserves the last valid source
  snapshot.
- Initialization errors are Brown Octopus-specific and include the setup
  command for missing models.
- The package remains credential-agnostic and does not store conversation
  history.
- Brown Octopus manages capability context; the host owns the LLM, conversation
  context, execution, credentials, authorization, identity, and refresh policy.

## CI and documentation

Added/updated:

- `.github/workflows/ci.yml` for Windows/Linux Python 3.13 tests and package
  build validation.
- CLI commands: `setup-models`, `doctor`, `inspect`, and `version`.
- README installation, quickstart, architecture boundaries, sessions, source
  updates, diagnostics, and recall-oriented v1 guidance.
- `docs/production.md` release and deployment guidance.
- CLI regression tests and installed-wheel validation coverage.

`twine` was not installed in the current local environment, so `twine check`
was not run locally. The CI package job is configured to run it.

## Retrieval regression confirmation

No retrieval redesign or tuning was made. The following remain unchanged:

- Qwen/Qwen3-Embedding-0.6B
- capability `name + description` representation
- deterministic spaCy intent analysis and `retrieval_text`
- Min-4 selection
- Bounded Max Gap
- TTL 8
- active capability cap 30
- current-turn-first active context ordering
- Jev non-default status

## Known limitations

- Live MCP acceptance tests require the configured external endpoint and are not
  offline unit tests.
- The current source checkout's original Windows environment has damaged/denied
  ML-package files; release validation must use a clean environment.
- `brown-octopus setup-models` requires network access when assets are absent.
- Distributed session atomicity depends on the supplied external `SessionStore`.
- Capability discovery network behavior belongs to the explicitly invoked
  `CapabilitySource.update()` path.
- Recall-oriented V3 selection may expose extra plausible capabilities; it is
  not a mathematical 100% recall guarantee.
- Conversation context and LLM context-window management remain host concerns.
