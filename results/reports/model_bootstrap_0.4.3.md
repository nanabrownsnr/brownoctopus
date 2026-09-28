# Brown Octopus 0.4.3 bootstrap release-candidate report

## Scope

This release candidate changes only managed spaCy bootstrap diagnostics and
wheel compatibility selection. Retrieval, V3 selection, sessions, sources,
indexes, and result contracts were not changed.

## Diagnosis

The 0.4.2 installer selected the first wheel returned by PyPI whenever a
universal wheel was not present. `curated-tokenizers` contains native wheels,
so that policy could select an artifact for the wrong Python ABI, operating
system, or architecture. The subsequent fresh-process import failure was
collapsed to a generic message.

The exact failing traceback from the consumer was not recoverable in this
checkout: the live empty-directory reproduction stalled before the first
download and produced no subprocess output. The 0.4.3 verifier now preserves
the command, exit code, stdout, stderr, and traceback in the setup error.

## Fix

`packaging.tags.sys_tags()` and `packaging.utils.parse_wheel_filename()` now
select only wheels compatible with the running interpreter and platform.
Universal wheels are preferred among compatible candidates. If no compatible
wheel exists, setup fails with the runtime and available filenames instead of
extracting an incompatible native artifact.

The known compatible Python 3.13 Windows x86-64 tokenizer artifact is:

```text
curated_tokenizers-0.0.10-cp313-cp313-win_amd64.whl
https://files.pythonhosted.org/packages/fd/e7/ff7fee7511f4264f7966c9717183a4c3e27ddf08648ef7c3c4a1f8161ec1/curated_tokenizers-0.0.10-cp313-cp313-win_amd64.whl
```

The pure-Python plugin artifacts are:

```text
spacy_curated_transformers-0.3.1-py2.py3-none-any.whl
curated_transformers-0.1.1-py2.py3-none-any.whl
```

The spaCy model remains the pinned `en_core_web_trf-3.8.0-py3-none-any.whl`.

After consumer validation, the doctor spaCy probe timeout was increased from
30 seconds to 180 seconds. Cold Windows ML startup can exceed 30 seconds even
when the managed model is valid; the direct analyzer success in the consumer
confirmed that this was a false-negative readiness check.

## Validation

- Focused source tests: **9 passed**.
- Focused installed-wheel tests in the no-pip clean environment: **9 passed**.
- Wheel import/version check: **passed**, installed package reports 0.4.3.
- Wheel and sdist build: **passed**.
- The broader source suite was blocked during collection by an existing
  Windows ACL error loading `transformers/models/afmoe/configuration_afmoe.py`;
  it produced 2 collection errors before Brown Octopus tests ran.

The requested empty-directory/no-pip live bootstrap was started with the
0.4.3 wheel but stalled before creating the first managed asset. It was
stopped after several minutes. Therefore the live
`setup-models -> doctor -> analyzer -> uv sync -> doctor` gate remains
unverified in this environment and must be rerun with working PyPI access.

## Artifacts

```text
dist/brown_octopus-0.4.3-py3-none-any.whl
dist/brown_octopus-0.4.3.tar.gz
```
