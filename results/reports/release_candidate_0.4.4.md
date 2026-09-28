# Brown Octopus 0.4.4 release-candidate validation

## Artifacts

```text
dist/brown_octopus-0.4.4-py3-none-any.whl
SHA-256: 99F78494223F82318C9934C5FDBE1542A6125F5A4B536BA0380EF4B041386B94

dist/brown_octopus-0.4.4.tar.gz
SHA-256: 939BB43ABF283AF5EE03A76BE957D4A281724DC725024FA648776D6432ADA9F5
```

Both artifacts were built before validation and were not rebuilt afterward.

## Clean installed-wheel validation

Environment: Windows x86-64, Python 3.13.15, isolated `.clean-env`, wheel
installed from the built artifact, managed model directory supplied separately.

```text
158 passed, 5 deselected, 2 warnings
```

The installed package reported version `0.4.4`. Analyzer initialization also
completed successfully with the managed model directory.

CLI smoke tests passed for:

```text
brown-octopus --help
brown-octopus --version
brown-octopus version
brown-octopus doctor
brown-octopus inspect
```

Doctor reported both models, the index, and Brown Octopus readiness as OK.

## Package validation

- Wheel metadata: valid; name `brown-octopus`, version `0.4.4`, Python `>=3.13`.
- Sdist metadata: present and readable.
- Wheel import: `from brown_octopus import Octopus` passed.
- Wheel build: passed.

## Platform status

The repository CI workflow is configured for Python 3.13 on Ubuntu and
Windows, plus an Ubuntu distribution-validation job. The completed local
acceptance run was Windows/Python 3.13. GitHub CI results were not reproduced
locally in this run.

## Retrieval freeze

No retrieval, V3, model, selector, session, source, index, or bootstrap
architecture changes were made for this release build. Only the package
version metadata was advanced from 0.4.3 to 0.4.4.
