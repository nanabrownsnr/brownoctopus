# ToolRet smoke evaluation on a GPU VM

This workflow uses the frozen Brown Octopus V3 code. It does not change V3
production parameters or the default 122-tool index.

## VM requirements

- NVIDIA driver visible to the VM (`nvidia-smi` succeeds)
- CUDA-enabled PyTorch environment
- NVIDIA GPU with at least 8 GB VRAM; an RTX 3070 is suitable
- 16 GB system RAM recommended
- 20 GB free disk space for model caches, ToolRet files, and shards

## Setup after cloning

```bash
uv sync --extra toolret
python -m spacy download en_core_web_trf
python scripts/check_gpu.py
python scripts/setup_models.py
python scripts/fetch_toolret_smoke_data.py
```

If `check_gpu.py` reports `CUDA available: False`, fix the CUDA-enabled
PyTorch/driver installation before starting the index build.

## Calibrate and build

```bash
python -m evals.toolret.build --calibrate
python -m evals.toolret.build --batch-size 32 --shard-size 1000
```

The calibration is advisory. Choose the fastest stable batch size that fits
the GPU. The build writes `data/indexes/toolret_smoke/shards/` and a manifest
after each shard, so an interrupted build resumes with the same command.

## Run the smoke test

```bash
python -m evals.toolret.smoke
```

Smoke evaluation refuses to start if the completed index is missing, preventing
an accidental multi-hour rebuild. It writes immutable raw JSON under
`results/raw/` and a report under `results/reports/`.

## Windows PowerShell

The same commands work with `uv run`:

```powershell
uv sync --extra toolret
python -m spacy download en_core_web_trf
python scripts/check_gpu.py
python scripts/setup_models.py
python scripts/fetch_toolret_smoke_data.py
python -m evals.toolret.build --calibrate
python -m evals.toolret.build --batch-size 32 --shard-size 1000
python -m evals.toolret.smoke
```

Generated benchmark data and indexes are intentionally gitignored. Commit the
source code and this setup document, then build the artifacts on the VM.
