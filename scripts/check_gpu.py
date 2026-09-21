"""Verify that the active Python environment can see the NVIDIA GPU."""

import sys

import torch


def main() -> int:
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA compiled: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("No CUDA device is visible to this Python environment.", file=sys.stderr)
        return 1
    for index in range(torch.cuda.device_count()):
        print(f"GPU {index}: {torch.cuda.get_device_name(index)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
