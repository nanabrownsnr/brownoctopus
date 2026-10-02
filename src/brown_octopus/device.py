"""Runtime accelerator detection for local model inference."""


def detect_compute_device() -> str:
    """Return the best device visible to the current PyTorch runtime.

    CUDA is preferred, followed by Apple MPS and Intel XPU when supported by
    the installed PyTorch build. CPU is always the safe fallback.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return "mps"

    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"

    return "cpu"

