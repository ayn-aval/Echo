"""The single place where this project decides which hardware to run on.

Apple Silicon Macs reach their GPU through PyTorch's `mps` backend, not CUDA.
A handful of operations have no mps implementation; PYTORCH_ENABLE_MPS_FALLBACK
lets those run on the CPU instead of raising, at a speed cost rather than a
crash. It must be set before torch is imported, which is why it is set here and
every other module reaches the GPU through this file.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402  (must come after the env var above)


def get_device() -> torch.device:
    """mps -> cuda -> cpu, in that order."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def describe() -> str:
    device = get_device()
    if device.type == "mps":
        return ("mps (Apple Silicon GPU). Unsupported operations fall back to CPU "
                "silently — if something is unexpectedly slow, that is why.")
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return "cpu (no GPU found — everything will be slow)"


if __name__ == "__main__":
    print(f"torch {torch.__version__}")
    print(f"device: {get_device()}")
    print(describe())
