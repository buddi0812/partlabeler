"""Pick the device and number format once, for every model in the engine.

RTX 30xx/40xx and newer: CUDA + bfloat16. Older GPUs such as Colab's T4 (compute 7.5, no fast
bfloat16): CUDA + float16. No GPU: CPU + float32. Override with PARTLABELER_DEVICE / PARTLABELER_DTYPE.
"""
import os

import torch

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def device() -> str:
    return os.environ.get("PARTLABELER_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")


def dtype(dev: str | None = None) -> torch.dtype:
    if os.environ.get("PARTLABELER_DTYPE"):
        return _DTYPES[os.environ["PARTLABELER_DTYPE"]]
    dev = dev or device()
    if dev != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def describe() -> str:
    dev = device()
    name = torch.cuda.get_device_name() if dev == "cuda" else "CPU"
    mem = f", {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB" if dev == "cuda" else ""
    return f"{name}{mem}, {str(dtype(dev)).replace('torch.', '')}"


def memory() -> str:
    """GPU memory held by this process, e.g. "3.1 of 12 GB"; empty before any model is on the GPU."""
    if not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return ""
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    return f"{torch.cuda.memory_reserved() / 2**30:.1f} of {total:.0f} GB"


def free_memory_gb() -> float | None:
    """Free memory on the GPU right now (all processes), or None without a GPU."""
    if device() != "cuda":
        return None
    return torch.cuda.mem_get_info()[0] / 2**30


def free_gpu_memory() -> None:
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
