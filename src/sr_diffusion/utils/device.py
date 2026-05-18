from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch


def get_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def autocast_context(device: torch.device, dtype_name: str | None) -> ContextManager[None]:
    if dtype_name in (None, "fp32", "float32"):
        return nullcontext()
    if device.type != "cuda":
        return nullcontext()
    if dtype_name in ("bf16", "bfloat16"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if dtype_name in ("fp16", "float16"):
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported dtype: {dtype_name}")
