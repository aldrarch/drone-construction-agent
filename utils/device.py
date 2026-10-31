"""
Utility for selecting the device used by the model.
"""

from __future__ import annotations

import torch


def get_device() -> torch.device:
    """Return the best available device: CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_info() -> dict:
    """Return a small human-readable description of the selected device."""
    device = get_device()
    info: dict = {"device": str(device), "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "unknown"
        info["gpu_name"] = torch.cuda.get_device_name(0)
    return info
