"""Torch device resolution shared across the harness.

The whole panel must run with or without a local GPU. `resolve_device("auto")`
picks CUDA when available and falls back to CPU, so the same code path works on a
laptop and on a CUDA box. Explicit "cpu" / "cuda" / "mps" are honored as given.
"""

from __future__ import annotations

import torch

_VALID_PREFIXES = ("cpu", "cuda", "mps")


def resolve_device(spec: str = "auto") -> torch.device:
    """Resolve a device spec to a concrete torch.device.

    Args:
        spec: One of "auto", "cpu", "cuda", "cuda:N", or "mps".
              "auto" -> CUDA if available, else CPU (never MPS, to keep results
              reproducible with the GPU reference; request "mps" explicitly).

    Raises:
        RuntimeError: if a specific accelerator is requested but unavailable.
    """
    spec = (spec or "auto").strip().lower()

    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not spec.startswith(_VALID_PREFIXES):
        raise ValueError(
            f"Unrecognized device spec {spec!r}; expected auto|cpu|cuda|cuda:N|mps."
        )

    if spec.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False. "
            "Use --device cpu, or install a CUDA build of torch (see README)."
        )
    if spec == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available on this machine.")

    return torch.device(spec)


def describe_device(device: torch.device) -> str:
    """Human-readable one-liner for logging."""
    if device.type == "cuda":
        idx = device.index or 0
        return f"cuda:{idx} ({torch.cuda.get_device_name(idx)})"
    return device.type
