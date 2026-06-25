"""Load NVIDIA's pretrained Ising surface-code pre-decoders.

The ``.pt`` checkpoints are bare state dicts; the architecture lives in the
vendored ``Ising-Decoding`` repo. We build the model from its public registry
(``model_id``) and load the weights, mirroring the repo's own inference path
(``code/workflows/run.py`` / ``code/export/checkpoint_to_safetensors.py``).

Public model mapping (confirmed in ``code/tests/test_inference_public_model.py``):
    fast     -> model_id 1, receptive field R=9   (Ising-Decoder-SurfaceCode-1-Fast.pt)
    accurate -> model_id 4, receptive field R=13  (Ising-Decoder-SurfaceCode-1-Accurate.pt)

Both CPU and GPU are supported: the model is moved to whatever device the caller
resolves. Input layout is ``(B, 4, T, D, D)`` -> output ``(B, out_channels, T, D, D)``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from ising_sim2real.paths import ISING_CODE, ISING_ROOT, MODELS_DIR


@dataclass(frozen=True)
class IsingModelSpec:
    """Static facts about a public Ising model variant."""

    name: str
    model_id: int
    receptive_field: int
    filename: str


# Keyed by the short names exposed on the CLI.
ISING_MODELS: dict[str, IsingModelSpec] = {
    "fast": IsingModelSpec(
        name="fast",
        model_id=1,
        receptive_field=9,
        filename="Ising-Decoder-SurfaceCode-1-Fast.pt",
    ),
    "accurate": IsingModelSpec(
        name="accurate",
        model_id=4,
        receptive_field=13,
        filename="Ising-Decoder-SurfaceCode-1-Accurate.pt",
    ),
}


@dataclass
class IsingModelInfo:
    """Everything the caller needs to describe a loaded model.

    Per the project guardrail, always report code distance ``d`` alongside the
    model receptive field ``R``; ``receptive_field`` is carried here so callers
    can print the two together.
    """

    name: str
    model_id: int
    receptive_field: int
    input_channels: int
    out_channels: int
    num_params: int
    checkpoint: Path
    device: torch.device


def _ensure_ising_on_path(ising_code: Path = ISING_CODE) -> None:
    """Put the vendored repo's ``code/`` dir on sys.path so its modules import."""
    if not ising_code.exists():
        raise FileNotFoundError(
            f"Vendored Ising-Decoding code not found at {ising_code}. "
            "Run `python scripts/setup_ising.py` to clone it."
        )
    p = str(ising_code)
    if p not in sys.path:
        sys.path.insert(0, p)


def resolve_checkpoint(spec: IsingModelSpec, checkpoint: Optional[Path] = None) -> Path:
    """Find the weights file, preferring the project copy, then the clone."""
    if checkpoint is not None:
        ckpt = Path(checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    candidates = [
        MODELS_DIR / spec.filename,                  # .pt from git-lfs (primary)
        MODELS_DIR / f"{spec.name}.safetensors",      # fp16 safetensors (HF fallback)
        ISING_ROOT / "models" / spec.filename,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"No weights for '{spec.name}' found in {[str(c) for c in candidates]}. "
        "Run `python scripts/setup_ising.py` to download them."
    )


def _load_state_dict_from_pt(path: Path, device: torch.device) -> dict:
    """Mirror code/workflows/run.py: unwrap common checkpoint formats + DDP prefix."""
    # weights_only=True: these checkpoints are bare tensor state dicts, so we
    # avoid unpickling arbitrary objects. Fall back only if the file legitimately
    # wraps non-tensor metadata.
    try:
        raw = torch.load(str(path), map_location=device, weights_only=True)
    except Exception:
        raw = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(raw, dict):
        if "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
        elif "state_dict" in raw:
            state_dict = raw["state_dict"]
        else:
            state_dict = raw
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(raw).__name__}")
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def load_ising_model(
    name: str = "fast",
    device: Optional[torch.device] = None,
    checkpoint: Optional[Path] = None,
) -> tuple[torch.nn.Module, IsingModelInfo]:
    """Build the architecture from the registry and load pretrained weights.

    Args:
        name: "fast" or "accurate".
        device: target torch device (defaults to CPU).
        checkpoint: explicit weights path; otherwise resolved from disk.

    Returns:
        (model in eval mode on ``device``, IsingModelInfo).
    """
    if name not in ISING_MODELS:
        raise ValueError(f"Unknown model {name!r}; choose from {list(ISING_MODELS)}.")
    spec = ISING_MODELS[name]
    device = device or torch.device("cpu")

    _ensure_ising_on_path()
    # Imported after sys.path is patched; resolves inside the vendored repo.
    from export.safetensors_utils import _build_minimal_cfg  # type: ignore
    from model.factory import ModelFactory  # type: ignore

    cfg = _build_minimal_cfg(spec.model_id)
    model = ModelFactory.create_model(cfg)

    ckpt = resolve_checkpoint(spec, checkpoint)
    if ckpt.suffix == ".safetensors":
        # HF ships fp16 safetensors; the repo provides a loader that returns a
        # ready model. Use it directly rather than our state-dict path.
        from export.safetensors_utils import load_safetensors  # type: ignore

        model, _meta = load_safetensors(str(ckpt), model_id=spec.model_id, device=str(device))
    else:
        state_dict = _load_state_dict_from_pt(ckpt, device)
        model.load_state_dict(state_dict, strict=True)

    model.eval().to(device)

    info = IsingModelInfo(
        name=spec.name,
        model_id=spec.model_id,
        receptive_field=spec.receptive_field,
        input_channels=int(cfg.model.input_channels),
        out_channels=int(cfg.model.out_channels),
        num_params=sum(p.numel() for p in model.parameters()),
        checkpoint=ckpt,
        device=device,
    )
    return model, info
