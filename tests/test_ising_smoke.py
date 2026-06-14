"""Smoke test: each Ising model loads and runs a forward pass on CPU.

Skips if the vendored repo or weights are absent (run scripts/setup_ising.py).
"""

from __future__ import annotations

import pytest
import torch

from ising_sim2real.ising.loader import ISING_MODELS, load_ising_model
from ising_sim2real.paths import ISING_CODE


def _have_assets(name: str) -> bool:
    from ising_sim2real.ising.loader import ISING_ROOT, MODELS_DIR
    spec = ISING_MODELS[name]
    weights = (MODELS_DIR / spec.filename).exists() or (
        ISING_ROOT / "models" / spec.filename
    ).exists()
    return ISING_CODE.exists() and weights


@pytest.mark.parametrize("name", list(ISING_MODELS))
def test_load_and_forward_cpu(name: str) -> None:
    if not _have_assets(name):
        pytest.skip(f"assets for '{name}' missing; run scripts/setup_ising.py")

    device = torch.device("cpu")
    model, info = load_ising_model(name, device=device)

    assert info.input_channels == 4
    assert info.receptive_field == ISING_MODELS[name].receptive_field
    assert info.num_params > 0

    r = info.receptive_field
    x = torch.randint(0, 2, (2, 4, r, r, r), dtype=torch.float32)
    with torch.no_grad():
        y = model(x)

    # Fully-convolutional, same-padding: spatial/temporal dims are preserved.
    assert y.shape == (2, info.out_channels, r, r, r)
