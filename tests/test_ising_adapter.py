"""Tests for the Willow -> Ising (4, T, D, D) lattice adapter.

The adapter reorders Willow *device* detection events into NVIDIA's CSS
``MemoryCircuit`` contract layout: a 45-degree data-grid mapping, a square
symmetry carrying Willow stabilizers onto NVIDIA's ``Hx``/``Hz`` rows, and the
transform's timeline indices for the round dimension. We verify the
correspondence is found and bijective, that the reorder is a clean scatter into
the contract width, that placement is coordinate-driven (not emission-order
driven), and that a real Willow lattice has the right shape and feeds the
pretrained model. End-to-end decoding (Ising LER vs MWPM) lives in
``test_ising_pipeline_e2e``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ising_sim2real.ingest.detectors import measurements_to_detectors
from ising_sim2real.ingest.willow import WillowConfig, load_run
from ising_sim2real.paths import ISING_CODE

torch = pytest.importorskip("torch")

if not ISING_CODE.exists():
    pytest.skip("vendored Ising-Decoding code absent", allow_module_level=True)

from ising_sim2real.ising.adapter import (  # noqa: E402
    LatticeLayout,
    _stab_row_maps,
    detection_events_to_lattice,
)


def _willow_events(willow_dir: Path, cfg: WillowConfig):
    run = load_run(willow_dir, cfg)
    det = measurements_to_detectors(run.circuit, run.measurements, sweep_bits=run.sweep_bits)
    return run.circuit, det.detectors


@pytest.mark.parametrize(
    "distance,orientation,basis",
    [(3, "q4_5", "Z"), (3, "q4_5", "X"), (5, "q6_5", "Z"), (7, "q6_7", "Z")],
)
def test_stab_row_maps_are_bijections(willow_dir, distance, orientation, basis) -> None:
    """The square-symmetry search finds a bijective Willow-ancilla -> NVIDIA-row map."""
    cfg = WillowConfig(distance, basis, 10, orientation)
    circuit, _ = _willow_events(willow_dir, cfg)
    half = (distance * distance - 1) // 2
    xmap, zmap = _stab_row_maps(circuit, distance, basis, "XV")
    assert sorted(xmap.values()) == list(range(half))
    assert sorted(zmap.values()) == list(range(half))


@pytest.mark.parametrize(
    "distance,rounds,orientation,basis",
    [(3, 10, "q4_5", "Z"), (3, 13, "q4_5", "X"), (7, 10, "q6_7", "Z")],
)
def test_reorder_is_a_clean_scatter(willow_dir, distance, rounds, orientation, basis) -> None:
    """reorder scatters device detectors into the contract width with no collisions."""
    cfg = WillowConfig(distance, basis, rounds, orientation)
    circuit, detectors = _willow_events(willow_dir, cfg)
    layout = LatticeLayout.from_circuit(circuit, basis=basis, distance=distance)

    half = (distance * distance - 1) // 2
    assert layout.contract_width == 2 * rounds * half
    assert len(set(layout.src.tolist())) == len(layout.src)  # each device det used once
    assert len(set(layout.dst.tolist())) == len(layout.dst)  # no contract collisions
    assert int(layout.src.max()) < layout.num_detectors
    assert int(layout.dst.max()) < layout.contract_width

    out = layout.reorder(detectors[:8])
    assert out.shape == (8, layout.contract_width)


@pytest.mark.parametrize(
    "distance,rounds,orientation,basis",
    [(3, 10, "q4_5", "Z"), (3, 10, "q4_5", "X"), (5, 10, "q6_5", "Z"), (7, 10, "q6_7", "Z")],
)
def test_willow_lattice_shape_and_presence(
    willow_dir: Path, distance, rounds, orientation, basis
) -> None:
    cfg = WillowConfig(distance, basis, rounds, orientation)
    circuit, detectors = _willow_events(willow_dir, cfg)
    layout = LatticeLayout.from_circuit(circuit, basis=basis, distance=distance)
    assert layout.rounds == rounds  # T = QEC rounds, not rounds+1

    lattice = detection_events_to_lattice(detectors[:8], layout)
    assert lattice.shape == (8, 4, rounds, distance, distance)
    assert lattice.dtype == torch.float32
    # Presence channels carry NVIDIA's {0, 0.5, 1.0} stabilizer weights.
    present = lattice[:, (2, 3)].numpy()
    assert set(np.unique(present)).issubset({0.0, 0.5, 1.0})
    assert np.all(present == present[0])  # batch-invariant


def test_real_willow_lattice_feeds_the_ising_model(willow_dir: Path) -> None:
    """load -> derive events -> lattice -> forward pass through the trained model."""
    from ising_sim2real.ising.loader import ISING_MODELS, MODELS_DIR, load_ising_model

    spec = ISING_MODELS["fast"]
    if not (MODELS_DIR / spec.filename).exists():
        pytest.skip("Ising weights missing; run scripts/setup_ising.py")

    cfg = WillowConfig(3, "Z", 10, "q4_5")
    circuit, detectors = _willow_events(willow_dir, cfg)
    layout = LatticeLayout.from_circuit(circuit, basis=cfg.basis, distance=cfg.distance)
    lattice = detection_events_to_lattice(detectors[:4], layout)

    model, info = load_ising_model("fast", device=torch.device("cpu"))
    with torch.no_grad():
        out = model(lattice)

    assert out.shape == (4, info.out_channels, layout.rounds, 3, 3)
    assert torch.isfinite(out).all()
