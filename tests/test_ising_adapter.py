"""Tests for the Willow -> Ising (4, T, D, D) lattice adapter.

The adapter's correctness standard is NVIDIA's own ``dets_to_predecoder_inputs``:
the public models were trained on exactly that transform, so a correct adapter
must reproduce it bit-for-bit. We verify that on synthetic circuits (where the
transform is the ground truth), then confirm a real Willow lattice has the right
shape and is consumable by the pretrained model.
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
    detection_events_to_lattice,
)


def _nvidia_transform():
    import sys

    if str(ISING_CODE) not in sys.path:
        sys.path.insert(0, str(ISING_CODE))
    from data.predecoder_transform import dets_to_predecoder_inputs  # type: ignore

    return dets_to_predecoder_inputs


def _synthetic(distance: int, rounds: int, basis: str):
    import stim

    code = f"surface_code:rotated_memory_{basis.lower()}"
    return stim.Circuit.generated(
        code,
        distance=distance,
        rounds=rounds,
        after_clifford_depolarization=0.02,
        before_measure_flip_probability=0.01,
    )


@pytest.mark.parametrize(
    "distance,rounds,basis",
    [(3, 2, "Z"), (3, 4, "Z"), (5, 3, "Z"), (3, 5, "X"), (5, 4, "X"), (7, 3, "Z")],
)
def test_matches_nvidia_transform_bit_for_bit(distance, rounds, basis) -> None:
    """On a synthetic circuit our lattice == NVIDIA's dets_to_predecoder_inputs."""
    circuit = _synthetic(distance, rounds, basis)
    det = circuit.compile_detector_sampler().sample(shots=512).astype(np.uint8)

    reference, _, _ = _nvidia_transform()(
        torch.as_tensor(det, dtype=torch.int64),
        distance=distance,
        n_rounds=rounds,
        basis=basis,
    )
    layout = LatticeLayout.from_circuit(circuit, basis=basis, distance=distance)
    mine = detection_events_to_lattice(det, layout)

    assert mine.shape == reference.shape == (512, 4, rounds, distance, distance)
    assert torch.equal(mine, reference.to(torch.float32))


def test_recovers_lattice_from_scrambled_detector_order() -> None:
    """A circuit whose detectors are permuted still yields the canonical lattice.

    Proves the placement is driven by detector coordinates, not by accidental
    emission order -- the property that lets it absorb Willow's ordering.
    """
    distance, rounds, basis = 5, 4, "Z"
    circuit = _synthetic(distance, rounds, basis)
    det = circuit.compile_detector_sampler().sample(shots=256).astype(np.uint8)
    canonical = detection_events_to_lattice(
        det, LatticeLayout.from_circuit(circuit, basis=basis, distance=distance)
    )

    # Rebuild only what the adapter reads -- detector coordinates -- in a
    # permuted order, then check the lattice is unchanged.
    import stim

    rng = np.random.default_rng(7)
    perm = rng.permutation(circuit.num_detectors)
    coords = circuit.get_detector_coordinates()
    builder = stim.Circuit()
    for new_idx in range(circuit.num_detectors):
        builder.append("DETECTOR", [], coords[int(perm[new_idx])])
    shuffled_layout = LatticeLayout.from_circuit(builder, basis=basis, distance=distance)

    det_shuffled = det[:, perm]
    recovered = detection_events_to_lattice(det_shuffled, shuffled_layout)
    assert torch.equal(recovered, canonical)


def test_reorder_is_a_permutation() -> None:
    circuit = _synthetic(3, 6, "Z")
    layout = LatticeLayout.from_circuit(circuit, basis="Z", distance=3)
    assert sorted(layout.to_nvidia.tolist()) == list(range(layout.num_detectors))


def _willow_events(willow_dir: Path, cfg: WillowConfig):
    run = load_run(willow_dir, cfg)
    det = measurements_to_detectors(
        run.circuit, run.measurements, sweep_bits=run.sweep_bits
    )
    return run.circuit, det.detectors


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
