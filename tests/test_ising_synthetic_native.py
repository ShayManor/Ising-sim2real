"""The Ising pre-decoder works -- on synthetic data in the model's native layout.

This is the control that isolates *model usage* from the Willow lattice mapping.
Data is sampled from NVIDIA's own ``MemoryCircuit`` (the circuit the pretrained
model's residual is laid out for, ``add_boundary_detectors=True``), fed to
``IsingPreDecoder`` in native mode (``circuit=None`` -> no reorder), and matched
against the ``MemoryCircuit`` DEM.

On this in-distribution data the pre-decoder lands at MWPM-level LER (it should:
the public models were trained/evaluated on exactly this path). If instead the
residual matcher is built from ``stim.Circuit.generated`` -- a different detector
ordering -- the LER collapses to chance, which is the bug this test guards.

Skips if the vendored Ising code or weights are absent.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ising_sim2real.paths import ISING_CODE

if not ISING_CODE.exists():
    pytest.skip("vendored Ising-Decoding code absent", allow_module_level=True)

import pymatching  # noqa: E402

from ising_sim2real.ising.loader import ISING_MODELS, load_ising_model  # noqa: E402
from ising_sim2real.ising.predecoder import IsingPreDecoder, memory_circuit  # noqa: E402
from ising_sim2real.metrics import logical_error_rate  # noqa: E402


def _have_weights(name: str = "fast") -> bool:
    from ising_sim2real.ising.loader import ISING_ROOT, MODELS_DIR

    spec = ISING_MODELS[name]
    return (MODELS_DIR / spec.filename).exists() or (
        ISING_ROOT / "models" / spec.filename
    ).exists()


@pytest.mark.skipif(not _have_weights(), reason="Ising weights missing; run scripts/setup_ising.py")
@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("distance", [3, 5, 7])
def test_synthetic_native_matches_mwpm(distance: int, basis: str) -> None:
    """Ising on native synthetic data is at MWPM level (not chance) at every D."""
    rounds, shots, p, rotation = 10, 1500, 2e-3, "XV"
    device = torch.device("cpu")

    mc = memory_circuit(distance, rounds, basis, rotation, p)
    circuit = mc.stim_circuit
    nobs = circuit.num_observables
    dem = circuit.detector_error_model(decompose_errors=True, approximate_disjoint_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)

    dets, obs = circuit.compile_detector_sampler().sample(shots=shots, separate_observables=True)
    dets = dets.astype(np.uint8)
    obs = obs.astype(bool).reshape(-1, nobs)

    mwpm = logical_error_rate(
        np.asarray(matcher.decode_batch(dets), dtype=bool).reshape(-1, nobs), obs
    )

    model, _ = load_ising_model("fast", device=device)
    # circuit=None => native mode: dets already in MemoryCircuit emission order.
    dec = IsingPreDecoder(model, None, basis, distance, rounds, device, rotation=rotation, syn_noise=p)
    ising = logical_error_rate(dec.decode_batch(dets).predictions, obs)

    # Far below chance, and competitive with MWPM (the bug parked it at ~0.5).
    assert ising < 0.1, f"Ising at/near chance on native d{distance} {basis}: {ising}"
    assert ising <= 1.5 * mwpm + 0.01, f"Ising ({ising}) >> MWPM ({mwpm}) on native data"
