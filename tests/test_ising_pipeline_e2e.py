"""End-to-end pipeline checks for the decoders on real Willow data.

Two things are pinned here, reflecting the corrected understanding of the
``LER == 0.5`` investigation:

* **MWPM decodes Willow** at every distance (the validated classical baseline).
* **The Ising pre-decoder on Willow is the open RQ4 step.** The model itself is
  correct -- ``test_ising_synthetic_native`` proves it reaches MWPM-level LER on
  native synthetic data. What is *not* yet solved is mapping Willow's XZZX device
  detectors into the model's CSS ``MemoryCircuit`` emission order (the residual
  matcher now lives in that layout). Until that mapping exists, the Willow Ising
  path does not decode, so it is marked ``xfail``; it flips to XPASS the day the
  ordering is implemented.

Skips if the local Willow tree or the Ising assets are absent.
"""

from __future__ import annotations

import pytest
import torch

from ising_sim2real.decoders.pymatching_decoder import PyMatchingDecoder
from ising_sim2real.ingest.detectors import measurements_to_detectors
from ising_sim2real.ingest.willow import WillowConfig, load_run
from ising_sim2real.ising.loader import ISING_MODELS
from ising_sim2real.metrics import logical_error_rate
from ising_sim2real.paths import ISING_CODE


def _have_ising_assets(name: str = "fast") -> bool:
    from ising_sim2real.ising.loader import ISING_ROOT, MODELS_DIR

    spec = ISING_MODELS[name]
    weights = (MODELS_DIR / spec.filename).exists() or (
        ISING_ROOT / "models" / spec.filename
    ).exists()
    return ISING_CODE.exists() and weights


def _willow_mwpm(willow_dir, cfg: WillowConfig, shots: int) -> float:
    run = load_run(willow_dir, cfg)
    det = measurements_to_detectors(run.circuit, run.measurements, sweep_bits=run.sweep_bits)
    return logical_error_rate(
        PyMatchingDecoder.from_dem(run.dem_si1000).decode_batch(det.detectors[:shots]).predictions,
        det.observables[:shots],
    )


def _willow_ising(willow_dir, cfg: WillowConfig, shots: int, rotation: str = "XV") -> float:
    from ising_sim2real.ising.loader import load_ising_model
    from ising_sim2real.ising.predecoder import IsingPreDecoder

    run = load_run(willow_dir, cfg)
    det = measurements_to_detectors(run.circuit, run.measurements, sweep_bits=run.sweep_bits)
    model, _ = load_ising_model("fast", device=torch.device("cpu"))
    dec = IsingPreDecoder(
        model, run.circuit, cfg.basis, cfg.distance, cfg.rounds, torch.device("cpu"), rotation=rotation
    )
    return logical_error_rate(dec.decode_batch(det.detectors[:shots]).predictions, det.observables[:shots])


@pytest.mark.parametrize("distance,orientation", [(3, "q4_5"), (7, "q6_7")])
def test_willow_mwpm_decodes(willow_dir, distance, orientation) -> None:
    """MWPM is a strong decoder on Willow at every distance (validation gate)."""
    cfg = WillowConfig(distance=distance, basis="Z", rounds=10, orientation=orientation)
    mwpm = _willow_mwpm(willow_dir, cfg, shots=1000)
    assert 0.0 <= mwpm < 0.2, f"MWPM unexpectedly weak on d{distance}: {mwpm}"


@pytest.mark.skipif(not _have_ising_assets(), reason="Ising assets missing; run scripts/setup_ising.py")
@pytest.mark.parametrize("distance,orientation,basis", [(3, "q4_5", "Z"), (3, "q4_5", "X"), (7, "q6_7", "Z")])
def test_willow_ising_decodes(willow_dir, distance, orientation, basis) -> None:
    """The Willow->CSS mapping works: Ising decodes Willow well below chance."""
    cfg = WillowConfig(distance=distance, basis=basis, rounds=10, orientation=orientation)
    ising = _willow_ising(willow_dir, cfg, shots=1000)
    # Far below chance (the bug parked it at ~0.5). MWPM still leads on real hardware.
    assert ising < 0.35, f"Willow Ising not decoding at d{distance} {basis}: {ising}"
