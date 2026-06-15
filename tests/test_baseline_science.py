"""Scientific validation of the ingest + classical-decode pipeline.

Beyond byte-exactness, the pipeline must reproduce the *physics* Google reported:
on a below-threshold device the logical error rate per cycle falls as the code
distance grows. Getting this right exercises ingest, the shipped DEMs, the MWPM
decoder, and the per-cycle metric together -- a wiring bug anywhere breaks it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ising_sim2real.decoders.pymatching_decoder import PyMatchingDecoder
from ising_sim2real.ingest.detectors import measurements_to_detectors
from ising_sim2real.ingest.willow import (
    WillowConfig,
    load_run,
    load_shipped_detection_data,
)
from ising_sim2real.metrics import logical_error_per_cycle, logical_error_rate

# One representative patch per distance (deep circuit for a clean per-cycle rate).
_PATCH = {3: "q4_5", 5: "q6_5", 7: "q6_7"}
_ROUNDS = 110
_SHOTS = 20000


def _per_cycle_ler(willow_dir: Path, cfg: WillowConfig) -> float:
    run = load_run(willow_dir, cfg)
    det = measurements_to_detectors(
        run.circuit, run.measurements, sweep_bits=run.sweep_bits
    )
    actual = load_shipped_detection_data(willow_dir, cfg).observables
    pred = PyMatchingDecoder.from_dem(run.dem_si1000).decode_batch(
        det.detectors[:_SHOTS]
    ).predictions
    return logical_error_per_cycle(
        logical_error_rate(pred, actual[:_SHOTS]), cfg.rounds
    )


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_error_suppression_with_code_distance(willow_dir: Path, basis: str) -> None:
    pc = {
        d: _per_cycle_ler(willow_dir, WillowConfig(d, basis, _ROUNDS, _PATCH[d]))
        for d in (3, 5, 7)
    }
    # Strict below-threshold suppression: each step up in distance helps.
    assert pc[3] > pc[5] > pc[7], pc
    # Suppression factor Lambda = eps_d / eps_{d+2} is comfortably above 1.
    assert pc[3] / pc[5] > 1.15 and pc[5] / pc[7] > 1.15, pc
    # Sanity bounds for distance-3..7 Willow per-cycle LER.
    assert all(0.001 < v < 0.02 for v in pc.values()), pc
