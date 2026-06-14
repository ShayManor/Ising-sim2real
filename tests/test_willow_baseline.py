"""Reproduction test: PyMatching baseline on Google Willow hardware data.

Phase 2 goal: PyMatching on Willow must recover Google's published numbers.
  - d=7 ≈ 0.143%/cycle  (Google Willow paper)
  - Λ(d=5 → d=7) ≈ 2.14

Currently only d3_at_q4_5/X/r13 is downloaded.  The d=3 test is a weaker
sanity check: LER must be positive, well below 50%, and agree with Google's
bundled correlated-matching predictions to within a tolerance (the two decoders
differ — correlated matching vs. standard MWPM — but should broadly agree).

Tests for d=5 and d=7 skip gracefully when that data is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import stim

from ising_sim2real.decoders.base import PyMatchingDecoder
from ising_sim2real.ingest.dataset import discover_configs
from ising_sim2real.ingest.willow import WillowConfig, load_run
from ising_sim2real.paths import DATA_DIR

# Locate the Willow dataset directory once at import time.
_WILLOW_DIR: Path | None = next(DATA_DIR.glob("google_*"), None)


def _require_config(distance: int, basis: str) -> tuple[Path, WillowConfig]:
    """Return (willow_dir, config) or skip the test if the data is absent."""
    if _WILLOW_DIR is None:
        pytest.skip("No Willow data directory found under data/")
    configs = [
        c
        for c in discover_configs(_WILLOW_DIR)
        if c.distance == distance and c.basis == basis
    ]
    if not configs:
        pytest.skip(
            f"No d={distance} {basis} config downloaded "
            f"(expected under {_WILLOW_DIR})"
        )
    return _WILLOW_DIR, configs[0]


# ---------------------------------------------------------------------------
# d=3 sanity check (data present in the repo)
# ---------------------------------------------------------------------------


def test_d3_pymatching_sanity() -> None:
    """PyMatching on d=3 Willow data: LER is finite, sane, and tracks bundled predictions."""
    willow_dir, cfg = _require_config(distance=3, basis="X")
    run = load_run(willow_dir, cfg)

    assert run.dem_si1000 is not None, (
        "Bundled error_model.dem missing — check decoding_results/ subdirectory"
    )

    decoder = PyMatchingDecoder.from_dem(run.dem_si1000)
    result = decoder.decode_batch(run.detection_data.detectors)

    shots = run.detection_data.observables.shape[0]
    errors = int(
        (result.predictions != run.detection_data.observables).any(axis=1).sum()
    )
    ler_per_cycle = errors / (shots * cfg.rounds)

    # d=3 is not well-suppressed; LER/cycle should be positive but well below random.
    assert 0 < ler_per_cycle < 0.20, (
        f"d={cfg.distance} T={cfg.rounds} LER/cycle={ler_per_cycle:.4%} "
        f"outside expected range (0, 20%)"
    )

    # Compare against Google's bundled correlated-matching predictions.
    # PyMatching (MWPM) and correlated matching should agree on ≥90% of shots.
    bundled_path = (
        willow_dir
        / f"d{cfg.distance}_at_{cfg.orientation}"
        / cfg.basis
        / f"r{cfg.rounds}"
        / "decoding_results"
        / "correlated_matching_decoder_with_si1000_prior"
        / "obs_flips_predicted.b8"
    )
    if bundled_path.is_file():
        bundled = stim.read_shot_data_file(
            path=str(bundled_path),
            format="b8",
            num_measurements=run.circuit.num_observables,
        )
        bundled_bool = np.asarray(bundled, dtype=bool).reshape(-1, run.circuit.num_observables)
        agreement = (result.predictions == bundled_bool).all(axis=1).mean()
        assert agreement > 0.90, (
            f"PyMatching agrees with bundled correlated-matching predictions on "
            f"only {agreement:.1%} of shots (expected >90%)"
        )


# ---------------------------------------------------------------------------
# Ground-truth reproduction checks (d=5, d=7): skip if data absent
# ---------------------------------------------------------------------------

# Approximate published LER/cycle from the Google Willow paper.
_PUBLISHED_LER_PER_CYCLE = {5: 0.00306, 7: 0.00143}
_TOLERANCE = 2.0   # allow 2× published value (MWPM vs correlated matching)


@pytest.mark.parametrize("distance,basis", [(5, "X"), (5, "Z"), (7, "X"), (7, "Z")])
def test_higher_distance_ler(distance: int, basis: str) -> None:
    """d=5/7 PyMatching LER/cycle is within 2× of Google's published values."""
    willow_dir, cfg = _require_config(distance=distance, basis=basis)
    run = load_run(willow_dir, cfg)

    assert run.dem_si1000 is not None

    decoder = PyMatchingDecoder.from_dem(run.dem_si1000)
    result = decoder.decode_batch(run.detection_data.detectors)

    shots = run.detection_data.observables.shape[0]
    errors = int(
        (result.predictions != run.detection_data.observables).any(axis=1).sum()
    )
    ler_per_cycle = errors / (shots * cfg.rounds)

    if distance in _PUBLISHED_LER_PER_CYCLE:
        limit = _PUBLISHED_LER_PER_CYCLE[distance] * _TOLERANCE
        assert ler_per_cycle < limit, (
            f"d={distance} {basis} LER/cycle={ler_per_cycle:.4%} "
            f"exceeds {_TOLERANCE}× published {_PUBLISHED_LER_PER_CYCLE[distance]:.4%}"
        )
