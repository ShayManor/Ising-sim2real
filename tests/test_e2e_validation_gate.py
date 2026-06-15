"""End-to-end validation gate (CLAUDE.md method step 2).

Loads real Willow shots, derives detection events with our pipeline, decodes
them with the classical MWPM baseline against the shipped SI1000 prior, and
checks the result against the dataset's own shipped decoder. No downstream
comparison number is trustworthy until these pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import stim

from ising_sim2real.decoders.pymatching_decoder import PyMatchingDecoder
from ising_sim2real.ingest.detectors import measurements_to_detectors
from ising_sim2real.ingest.willow import (
    WillowConfig,
    config_dir,
    load_run,
    load_shipped_detection_data,
)
from ising_sim2real.metrics import logical_error_per_cycle, logical_error_rate

# Three round counts on the same patch: short (no temporal correlations),
# medium, and deep into the exponential-decay regime.
_ROUNDS = (1, 13, 110)


def _cfg(rounds: int) -> WillowConfig:
    return WillowConfig(distance=3, basis="Z", rounds=rounds, orientation="q4_5")


def _shipped_corr_match_predictions(willow_dir: Path, cfg: WillowConfig) -> np.ndarray:
    path = (
        config_dir(willow_dir, cfg)
        / "decoding_results"
        / "correlated_matching_decoder_with_si1000_prior"
        / "obs_flips_predicted.b8"
    )
    return stim.read_shot_data_file(
        path=str(path), format="b8", num_observables=1, bit_pack=False
    ).astype(bool)


@pytest.mark.parametrize("rounds", _ROUNDS)
def test_derived_events_decode_identically_to_shipped(
    willow_dir: Path, rounds: int
) -> None:
    """Our derived detection events and the shipped ones decode to the same bits."""
    cfg = _cfg(rounds)
    run = load_run(willow_dir, cfg)
    derived = measurements_to_detectors(
        run.circuit, run.measurements, sweep_bits=run.sweep_bits
    )
    shipped = load_shipped_detection_data(willow_dir, cfg)

    decoder = PyMatchingDecoder.from_dem(run.dem_si1000)
    pred_derived = decoder.decode_batch(derived.detectors).predictions
    pred_shipped = decoder.decode_batch(shipped.detectors).predictions

    assert np.array_equal(pred_derived, pred_shipped)


@pytest.mark.parametrize("rounds", _ROUNDS)
def test_mwpm_is_no_better_than_correlated_matching(
    willow_dir: Path, rounds: int
) -> None:
    """Uncorrelated MWPM cannot beat the shipped correlated-matching decoder.

    Both use the same SI1000 prior, so any gap is the correlated decoder's
    advantage. If our pipeline mis-wired detectors or the observable, MWPM would
    spuriously look *better* (or random) -- this catches that.
    """
    cfg = _cfg(rounds)
    run = load_run(willow_dir, cfg)
    derived = measurements_to_detectors(
        run.circuit, run.measurements, sweep_bits=run.sweep_bits
    )
    actual = load_shipped_detection_data(willow_dir, cfg).observables

    pred_mwpm = PyMatchingDecoder.from_dem(run.dem_si1000).decode_batch(
        derived.detectors
    ).predictions
    ler_mwpm = logical_error_rate(pred_mwpm, actual)
    ler_corr = logical_error_rate(_shipped_corr_match_predictions(willow_dir, cfg), actual)

    # MWPM >= correlated matching, within Monte-Carlo noise on 50k shots.
    assert ler_mwpm >= ler_corr - 0.003
    # And both decoders agree the device is well below 50% (sane wiring).
    assert ler_corr < 0.5 and ler_mwpm < 0.5


def test_per_cycle_logical_error_rate_is_stable_across_round_counts(
    willow_dir: Path,
) -> None:
    """Per-cycle LER should be ~constant once past the first cycle.

    This is the quantity Google reports; a correct pipeline makes the deep-circuit
    (r=110) and medium (r=13) per-cycle rates agree to within ~20%.
    """
    per_cycle = {}
    for rounds in (13, 110):
        cfg = _cfg(rounds)
        run = load_run(willow_dir, cfg)
        derived = measurements_to_detectors(
            run.circuit, run.measurements, sweep_bits=run.sweep_bits
        )
        actual = load_shipped_detection_data(willow_dir, cfg).observables
        pred = PyMatchingDecoder.from_dem(run.dem_si1000).decode_batch(
            derived.detectors
        ).predictions
        per_cycle[rounds] = logical_error_per_cycle(
            logical_error_rate(pred, actual), rounds
        )

    lo, hi = sorted(per_cycle.values())
    assert hi / lo < 1.2, per_cycle
    # Distance-3 Willow per-cycle LER lands in the low-percent regime.
    assert all(0.001 < v < 0.05 for v in per_cycle.values()), per_cycle
