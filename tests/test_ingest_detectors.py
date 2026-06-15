"""Tests for deriving detection events / observable flips from measurements.

The ground truth is the dataset's own shipped ``detection_events.b8`` and
``obs_flips_actual.b8``, which Google produced with Stim's ``m2d`` converter.
If our derivation does not reproduce those bit-for-bit, nothing downstream can
be trusted (CLAUDE.md method step 2: the validation gate).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import stim

from ising_sim2real.ingest.detectors import measurements_to_detectors


def _leaf(willow_dir: Path) -> Path:
    return willow_dir / "d3_at_q4_5" / "Z" / "r01"


def _load_inputs(leaf: Path):
    """Read circuit + raw measurement/sweep records straight from disk (test side)."""
    circuit = stim.Circuit.from_file(leaf / "circuit_ideal.stim")
    measurements = stim.read_shot_data_file(
        path=str(leaf / "measurements.b8"),
        format="b8",
        num_measurements=circuit.num_measurements,
        bit_pack=False,
    )
    sweep_bits = stim.read_shot_data_file(
        path=str(leaf / "sweep_bits.b8"),
        format="b8",
        num_measurements=circuit.num_sweep_bits,
        bit_pack=False,
    )
    return circuit, measurements.astype(bool), sweep_bits.astype(bool)


def test_derived_detectors_match_shipped(willow_dir: Path) -> None:
    leaf = _leaf(willow_dir)
    circuit, measurements, sweep_bits = _load_inputs(leaf)

    shipped_det = stim.read_shot_data_file(
        path=str(leaf / "detection_events.b8"),
        format="b8",
        num_detectors=circuit.num_detectors,
        bit_pack=False,
    ).astype(bool)

    result = measurements_to_detectors(circuit, measurements, sweep_bits=sweep_bits)

    assert result.detectors.shape == shipped_det.shape
    assert np.array_equal(result.detectors, shipped_det)


def test_derived_observables_match_shipped(willow_dir: Path) -> None:
    leaf = _leaf(willow_dir)
    circuit, measurements, sweep_bits = _load_inputs(leaf)

    shipped_obs = stim.read_shot_data_file(
        path=str(leaf / "obs_flips_actual.b8"),
        format="b8",
        num_observables=circuit.num_observables,
        bit_pack=False,
    ).astype(bool)

    result = measurements_to_detectors(circuit, measurements, sweep_bits=sweep_bits)

    assert result.observables.shape == shipped_obs.shape
    assert np.array_equal(result.observables, shipped_obs)


# A representative spread across distance, patch, basis, and round count. The full
# 420-config byte-exact sweep is verified out-of-band; this keeps a fast guard in
# the suite. Each tuple is (patch, basis, rounds_dir).
_SPREAD = [
    ("d3_at_q4_5", "Z", "r01"),
    ("d3_at_q4_5", "X", "r13"),
    ("d3_at_q8_9", "Z", "r30"),
    ("d5_at_q6_5", "Z", "r10"),
    ("d5_at_q6_9", "X", "r50"),
    ("d7_at_q6_7", "Z", "r10"),
    ("d7_at_q6_7", "X", "r110"),
]


@pytest.mark.parametrize("patch,basis,rounds_dir", _SPREAD)
def test_m2d_matches_shipped_across_configs(
    willow_dir: Path, patch: str, basis: str, rounds_dir: str
) -> None:
    """Derived detection events AND observable flips equal the shipped b8 exactly."""
    leaf = willow_dir / patch / basis / rounds_dir
    circuit, measurements, sweep_bits = _load_inputs(leaf)
    result = measurements_to_detectors(circuit, measurements, sweep_bits=sweep_bits)

    shipped_det = stim.read_shot_data_file(
        path=str(leaf / "detection_events.b8"), format="b8",
        num_detectors=circuit.num_detectors, bit_pack=False,
    ).astype(bool)
    shipped_obs = stim.read_shot_data_file(
        path=str(leaf / "obs_flips_actual.b8"), format="b8",
        num_observables=circuit.num_observables, bit_pack=False,
    ).astype(bool)

    assert np.array_equal(result.detectors, shipped_det)
    assert np.array_equal(result.observables, shipped_obs)
    # Structural invariant relied on by the lattice adapter.
    distance = int(patch[1])
    half = (distance * distance - 1) // 2
    rounds = int(rounds_dir[1:])
    assert circuit.num_detectors == 2 * rounds * half
