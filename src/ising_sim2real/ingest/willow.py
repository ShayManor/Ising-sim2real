"""Load Google Willow below-threshold circuits and measurement records.

SCAFFOLD ONLY -- interfaces are defined; bodies are not implemented yet.

Dataset: Zenodo 10.5281/zenodo.13273331, rotated surface code at d = 3, 5, 7,
X and Z memory. Each configuration ships a Stim circuit, measurement records,
and reference DEMs (SI1000 circuit-level and an RL-optimized DEM).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim

from ising_sim2real.ingest.detectors import DetectionData

# Decoding-result subdirectories that ship a reference DEM prior.
_SI1000_PATHWAY = "correlated_matching_decoder_with_si1000_prior"
_RL_PATHWAY = "correlated_matching_decoder_with_rl_optimized_prior"


@dataclass(frozen=True)
class WillowConfig:
    """One Willow experimental configuration."""

    distance: int          # code distance d (3, 5, or 7)
    basis: str             # "X" or "Z" memory
    rounds: int            # number of QEC rounds / cycles
    orientation: str       # patch location label, e.g. "q4_5"


@dataclass
class WillowRun:
    """A loaded configuration: circuit, raw measurements, and reference DEMs."""

    config: WillowConfig
    circuit: stim.Circuit
    measurements: np.ndarray          # shape (shots, num_measurements), bool
    sweep_bits: np.ndarray            # shape (shots, num_sweep_bits), bool
    dem_si1000: stim.DetectorErrorModel | None
    dem_rl: stim.DetectorErrorModel | None


def config_dir(data_dir: Path, config: WillowConfig) -> Path:
    """Resolve the leaf directory for a configuration.

    Layout (per the dataset README): ``<patch>/<basis>/<cycles>`` where the patch
    encodes distance + location (e.g. ``d3_at_q4_5``) and the cycles directory is
    ``r`` followed by the round count zero-padded to two digits (``r01``, ``r13``,
    ``r110``).
    """
    patch = f"d{config.distance}_at_{config.orientation}"
    cycles = f"r{config.rounds:02d}"
    return Path(data_dir) / patch / config.basis / cycles


def _read_bits(path: Path, num_bits: int) -> np.ndarray:
    """Read a b8 shot file into a (shots, num_bits) bool array."""
    return stim.read_shot_data_file(
        path=str(path),
        format="b8",
        num_measurements=num_bits,
        bit_pack=False,
    ).astype(np.bool_)


def _load_dem(leaf: Path, pathway: str) -> stim.DetectorErrorModel | None:
    """Load a shipped DEM prior if present, else None."""
    dem_path = leaf / "decoding_results" / pathway / "error_model.dem"
    if not dem_path.exists():
        return None
    return stim.DetectorErrorModel.from_file(dem_path)


def load_run(data_dir: Path, config: WillowConfig) -> WillowRun:
    """Load one Willow configuration from the local dataset directory.

    Returns the Stim circuit, the raw measurement and sweep-bit record arrays,
    and the shipped reference DEMs for that (distance, basis, orientation,
    rounds).
    """
    leaf = config_dir(data_dir, config)
    circuit = stim.Circuit.from_file(leaf / "circuit_ideal.stim")

    measurements = _read_bits(leaf / "measurements.b8", circuit.num_measurements)
    sweep_bits = _read_bits(leaf / "sweep_bits.b8", circuit.num_sweep_bits)

    return WillowRun(
        config=config,
        circuit=circuit,
        measurements=measurements,
        sweep_bits=sweep_bits,
        dem_si1000=_load_dem(leaf, _SI1000_PATHWAY),
        dem_rl=_load_dem(leaf, _RL_PATHWAY),
    )


def load_shipped_detection_data(data_dir: Path, config: WillowConfig) -> DetectionData:
    """Load the dataset's own shipped detection events and observable flips.

    These are the ground truth that our m2d derivation must reproduce (the
    validation gate). Kept separate from :func:`load_run` so the raw-input path
    and the reference path never get conflated.
    """
    leaf = config_dir(data_dir, config)
    circuit = stim.Circuit.from_file(leaf / "circuit_ideal.stim")
    detectors = _read_bits(leaf / "detection_events.b8", circuit.num_detectors)
    observables = _read_bits(leaf / "obs_flips_actual.b8", circuit.num_observables)
    return DetectionData(detectors=detectors, observables=observables)
