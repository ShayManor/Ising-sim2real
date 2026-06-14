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


@dataclass(frozen=True)
class WillowConfig:
    """One Willow experimental configuration."""

    distance: int          # code distance d (3, 5, or 7)
    basis: str             # "X" or "Z" memory
    rounds: int            # number of QEC rounds / cycles
    orientation: str       # code orientation label as shipped by the dataset


@dataclass
class WillowRun:
    """A loaded configuration: circuit, raw measurements, and reference DEMs."""

    config: WillowConfig
    circuit: stim.Circuit
    measurements: np.ndarray          # shape (shots, num_measurements), bool
    dem_si1000: stim.DetectorErrorModel | None
    dem_rl: stim.DetectorErrorModel | None


def load_run(data_dir: Path, config: WillowConfig) -> WillowRun:
    """Load one Willow configuration from the local dataset directory.

    Returns the Stim circuit, the raw measurement record array, and the shipped
    reference DEMs for that (distance, basis, orientation, rounds).
    """
    raise NotImplementedError("Ingest pipeline not implemented yet (method step 1).")
