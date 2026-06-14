"""Load Google Willow below-threshold circuits and pre-computed detection events.

Dataset: Zenodo 10.5281/zenodo.13273331, rotated surface code at d = 3, 5, 7,
X and Z memory. Each configuration ships a Stim circuit, pre-computed detection
events, observable flips, and a fitted SI1000 detector error model.

Directory layout (one downloaded config shown):
    data/google_105Q_surface_code_d3_d5_d7/
      d{D}_at_{orientation}/        e.g. d3_at_q4_5
        {basis}/                    X or Z
          r{R}/
            metadata.json
            circuit_noisy_si1000.stim
            detection_events.b8      # (shots, num_detectors) bit-packed, bool
            obs_flips_actual.b8      # (shots, num_observables) bit-packed, bool
            measurements.b8          # raw meas bits; not needed for the baseline
            sweep_bits.b8
            decoding_results/
              correlated_matching_decoder_with_si1000_prior/
                error_model.dem      # fitted SI1000 DEM — use this for PyMatching
                obs_flips_predicted.b8
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import stim

from ising_sim2real.ingest.detectors import DetectionData, load_detection_data


@dataclass(frozen=True)
class WillowConfig:
    """One Willow experimental configuration."""

    distance: int        # code distance d (3, 5, or 7)
    basis: str           # "X" or "Z" memory
    rounds: int          # number of QEC rounds T
    orientation: str     # chip-placement label, e.g. "q4_5" (from "d3_at_q4_5")


@dataclass
class WillowRun:
    """A loaded configuration: circuit, pre-computed detection data, and reference DEM."""

    config: WillowConfig
    circuit: stim.Circuit
    detection_data: DetectionData               # pre-computed from dataset files
    dem_si1000: stim.DetectorErrorModel | None  # bundled fitted DEM; use for PyMatching
    dem_rl: stim.DetectorErrorModel | None      # RL-optimised DEM; None in current dataset


def _find_run_dir(data_dir: Path, config: WillowConfig) -> Path:
    run_dir = (
        data_dir
        / f"d{config.distance}_at_{config.orientation}"
        / config.basis
        / f"r{config.rounds}"
    )
    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"Willow run directory not found: {run_dir}\n"
            f"  config={config}\n"
            f"  Download from Zenodo DOI 10.5281/zenodo.13273331"
        )
    return run_dir


def load_run(data_dir: Path, config: WillowConfig) -> WillowRun:
    """Load one Willow configuration from the local dataset directory.

    Returns the Stim circuit, pre-computed detection events, observable flips,
    and the bundled SI1000 DEM for that (distance, basis, orientation, rounds).
    """
    run_dir = _find_run_dir(data_dir, config)
    circuit = stim.Circuit.from_file(run_dir / "circuit_noisy_si1000.stim")

    detection_data = load_detection_data(
        run_dir,
        num_detectors=circuit.num_detectors,
        num_observables=circuit.num_observables,
    )

    dem_path = (
        run_dir
        / "decoding_results"
        / "correlated_matching_decoder_with_si1000_prior"
        / "error_model.dem"
    )
    dem_si1000 = (
        stim.DetectorErrorModel.from_file(dem_path) if dem_path.is_file() else None
    )

    return WillowRun(
        config=config,
        circuit=circuit,
        detection_data=detection_data,
        dem_si1000=dem_si1000,
        dem_rl=None,
    )
