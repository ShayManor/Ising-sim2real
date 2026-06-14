"""Derive detection events and observable flips from measurement records.

Uses Stim's measurement-to-detector conversion (m2d) so detection events are
defined consistently with the shipped DEMs. The output of this stage is what
every decoder and the Ising adapter consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import stim


@dataclass
class DetectionData:
    """Detector firings and the logical observable flips to predict against."""

    detectors: np.ndarray      # shape (shots, num_detectors), bool
    observables: np.ndarray    # shape (shots, num_observables), bool


def load_detection_data(
    run_dir: Path,
    num_detectors: int,
    num_observables: int,
) -> DetectionData:
    """Read pre-computed detection events and observable flips from a Willow run dir.

    The Willow dataset ships detection_events.b8 and obs_flips_actual.b8 as
    separate files (observables are NOT appended to the detector file). The
    baseline reads these directly; no m2d conversion is needed.
    """
    detectors = stim.read_shot_data_file(
        path=str(run_dir / "detection_events.b8"),
        format="b8",
        num_measurements=num_detectors,
    )
    observables = stim.read_shot_data_file(
        path=str(run_dir / "obs_flips_actual.b8"),
        format="b8",
        num_measurements=num_observables,
    )
    return DetectionData(
        detectors=np.asarray(detectors, dtype=bool),
        observables=np.asarray(observables, dtype=bool).reshape(-1, num_observables),
    )


def measurements_to_detectors(
    circuit: stim.Circuit,
    measurements: np.ndarray,
) -> DetectionData:
    """Convert raw measurement records to detection events + observable flips.

    Wraps circuit.compile_m2d_converter() so detector indexing matches the
    circuit's DEM exactly.

    NOTE (Phase 3): the Ising adapter will use this path. Before wiring it,
    verify that Stim's emission order for the Willow circuits matches the
    [X_r0, Z_r0, ..., X_rT, Z_rT] interleaving that dets_to_predecoder_inputs
    expects in its (B, 2*T*half) flat layout. The baseline does NOT call this.
    """
    converter = circuit.compile_m2d_converter()
    dets_and_obs = converter.convert(
        measurements=np.asarray(measurements, dtype=bool),
        append_observables=True,
    )
    num_obs = circuit.num_observables
    return DetectionData(
        detectors=np.asarray(dets_and_obs[:, :-num_obs], dtype=bool),
        observables=np.asarray(dets_and_obs[:, -num_obs:], dtype=bool),
    )
