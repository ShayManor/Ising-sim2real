"""Derive detection events and observable flips from measurement records.

SCAFFOLD ONLY -- interfaces are defined; bodies are not implemented yet.

Uses Stim's measurement-to-detector conversion (m2d) so detection events are
defined consistently with the shipped DEMs. The output of this stage is what
every decoder and the Ising adapter consume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim


@dataclass
class DetectionData:
    """Detector firings and the logical observable flips to predict against."""

    detectors: np.ndarray      # shape (shots, num_detectors), bool
    observables: np.ndarray    # shape (shots, num_observables), bool


def measurements_to_detectors(
    circuit: stim.Circuit,
    measurements: np.ndarray,
) -> DetectionData:
    """Convert raw measurement records to detection events + observable flips.

    Wraps ``circuit.compile_m2d_converter()`` so detector indexing matches the
    circuit's DEM exactly.
    """
    raise NotImplementedError("Ingest pipeline not implemented yet (method step 1).")
