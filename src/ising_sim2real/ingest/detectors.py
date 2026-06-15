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
    sweep_bits: np.ndarray | None = None,
) -> DetectionData:
    """Convert raw measurement records to detection events + observable flips.

    Wraps ``circuit.compile_m2d_converter()`` so detector indexing matches the
    circuit's DEM exactly. The Willow circuits initialize data qubits with
    ``CX sweep[k] q`` gates, so the per-shot ``sweep_bits`` must be supplied for
    the conversion to reproduce the dataset's shipped detection events.

    Args:
        circuit: the (ideal) Stim circuit carrying the DETECTOR/OBSERVABLE
            annotations.
        measurements: shape (shots, circuit.num_measurements), bool-like.
        sweep_bits: shape (shots, circuit.num_sweep_bits), bool-like. Required
            whenever the circuit references sweep bits; may be omitted otherwise.

    Returns:
        DetectionData with detector firings and observable flips, both bool.
    """
    converter = circuit.compile_m2d_converter()
    detectors, observables = converter.convert(
        measurements=np.ascontiguousarray(measurements, dtype=np.bool_),
        sweep_bits=(
            None if sweep_bits is None
            else np.ascontiguousarray(sweep_bits, dtype=np.bool_)
        ),
        separate_observables=True,
    )
    return DetectionData(detectors=detectors, observables=observables)
