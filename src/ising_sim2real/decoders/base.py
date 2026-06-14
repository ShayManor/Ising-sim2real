"""Common decoder interface for the panel, plus the PyMatching baseline decoder.

Every decoder consumes detection events and predicts observable flips, exposing
the same surface so the harness can swap them and record LER per cycle and decode
latency uniformly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import pymatching
import stim


@dataclass
class DecodeResult:
    """Predicted observable flips plus the wall-clock cost of decoding."""

    predictions: np.ndarray    # shape (shots, num_observables), bool
    seconds: float             # total decode time for the batch


@runtime_checkable
class Decoder(Protocol):
    """Minimal contract shared by every decoder in the panel."""

    name: str

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> "Decoder":
        """Construct a decoder from a detector error model."""
        ...

    def decode_batch(self, detectors: np.ndarray) -> DecodeResult:
        """Predict observable flips for a batch of detection-event shots."""
        ...


class PyMatchingDecoder:
    """MWPM decoder backed by PyMatching, implementing the Decoder protocol.

    Constructed from a detector error model; works with either the bundled
    Willow DEM or a DEM derived from a Stim circuit via detector_error_model().
    """

    name: str = "pymatching"

    def __init__(self, matcher: pymatching.Matching) -> None:
        self._matcher = matcher

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> "PyMatchingDecoder":
        return cls(pymatching.Matching.from_detector_error_model(dem))

    def decode_batch(self, detectors: np.ndarray) -> DecodeResult:
        t0 = time.perf_counter()
        preds = self._matcher.decode_batch(detectors.astype(np.uint8))
        return DecodeResult(
            predictions=preds.astype(bool),
            seconds=time.perf_counter() - t0,
        )
