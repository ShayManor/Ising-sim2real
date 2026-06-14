"""Common decoder interface for the panel.

SCAFFOLD ONLY -- the protocol is defined; concrete adapters are not implemented.

Every decoder consumes detection events and predicts observable flips, exposing
the same surface so the harness can swap them and record LER per cycle and decode
latency uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
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
