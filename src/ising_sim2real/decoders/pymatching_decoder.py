"""PyMatching (MWPM) decoder adapter for the panel.

The first concrete member of the decoder panel and the classical baseline the
ingest pipeline is validated against (CLAUDE.md method step 2). Wraps
``pymatching.Matching`` behind the shared :class:`Decoder` contract so the
harness can swap it for any other decoder.
"""

from __future__ import annotations

import time

import numpy as np
import pymatching
import stim

from ising_sim2real.decoders.base import DecodeResult


class PyMatchingDecoder:
    """Minimum-weight perfect matching decoder built from a DEM."""

    name = "pymatching"

    def __init__(self, matching: pymatching.Matching) -> None:
        self._matching = matching

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> "PyMatchingDecoder":
        return cls(pymatching.Matching.from_detector_error_model(dem))

    def decode_batch(self, detectors: np.ndarray) -> DecodeResult:
        detectors = np.ascontiguousarray(detectors, dtype=np.uint8)
        start = time.perf_counter()
        predictions = self._matching.decode_batch(detectors)
        seconds = time.perf_counter() - start
        return DecodeResult(predictions=predictions.astype(bool), seconds=seconds)
