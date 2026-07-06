"""Belief-matching decoder adapter (BP-informed correlated matching).

Belief propagation reweights the matching graph per shot before PyMatching runs --
the "correlated matching" family the Willow paper's Libra decoder belongs to, and
the strong classical baseline any decoder claiming a hardware win must beat. Wraps
``beliefmatching.BeliefMatching`` behind the shared :class:`Decoder` contract.
"""

from __future__ import annotations

import time

import numpy as np
import stim

from ising_sim2real.decoders.base import DecodeResult


class BeliefMatchingDecoder:
    """Belief propagation + minimum-weight perfect matching, built from a DEM."""

    name = "beliefmatching"

    def __init__(self, matching) -> None:
        self._matching = matching

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> "BeliefMatchingDecoder":
        from beliefmatching import BeliefMatching

        return cls(BeliefMatching.from_detector_error_model(dem))

    def decode_batch(self, detectors: np.ndarray) -> DecodeResult:
        detectors = np.ascontiguousarray(detectors, dtype=np.uint8)
        start = time.perf_counter()
        predictions = self._matching.decode_batch(detectors)
        seconds = time.perf_counter() - start
        return DecodeResult(predictions=np.asarray(predictions).astype(bool), seconds=seconds)
