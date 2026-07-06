"""BP+OSD and BP+LSD decoder adapters (via the ``ldpc`` library).

Both consume a stim DEM, converted to (check, observables, priors) matrices with
beliefmatching's helper, and expose the shared :class:`Decoder` contract. BP+OSD
and BP+LSD differ only in the ldpc backend class, so a common base does the
per-shot decode and each subclass names and builds its backend.
"""

from __future__ import annotations

import time

import numpy as np
import stim

from ising_sim2real.decoders.base import DecodeResult


def _dem_matrices(dem: stim.DetectorErrorModel):
    """Return (check_matrix csr, observables_matrix csr, error priors) for a DEM."""
    from beliefmatching import detector_error_model_to_check_matrices

    m = detector_error_model_to_check_matrices(dem)
    return m.check_matrix.tocsr(), m.observables_matrix.tocsr(), np.asarray(m.priors, dtype=float)


class _BpDecoder:
    """Shared per-shot decode: BP(+post-processing) error guess -> observable flips."""

    name = "bp"

    def __init__(self, decoder, observables) -> None:
        self._decoder = decoder
        self._obs = observables  # csr (num_observables, num_errors)

    def decode_batch(self, detectors: np.ndarray) -> DecodeResult:
        detectors = np.ascontiguousarray(detectors, dtype=np.uint8)
        preds = np.empty((detectors.shape[0], self._obs.shape[0]), dtype=bool)
        start = time.perf_counter()
        for i, syndrome in enumerate(detectors):
            error = self._decoder.decode(syndrome)
            preds[i] = np.asarray(self._obs @ error).ravel() % 2
        seconds = time.perf_counter() - start
        return DecodeResult(predictions=preds, seconds=seconds)


class BpOsdDecoder(_BpDecoder):
    """Belief propagation with ordered-statistics decoding post-processing."""

    name = "bposd"

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> "BpOsdDecoder":
        from ldpc import BpOsdDecoder as _Osd

        check, obs, priors = _dem_matrices(dem)
        decoder = _Osd(check, error_channel=list(priors), max_iter=30,
                       bp_method="ms", osd_method="osd_cs", osd_order=7)
        return cls(decoder, obs)


class BpLsdDecoder(_BpDecoder):
    """Belief propagation with localized-statistics decoding post-processing."""

    name = "bplsd"

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> "BpLsdDecoder":
        from ldpc import BpLsdDecoder as _Lsd

        check, obs, priors = _dem_matrices(dem)
        decoder = _Lsd(check, error_channel=list(priors), max_iter=30,
                       bp_method="ms", lsd_order=0)
        return cls(decoder, obs)
