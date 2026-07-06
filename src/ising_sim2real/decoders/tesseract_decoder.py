"""Tesseract decoder adapter (``quantumlib/tesseract-decoder``).

A most-likely-error A*-search decoder. Uses the library's sinter-compat wrapper,
which compiles a DEM into a decoder consuming bit-packed detection events and
emitting bit-packed observable predictions. ``det_beam`` bounds the search for
tractability at higher distance/round counts.
"""

from __future__ import annotations

import time

import numpy as np
import stim

from ising_sim2real.decoders.base import DecodeResult


class TesseractDecoder:
    """A*-search most-likely-error decoder, built from a DEM."""

    name = "tesseract"

    def __init__(self, compiled, num_observables: int) -> None:
        self._compiled = compiled
        self._num_observables = num_observables

    @classmethod
    def from_dem(cls, dem: stim.DetectorErrorModel) -> "TesseractDecoder":
        from tesseract_decoder.tesseract_sinter_compat import TesseractSinterDecoder

        tsd = TesseractSinterDecoder(beam_climbing=True, det_beam=20)
        compiled = tsd.compile_decoder_for_dem(dem=dem)
        return cls(compiled, dem.num_observables)

    def decode_batch(self, detectors: np.ndarray) -> DecodeResult:
        packed = np.packbits(
            np.ascontiguousarray(detectors, dtype=np.uint8), axis=1, bitorder="little"
        )
        start = time.perf_counter()
        out = self._compiled.decode_shots_bit_packed(bit_packed_detection_event_data=packed)
        seconds = time.perf_counter() - start
        preds = np.unpackbits(np.asarray(out, dtype=np.uint8), axis=1, bitorder="little")
        preds = preds[:, : self._num_observables].astype(bool)
        return DecodeResult(predictions=preds, seconds=seconds)
