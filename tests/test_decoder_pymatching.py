"""Tests for the PyMatching decoder adapter."""

from __future__ import annotations

import numpy as np
import stim

from ising_sim2real.decoders.base import Decoder, DecodeResult
from ising_sim2real.decoders.pymatching_decoder import PyMatchingDecoder


def _toy_dem() -> stim.DetectorErrorModel:
    # Two detectors, one observable: a single error flips det0+det1+obs.
    return stim.DetectorErrorModel("""
        error(0.1) D0 D1 L0
        detector D0
        detector D1
    """)


def test_conforms_to_decoder_protocol() -> None:
    dec = PyMatchingDecoder.from_dem(_toy_dem())
    assert isinstance(dec, Decoder)
    assert isinstance(dec.name, str) and dec.name


def test_decode_batch_returns_decode_result_shapes() -> None:
    dec = PyMatchingDecoder.from_dem(_toy_dem())
    detectors = np.array([[1, 1], [0, 0]], dtype=bool)
    result = dec.decode_batch(detectors)

    assert isinstance(result, DecodeResult)
    assert result.predictions.shape == (2, 1)
    assert result.predictions.dtype == bool
    assert result.seconds >= 0.0


def test_zero_syndrome_predicts_no_flip() -> None:
    dec = PyMatchingDecoder.from_dem(_toy_dem())
    detectors = np.zeros((5, 2), dtype=bool)
    result = dec.decode_batch(detectors)
    assert not result.predictions.any()


def test_single_edge_error_is_matched_to_its_observable() -> None:
    # Firing both detectors is explained by the single error mechanism, which
    # flips L0 -> the decoder must predict an observable flip.
    dec = PyMatchingDecoder.from_dem(_toy_dem())
    result = dec.decode_batch(np.array([[1, 1]], dtype=bool))
    assert bool(result.predictions[0, 0]) is True
