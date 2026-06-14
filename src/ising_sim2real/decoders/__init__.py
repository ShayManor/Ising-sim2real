"""Decoder panel adapters.

A common interface (``base.Decoder``) that every decoder in the panel implements
so the harness can run them interchangeably: PyMatching (correlated and
uncorrelated MWPM), BP+OSD / BP+LSD via ldpc, Tesseract, the NVIDIA Ising
pre-decoder + PyMatching, and optionally cudaq-qec / Fusion Blossom.

Implemented:
    PyMatchingDecoder  — MWPM via PyMatching (Phase 2 baseline)
"""

from ising_sim2real.decoders.base import DecodeResult, Decoder, PyMatchingDecoder

__all__ = ["Decoder", "DecodeResult", "PyMatchingDecoder"]
