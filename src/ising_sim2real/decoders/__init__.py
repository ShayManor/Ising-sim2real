"""Decoder panel adapters (SCAFFOLD ONLY).

A common interface (``base.Decoder``) that every decoder in the panel implements
so the harness can run them interchangeably: PyMatching (correlated and
uncorrelated MWPM), BP+OSD / BP+LSD via ldpc, Tesseract, the NVIDIA Ising
pre-decoder + PyMatching, and optionally cudaq-qec / Fusion Blossom.
"""
