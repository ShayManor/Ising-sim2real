"""Map Willow detection events into the Ising model's lattice layout.

SCAFFOLD ONLY -- interfaces are defined; bodies are not implemented yet.

The pre-decoder consumes a ``(B, 4, T, D, D)`` tensor: B shots, 4 detector
channels, T rounds, on a D x D lattice. This module turns the flat detection
events produced by ``ingest.detectors`` into that layout and back. Per the
method, validate this adapter by matching the classical MWPM baseline before
trusting any Ising number (method step 4).
"""

from __future__ import annotations

import numpy as np
import torch


def detection_events_to_lattice(
    detectors: np.ndarray,
    distance: int,
    rounds: int,
) -> torch.Tensor:
    """Reshape flat detection events into the ``(B, 4, T, D, D)`` model input.

    Args:
        detectors: shape (shots, num_detectors), bool.
        distance: code distance D.
        rounds: number of rounds T.

    Returns:
        Float tensor of shape (shots, 4, rounds, distance, distance).
    """
    raise NotImplementedError("Ising adapter not implemented yet (method step 4).")


def lattice_to_detection_events(
    lattice: torch.Tensor,
    num_detectors: int,
) -> np.ndarray:
    """Inverse of :func:`detection_events_to_lattice` for the cleaned syndrome."""
    raise NotImplementedError("Ising adapter not implemented yet (method step 4).")
