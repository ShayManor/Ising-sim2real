"""Logical-error-rate metrics shared across the decoder panel.

Every decoder is scored by the same two numbers: the fraction of shots whose
logical observable was mispredicted (the logical error rate), and that rate
expressed per QEC cycle so configurations with different round counts compare on
equal footing (the quantity Google reports for the Willow memory experiments).
"""

from __future__ import annotations

import numpy as np


def logical_error_rate(predicted: np.ndarray, actual: np.ndarray) -> float:
    """Fraction of shots with at least one mispredicted observable.

    Args:
        predicted: shape (shots, num_observables), bool-like.
        actual: shape (shots, num_observables), bool-like.

    Returns:
        The logical error rate over the batch in [0, 1].
    """
    predicted = np.asarray(predicted, dtype=bool)
    actual = np.asarray(actual, dtype=bool)
    if predicted.shape != actual.shape:
        raise ValueError(
            f"shape mismatch: predicted {predicted.shape} vs actual {actual.shape}"
        )
    mismatched = np.any(predicted != actual, axis=tuple(range(1, predicted.ndim)))
    return float(np.mean(mismatched))


def logical_error_per_cycle(error_rate: float, rounds: int) -> float:
    """Convert a total logical error rate into a per-cycle rate.

    Treats each of ``rounds`` cycles as an independent flip channel: the logical
    fidelity ``1 - 2*eps`` accumulates multiplicatively, so
    ``1 - 2*eps_total = (1 - 2*eps_cycle) ** rounds``. Inverting,

        eps_cycle = (1 - (1 - 2*eps_total) ** (1/rounds)) / 2.

    Args:
        error_rate: total logical error rate over all rounds, in [0, 0.5].
        rounds: number of QEC cycles (>= 1).

    Returns:
        The per-cycle logical error rate.
    """
    if rounds < 1:
        raise ValueError(f"rounds must be >= 1, got {rounds}")
    fidelity = 1.0 - 2.0 * error_rate
    if fidelity <= 0.0:
        # At or past the totally-depolarized point a per-cycle rate is undefined;
        # clamp to 0.5 rather than take a root of a non-positive number.
        return 0.5
    return float((1.0 - fidelity ** (1.0 / rounds)) / 2.0)
