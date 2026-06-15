"""Tests for logical-error-rate metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ising_sim2real.metrics import logical_error_per_cycle, logical_error_rate


def test_logical_error_rate_counts_mismatched_shots() -> None:
    predicted = np.array([[0], [1], [0], [1]], dtype=bool)
    actual = np.array([[0], [0], [0], [0]], dtype=bool)
    # shots 1 and 3 are wrong -> 2/4.
    assert logical_error_rate(predicted, actual) == 0.5


def test_logical_error_rate_is_any_mismatch_across_observables() -> None:
    # A shot is a logical error if ANY observable is mispredicted.
    predicted = np.array([[0, 0], [1, 0], [0, 1]], dtype=bool)
    actual = np.array([[0, 0], [0, 0], [0, 1]], dtype=bool)
    # shot0 correct, shot1 wrong (obs0), shot2 correct -> 1/3.
    assert logical_error_rate(predicted, actual) == pytest.approx(1 / 3)


def test_logical_error_per_cycle_inverts_accumulation() -> None:
    # If each cycle independently flips with prob p, the total over r cycles is
    # eps = (1 - (1-2p)^r)/2. The helper must recover p from eps.
    p = 0.007
    rounds = 13
    eps_total = (1 - (1 - 2 * p) ** rounds) / 2
    assert logical_error_per_cycle(eps_total, rounds) == pytest.approx(p, abs=1e-9)


def test_logical_error_per_cycle_single_round_is_identity() -> None:
    assert logical_error_per_cycle(0.042, rounds=1) == pytest.approx(0.042)


def test_logical_error_per_cycle_is_below_total() -> None:
    # Spread over many cycles, the per-cycle rate is smaller than the total.
    assert logical_error_per_cycle(0.3, rounds=50) < 0.3
