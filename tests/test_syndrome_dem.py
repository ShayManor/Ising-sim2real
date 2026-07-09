"""Unit tests for the syndrome-estimated DEM math primitives (arXiv:2606.11496
Appendix B). No network, no real data -- pure numpy on hand-constructed arrays.
"""

from __future__ import annotations

import numpy as np
import stim

from ising_sim2real.ingest.syndrome_dem import (
    bootstrap_mean_std,
    nonempty_subsets,
    parse_dem_events,
    regularize_moment,
    spins_from_detection_events,
)


def test_spins_from_detection_events():
    det = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool)
    spins = spins_from_detection_events(det)
    expected = np.array([[1.0, -1.0, 1.0], [-1.0, -1.0, 1.0], [1.0, 1.0, -1.0]])
    np.testing.assert_array_equal(spins, expected)


def test_nonempty_subsets_size_one():
    assert nonempty_subsets(frozenset({5})) == [frozenset({5})]


def test_nonempty_subsets_size_two():
    result = set(nonempty_subsets(frozenset({1, 2})))
    assert result == {frozenset({1}), frozenset({2}), frozenset({1, 2})}


def test_nonempty_subsets_size_three_count():
    # 2^3 - 1 = 7 nonempty subsets
    result = nonempty_subsets(frozenset({1, 2, 3}))
    assert len(result) == 7
    assert len(set(result)) == 7  # all distinct


def test_bootstrap_mean_std_constant_array():
    # A constant array has zero variance under any resampling.
    values = np.full(1000, -0.5)
    mean, std = bootstrap_mean_std(values, n_bootstrap=50, seed=0)
    assert abs(mean - (-0.5)) < 1e-9
    assert std < 1e-9


def test_bootstrap_mean_std_reproducible():
    rng = np.random.default_rng(1)
    values = rng.normal(loc=0.1, scale=1.0, size=2000)
    a = bootstrap_mean_std(values, n_bootstrap=100, seed=7)
    b = bootstrap_mean_std(values, n_bootstrap=100, seed=7)
    assert a == b  # same seed -> bit-identical


def test_regularize_moment_keeps_nonnegative_untouched():
    values = np.full(1000, 0.3)
    assert regularize_moment(values, raw_moment=0.3) == 0.3


def test_regularize_moment_resolved_negative_kept():
    # Strongly, consistently negative (low variance relative to the mean) ->
    # the sign is "resolved" (SNR >= 0.5) -> keep the negative value.
    rng = np.random.default_rng(2)
    values = rng.normal(loc=-0.9, scale=0.05, size=2000)
    raw = float(values.mean())
    result = regularize_moment(values, raw_moment=raw, seed=3)
    assert result < 0
    assert abs(result - raw) < 1e-6


def test_regularize_moment_unresolved_negative_floored():
    # Small negative mean swamped by noise (SNR < 0.5) -> floored to a
    # positive value (the bootstrap std), not kept negative.
    rng = np.random.default_rng(4)
    values = rng.normal(loc=-0.01, scale=1.0, size=2000)
    raw = float(values.mean())
    assert raw < 0  # sanity check on the constructed fixture
    result = regularize_moment(values, raw_moment=raw, seed=5)
    assert result > 0


def test_parse_dem_events_simple_graphlike():
    dem = stim.DetectorErrorModel("error(0.1) D0 D1\nerror(0.2) D2\n")
    events = parse_dem_events(dem)
    assert set(events) == {frozenset({0, 1}), frozenset({2})}


def test_parse_dem_events_ignores_separator():
    # The `^` is a suggested-decomposition hint (stim.DemTarget.is_separator()),
    # not an event boundary -- this is ONE event with support {D1, D4, D20}.
    dem = stim.DetectorErrorModel("error(0.05) D1 D4 ^ D20\n")
    events = parse_dem_events(dem)
    assert events == [frozenset({1, 4, 20})]


def test_parse_dem_events_ignores_logical_observable_targets():
    # A logical-observable target (L0) must not appear in the detector support.
    dem = stim.DetectorErrorModel("error(0.1) D0 D1 L0\n")
    events = parse_dem_events(dem)
    assert events == [frozenset({0, 1})]


def test_parse_dem_events_skips_non_error_instructions():
    dem = stim.DetectorErrorModel(
        "error(0.1) D0 D1\ndetector(0, 0) D0\nlogical_observable L0\n"
    )
    events = parse_dem_events(dem)
    assert events == [frozenset({0, 1})]
