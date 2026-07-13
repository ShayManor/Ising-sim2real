# tests/test_bootstrap_resampling.py
import numpy as np
import pytest

from ising_sim2real.ingest.param_fit import resample_shots


def _events():
    # 6 shots x 4 detectors, distinct rows so resampling is observable
    return np.array(
        [[0, 0, 0, 0],
         [1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1],
         [1, 1, 1, 1]], dtype=bool)


def test_resample_is_deterministic_for_same_seed():
    ev = _events()
    a = resample_shots(ev, [7, 3, 30, 0])
    b = resample_shots(ev, [7, 3, 30, 0])
    np.testing.assert_array_equal(a, b)


def test_resample_differs_across_draws():
    ev = _events()
    a = resample_shots(ev, [0, 3, 30, 0])
    b = resample_shots(ev, [1, 3, 30, 0])
    assert not np.array_equal(a, b)


def test_resample_preserves_shape_and_dtype():
    ev = _events()
    out = resample_shots(ev, [0, 3, 30, 0])
    assert out.shape == ev.shape
    assert out.dtype == ev.dtype


def test_resample_rows_are_drawn_from_input():
    ev = _events()
    out = resample_shots(ev, [2, 5, 10, 1])
    input_rows = {tuple(r) for r in ev.tolist()}
    for row in out.tolist():
        assert tuple(row) in input_rows


def test_resample_changes_mean_moment():
    # a lopsided column so a with-replacement resample almost surely shifts its mean
    ev = np.zeros((50, 3), dtype=bool)
    ev[:5, 0] = True  # column 0 fires in 5/50 shots
    base = ev[:, 0].mean()
    shifted = [resample_shots(ev, [d, 3, 30, 0])[:, 0].mean() for d in range(20)]
    assert any(m != base for m in shifted)
