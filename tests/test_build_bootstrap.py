import numpy as np

from scripts import build_bootstrap as bb


def test_param_covariance_detects_perfect_correlation():
    x = np.linspace(-1, 1, 40)
    m = np.column_stack([x, 2 * x])  # column 1 = 2 * column 0
    cov = bb.param_covariance(m)
    corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    assert abs(corr - 1.0) < 1e-9


def test_centered_pooled_matrix_removes_patch_means():
    params = ["a", "b"]
    draws_by_patch = {
        "d3_q1": [{"a": 10.0, "b": 0.0}, {"a": 12.0, "b": 1.0}],
        "d5_q2": [{"a": 0.0, "b": 5.0}, {"a": 2.0, "b": 7.0}],
    }
    m = bb.centered_pooled_matrix(draws_by_patch, params)
    assert m.shape == (4, 2)
    # each patch block is mean-centered, so every column sums to ~0
    np.testing.assert_allclose(m.sum(axis=0), [0.0, 0.0], atol=1e-9)


def test_ordering_sorts_by_percycle_ler():
    # aggregate() returns {(label, distance_str): mean_per_cycle_ler_float}
    agg = {
        ("mwpm", "3"): 0.005,
        ("tesseract", "3"): 0.003,
        ("bplsd", "3"): 0.004,
    }
    assert bb.ordering(agg, "3") == ["tesseract", "bplsd", "mwpm"]


def test_pair_flip_rate_counts_only_reversed_draws():
    baseline = ["tesseract", "mwpm", "bplsd"]      # mwpm before bplsd
    draws = [
        ["tesseract", "mwpm", "bplsd"],            # same order
        ["tesseract", "bplsd", "mwpm"],            # flipped
        ["bplsd", "tesseract", "mwpm"],            # flipped (bplsd before mwpm)
    ]
    assert bb.pair_flip_rate(draws, baseline, "mwpm", "bplsd") == 2 / 3
