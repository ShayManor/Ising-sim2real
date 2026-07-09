"""Estimate a DEM's edge probabilities directly from real detection-event
statistics -- the ``syndrome`` rung of the noise-fidelity ladder.

Implements the method of Evangelia Takou, Cesar Benito, Arian Vezvaee, Daniel A.
Lidar, Kenneth R. Brown, "Logical error estimation from syndrome data of
surface-code experiments," arXiv:2606.11496v2 (Appendix B), evaluated on this
exact Willow dataset in the source paper. The graph/hyperedge structure (which
detector sets form valid DEM events, and which logical observable each flips) is
inherited unchanged from a reference DEM (here, the shipped SI1000 DEM); only the
per-event probability is re-estimated from real detection-event correlations.
No ground-truth logical outcomes are used anywhere in this module.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

Support = frozenset[int]


def spins_from_detection_events(detection_events: np.ndarray) -> np.ndarray:
    """Convert a ``(shots, num_detectors)`` bool array to spins in ``{-1,+1}``.

    ``s_i = (-1)^{v_i} = 1 - 2*v_i``, so a detector that DIDN'T fire (``v_i=0``)
    has spin ``+1``, and one that fired has spin ``-1`` (paper Eq. B1).
    """
    return 1.0 - 2.0 * detection_events.astype(np.float64)


def nonempty_subsets(support: Support) -> list[Support]:
    """All nonempty subsets of ``support``, as a list of frozensets.

    For ``|support| <= 4`` (this dataset's max event size) this is at most 15
    subsets -- cheap to enumerate on demand rather than precompute.
    """
    items = sorted(support)
    out: list[Support] = []
    for r in range(1, len(items) + 1):
        for combo in combinations(items, r):
            out.append(frozenset(combo))
    return out


def bootstrap_mean_std(
    values: np.ndarray, n_bootstrap: int = 100, seed: int = 0
) -> tuple[float, float]:
    """Bootstrap-resample ``values`` (one scalar per shot) ``n_bootstrap`` times
    (sampling shot indices with replacement); return (mean, std) of the
    resulting distribution of resample means.
    """
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    return float(means.mean()), float(means.std())


def regularize_moment(
    spin_products: np.ndarray,
    raw_moment: float,
    n_bootstrap: int = 100,
    seed: int = 0,
    snr_threshold: float = 0.5,
) -> float:
    """Apply the paper's Appendix B sign-resolution procedure to one moment.

    ``spin_products`` is the ``(shots,)`` array of per-shot spin products whose
    mean is ``raw_moment`` (i.e. the array that was averaged to produce it).
    Non-negative moments are returned unchanged -- regularization only concerns
    negative ("unresolved sign") moments. For a negative moment, bootstrap the
    sign: if the signal-to-noise ratio is below ``snr_threshold``, the negative
    sign isn't statistically resolved by the data, so replace it with a small
    positive floor (the bootstrap standard deviation) rather than propagate
    noise as a fake negative correlation. Otherwise the negative value is real
    signal and is kept.
    """
    if raw_moment >= 0:
        return raw_moment
    s_bar, sigma_s = bootstrap_mean_std(spin_products, n_bootstrap=n_bootstrap, seed=seed)
    if s_bar < 0 and sigma_s > 0 and abs(s_bar) / sigma_s < snr_threshold:
        return sigma_s
    return raw_moment
