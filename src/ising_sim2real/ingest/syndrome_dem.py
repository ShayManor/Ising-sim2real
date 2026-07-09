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
import stim

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


def parse_dem_events(dem: "stim.DetectorErrorModel") -> list[Support]:
    """One detector-support set per ``error`` instruction in the flattened DEM.

    ``^`` separators (``stim.DemTarget.is_separator()``) are a suggested
    decomposition hint for matching decoders, not an event boundary -- an
    instruction like ``error(p) D1 D4 ^ D20`` is a SINGLE event whose true
    support is the union of every detector target on both sides, ``{1,4,20}``.
    Logical-observable targets (``is_logical_observable_id()``) are excluded
    from the support set; they don't affect probability estimation (the
    logical-observable assignment is inherited unchanged from the reference
    DEM, per the paper's Appendix A).
    """
    events: list[Support] = []
    for instr in dem.flattened():
        if instr.type != "error":
            continue
        targets = instr.targets_copy()
        support = frozenset(t.val for t in targets if t.is_relative_detector_id())
        events.append(support)
    return events


def estimate_dem_from_syndromes(
    dem_si1000: "stim.DetectorErrorModel",
    detection_events: np.ndarray,
) -> "stim.DetectorErrorModel":
    """Build a DEM with the same graph as ``dem_si1000`` but probabilities
    estimated from real ``detection_events`` (paper Eqs. B5-B12, applied
    uniformly to every event size 1-4 -- this subsumes the simpler no-hyperedge
    formulas B3/B4 as the special case where an event has no strict supersets
    in this DEM, so only one code path is needed).

    ``detection_events`` is a ``(shots, num_detectors)`` bool array of REAL
    Willow detection events for the same config ``dem_si1000`` was shipped for.
    """
    events = parse_dem_events(dem_si1000)

    # Guard the ambiguous case from the paper's Appendix A: two events sharing
    # a detector support but flipping different logical observables can't be
    # told apart from detector moments alone (which never see logical
    # outcomes). Expected to never trigger on this project's shipped SI1000
    # DEMs (DEM construction already merges same-signature faults), but fail
    # loudly rather than silently pick one arbitrarily if it ever does.
    seen: dict[Support, Support] = {}
    for instr in dem_si1000.flattened():
        if instr.type != "error":
            continue
        targets = instr.targets_copy()
        support = frozenset(t.val for t in targets if t.is_relative_detector_id())
        logicals = frozenset(t.val for t in targets if t.is_logical_observable_id())
        if support in seen and seen[support] != logicals:
            raise NotImplementedError(
                f"support set {sorted(support)} appears with two different "
                f"logical-observable assignments ({sorted(seen[support])} vs "
                f"{sorted(logicals)}) -- ambiguous under detector moments alone "
                "(arXiv:2606.11496 Appendix A); not handled."
            )
        seen[support] = logicals

    spins = spins_from_detection_events(detection_events)

    needed: set[Support] = set()
    for support in events:
        needed.update(nonempty_subsets(support))

    moments: dict[Support, float] = {}
    for subset in needed:
        cols = [spins[:, i] for i in sorted(subset)]
        products = cols[0].copy()
        for col in cols[1:]:
            products = products * col
        raw = float(products.mean())
        moments[subset] = regularize_moment(products, raw) if raw < 0 else raw

    supports_by_size: dict[int, list[Support]] = {1: [], 2: [], 3: [], 4: []}
    for support in events:
        supports_by_size[len(support)].append(support)

    q: dict[Support, float] = {}
    p: dict[Support, float] = {}
    for size in (4, 3, 2, 1):
        for support in supports_by_size[size]:
            r_e = 1.0
            for subset in nonempty_subsets(support):
                sign = 1 if (len(subset) % 2 == 1) else -1
                r_e *= moments[subset] ** sign

            if r_e < 0:
                # Fractional root of a negative number is unphysical -- this
                # event's rate is not resolved by the data; treat it as
                # probability 0 (q=1 is the neutral element for the correction
                # product used by smaller subsets below, per Eq. B8).
                p[support] = 0.0
                q[support] = 1.0
                continue

            correction = 1.0
            for other_size in range(size + 1, 5):
                for other_support in supports_by_size[other_size]:
                    if support < other_support:  # strict subset
                        correction *= q[other_support] ** -1

            q_e = (r_e ** (1.0 / (2 ** (size - 1)))) * correction
            p_e = (1.0 - q_e) / 2.0

            if not (0.0 <= p_e <= 1.0):
                p_e = 0.0
                q_e = 1.0

            p[support] = p_e
            q[support] = q_e

    out = stim.DetectorErrorModel()
    for instr in dem_si1000.flattened():
        if instr.type == "error":
            targets = instr.targets_copy()
            support = frozenset(t.val for t in targets if t.is_relative_detector_id())
            out.append("error", p[support], targets)
        else:
            out.append(instr)
    return out
