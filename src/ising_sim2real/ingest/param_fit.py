"""Fit the 25-param circuit-level noise model to real Willow detection-event
statistics -- the ``fit`` rung of the noise-fidelity ladder.

Forward direction only in this module so far: given a candidate circuit's DEM,
compute the detector-fraction and pairwise-correlation MOMENTS it predicts,
analytically (inclusion-exclusion over each detector's covering fault
mechanisms -- the mathematical dual of ``syndrome_dem.py``'s inversion, derived
from first principles in this module's docstrings, not a paper transcription).
No sampling anywhere in this computation -- a DEM fully determines these moments
in closed form.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations

import numpy as np
import stim
from scipy.optimize import minimize

from ising_sim2real.paths import ISING_CODE


def resample_shots(detection_events: "np.ndarray", seed) -> "np.ndarray":
    """With-replacement bootstrap resample of the SHOT rows of a
    ``(shots, num_detectors)`` detection-events array. ``seed`` is any
    ``np.random.default_rng``-acceptable seed (the joint-bootstrap caller passes
    ``[draw, distance, rounds, basis_int, crc32(orientation)]`` so each config's resample is
    deterministic and independent -- across rounds, basis, AND patch). Returns a new array of identical shape and
    dtype -- the outer bootstrap draws that ``fit_noise_model`` then fits to."""
    rng = np.random.default_rng(seed)
    n_shots = detection_events.shape[0]
    idx = rng.integers(0, n_shots, size=n_shots)
    return detection_events[idx]


# Forward-difference step for the parallel finite-difference gradient built by
# `fit_noise_model` below -- matches `scipy.optimize.minimize`'s own default
# `eps` option for L-BFGS-B's numerical jacobian (used when no `jac=` is
# supplied), so switching to a custom parallel gradient does not itself change
# the underlying finite-difference math, only who computes each probe point.
_FD_EPS = 1e-8


def pairs_from_dem(dem: "stim.DetectorErrorModel") -> list[tuple[int, int]]:
    """Every detector pair that co-occurs in at least one DEM event's support --
    the "window" of pairs worth a pij prediction (pairs that never share an
    event carry no physical correlation this DEM's own graph structure
    represents, so predicting/fitting them would be noise, not signal).
    """
    pairs: set[tuple[int, int]] = set()
    for instr in dem.flattened():
        if instr.type != "error":
            continue
        support = sorted(t.val for t in instr.targets_copy() if t.is_relative_detector_id())
        for i, j in combinations(support, 2):
            pairs.add((i, j))
    return sorted(pairs)


def predicted_moments(
    dem: "stim.DetectorErrorModel", pairs: list[tuple[int, int]]
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """Analytic ``<s_i>`` and ``<s_i s_j>`` moments predicted by ``dem``'s error
    probabilities (see module docstring for the derivation). ``pairs`` bounds
    which detector pairs get a pij prediction -- pass ``pairs_from_dem(dem)`` for
    the DEM's own graph-adjacent pairs, the intended usage.
    """
    events: list[tuple[frozenset[int], float]] = []
    for instr in dem.flattened():
        if instr.type != "error":
            continue
        support = frozenset(t.val for t in instr.targets_copy() if t.is_relative_detector_id())
        p = instr.args_copy()[0]
        events.append((support, p))

    covering: dict[int, list[tuple[frozenset[int], float]]] = {}
    for support, p in events:
        for d in support:
            covering.setdefault(d, []).append((support, p))

    singles: dict[int, float] = {}
    for d, evs in covering.items():
        prod = 1.0
        for _support, p in evs:
            prod *= (1.0 - 2.0 * p)
        singles[d] = prod

    pijs: dict[tuple[int, int], float] = {}
    for i, j in pairs:
        prod = 1.0
        for support, p in covering.get(i, []):
            if j not in support:
                prod *= (1.0 - 2.0 * p)
        for support, p in covering.get(j, []):
            if i not in support:
                prod *= (1.0 - 2.0 * p)
        pijs[(i, j)] = prod

    return singles, pijs


def real_moments(
    detection_events: "np.ndarray", pairs: list[tuple[int, int]]
) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    """Empirical ``<s_i>`` and ``<s_i s_j>`` moments from real ``(shots,
    num_detectors)`` detection events -- reuses ``syndrome_dem.py``'s spin
    conversion directly, the same statistic that module's inversion consumes.
    """
    from ising_sim2real.ingest.syndrome_dem import spins_from_detection_events

    spins = spins_from_detection_events(detection_events)
    singles = {d: float(spins[:, d].mean()) for d in range(spins.shape[1])}
    pijs = {(i, j): float((spins[:, i] * spins[:, j]).mean()) for i, j in pairs}
    return singles, pijs


def bootstrap_mean_std_batch(
    values_matrix: "np.ndarray", n_bootstrap: int = 50, seed: int = 0
) -> tuple["np.ndarray", "np.ndarray"]:
    """Vectorized batch analogue of ``syndrome_dem.bootstrap_mean_std``: computes
    the SAME statistic -- mean and std, across ``n_bootstrap`` resamples of
    ``n_shots``-sized draws-with-replacement, of each column's resampled mean
    -- for EVERY column of ``values_matrix`` (shape ``(n_shots, n_items)``) at
    once, instead of one independent Python-level call per column.

    For each of the ``n_bootstrap`` resamples, ONE shared set of resampled
    shot indices is drawn and every column's resampled mean is computed
    simultaneously as a single matrix-vector product against
    ``values_matrix`` (BLAS-backed, one sequential pass over the data) rather
    than by gathering (``values_matrix[idx]``) a full ``(n_shots, n_items)``
    copy per resample: a resampled mean is
    ``mean(values_matrix[idx], axis=0) == (counts @ values_matrix) / n_shots``
    where ``counts[k]`` is how many times shot ``k`` was drawn in this
    resample (``np.bincount(idx)``) -- algebraically identical, but the gather
    form was measured to be SLOWER than the original per-item Python loop for
    wide matrices (a random-row gather into an >100MB working array thrashes
    cache far worse than gathering into the original tiny single-column
    array; a dense matvec is a single cache-friendly streaming pass and lets
    BLAS vectorize across all columns at once). This is what eliminates the
    O(n_items) Python loop -- ``bootstrap_mean_std`` redraws indices and
    reduces one column at a time -- that dominates ``fit_noise_model``'s
    per-config weight precompute on real Willow configs (up to ~16,762
    co-occurring detector pairs). Sharing resample indices across columns
    correlates the columns' bootstrap errors with each other, but does not
    bias any single column's own (mean, std): each column's distribution over
    the ``n_bootstrap`` resample means is still exactly the same
    draws-with-replacement bootstrap ``bootstrap_mean_std`` computes for that
    column alone, so per-column outputs are statistically equivalent to
    (though not seeded/ordered identically to) the per-item loop.

    Does not chunk internally -- callers passing very wide matrices (e.g. tens
    of thousands of pairs) should slice into column blocks themselves to
    bound the ``values_matrix`` working set; see the chunked call sites in
    ``fit_noise_model``.
    """
    rng = np.random.default_rng(seed)
    n_shots, n_items = values_matrix.shape
    resample_means = np.empty((n_bootstrap, n_items), dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_shots, size=n_shots)
        counts = np.bincount(idx, minlength=n_shots).astype(np.float64)
        resample_means[b] = (counts @ values_matrix) / n_shots
    return resample_means.mean(axis=0), resample_means.std(axis=0)


def _params_to_vector(noise) -> "np.ndarray":
    """25-param NoiseModel -> a flat, ordered numpy vector (order defined here,
    consistently used by _vector_to_params -- p_prep_X and p_meas_X are always
    tied equal to their _Z counterparts, so this is a 23-DOF optimization
    surfaced through a 25-field NoiseModel; see Global Constraints)."""
    from dataclasses import fields

    order = [f.name for f in fields(noise) if not f.name.startswith("_")
             and f.name not in ("p_prep_X", "p_meas_X")]
    return np.array([getattr(noise, name) for name in order], dtype=np.float64), order


def _vector_to_dict(vec: "np.ndarray", order: list[str]) -> dict[str, float]:
    # Clamp to the [0, 1] box bounds: L-BFGS-B can return a value a float ULP past
    # a bound (e.g. p_meas 1.00000001), which NoiseModel.validate() rejects. The
    # overshoot is FP noise, not signal, so clamping is exact, not a fudge.
    d = {name: float(min(1.0, max(0.0, v))) for name, v in zip(order, vec)}
    d["p_prep_X"] = d["p_prep_Z"]
    d["p_meas_X"] = d["p_meas_Z"]
    return d


def _vector_to_params(vec: "np.ndarray", order: list[str], noise_cls):
    return noise_cls(**_vector_to_dict(vec, order))


def _joint_constraint_violation(d: dict[str, float]) -> float:
    """``NoiseModel.validate()`` enforces ``sum(p_cnot_*) <= 1``,
    ``sum(p_idle_cnot_*) <= 1``, and ``sum(p_idle_spam_*) <= 1`` -- joint
    constraints the optimizer's independent per-parameter box bounds don't
    encode. Returns ``0.0`` if every group sum is within bounds, else a
    smooth positive penalty proportional to the overage (used by ``loss``
    below to keep L-BFGS-B's line search from ever constructing an infeasible
    ``NoiseModel`` -- found empirically: large bootstrap-derived
    inverse-variance weights can drive the very first, unscaled line-search
    step to the box corner on every ``p_cnot_*`` axis simultaneously, which
    ``NoiseModel.__post_init__`` rejects with a ``ValueError`` that would
    otherwise crash the optimization outright).
    """
    violation = 0.0
    for prefix in ("p_cnot_", "p_idle_cnot_", "p_idle_spam_"):
        total = sum(v for k, v in d.items() if k.startswith(prefix))
        if total > 1.0:
            violation += (total - 1.0) ** 2
    return violation


def _loss_over_configs(candidate, precomputed) -> float:
    """Sum of weighted squared moment residuals for ``candidate`` (a NoiseModel
    instance) over every precomputed config -- the per-point computation
    ``fit_noise_model``'s ``loss`` closure used to do inline, factored out to a
    top-level function so it is picklable/callable from a worker process (see
    ``_worker_loss`` below, which parallelizes exactly this call across the
    finite-difference gradient's probe points).
    """
    from ising_sim2real.ingest.noise_injector import inject_noise_model

    total = 0.0
    for circuit_noisy_template, pairs, real_singles, real_pijs, single_weights, pair_weights in precomputed:
        noisy_circuit = inject_noise_model(circuit_noisy_template, candidate)
        # approximate_disjoint_errors=True is REQUIRED once PAULI_CHANNEL_2
        # sites exist (confirmed directly against stim: omitting it raises
        # "PAULI_CHANNEL_2 ... requires the approximate_disjoint_errors
        # option") -- every candidate circuit has nonzero 2Q-gate params, so
        # this is on the hot path of every optimizer iteration, for every
        # config.
        dem = noisy_circuit.detector_error_model(approximate_disjoint_errors=True)
        pred_singles, pred_pijs = predicted_moments(dem, pairs)
        for det, w in single_weights.items():
            total += w * (pred_singles.get(det, 1.0) - real_singles[det]) ** 2
        for pair, w in pair_weights.items():
            total += w * (pred_pijs.get(pair, 1.0) - real_pijs[pair]) ** 2
    return total


def _loss_vec(vec: "np.ndarray", order: list[str], noise_cls, precomputed) -> float:
    """Vector -> scalar loss (constraint-penalty check + ``_loss_over_configs``)
    -- exactly what ``fit_noise_model``'s old inline ``loss`` closure computed.
    Called by every worker process's ``_worker_loss``, so the formula lives in
    one place regardless of how many processes evaluate it in parallel.
    """
    d = _vector_to_dict(vec, order)
    violation = _joint_constraint_violation(d)
    if violation > 0.0:
        return 1e8 * (1.0 + violation)
    candidate = noise_cls(**d)
    return _loss_over_configs(candidate, precomputed)


# Process-local state for worker processes in fit_noise_model's persistent
# ProcessPoolExecutor -- populated once per worker by `_init_worker` (NOT
# re-sent on every gradient evaluation), so the only per-call IPC payload is a
# single small parameter vector in and a float out.
_worker_state: dict = {}


def _init_worker(precomputed, order: list[str], ising_code_path: str) -> None:
    """ProcessPoolExecutor initializer -- runs ONCE per worker process when the
    pool is created (not per call), stashing this fit's per-config precomputed
    data, parameter order, and the NoiseModel class in process-local globals.
    This is what makes a PERSISTENT pool (created once per ``fit_noise_model``
    call, reused across the whole L-BFGS-B run) worthwhile over spawning a
    fresh pool per gradient evaluation: ``precomputed`` (per-config circuits,
    moments, weights) is pickled to each worker exactly once.
    """
    import sys

    if ising_code_path not in sys.path:
        sys.path.insert(0, ising_code_path)
    from qec.noise_model import NoiseModel  # type: ignore

    _worker_state["precomputed"] = precomputed
    _worker_state["order"] = order
    _worker_state["noise_cls"] = NoiseModel


def _worker_loss(vec: "np.ndarray") -> float:
    """Runs in a worker process (see ``_init_worker``): evaluate ``_loss_vec``
    for one candidate parameter vector against this worker's stashed
    precomputed data. This is the unit of parallel work dispatched by
    ``fit_noise_model``'s ``fun_and_grad`` -- one call per base/perturbed
    finite-difference gradient probe point, embarrassingly parallel since
    every point's loss is independent of every other point's.
    """
    state = _worker_state
    return _loss_vec(vec, state["order"], state["noise_cls"], state["precomputed"])


def fit_noise_model(
    configs: list[tuple["stim.DetectorErrorModel", "stim.Circuit", "np.ndarray"]],
    init,
    n_workers: int | None = None,
):
    """Fit a SINGLE shared 25-param ``NoiseModel`` (23 free DOF -- see
    ``_params_to_vector``) to real detection events pooled over MULTIPLE
    configs, by minimizing weighted least squares between real and predicted
    ``<s_i>``/``<s_i s_j>`` moments, summed over every config, at each
    candidate point.

    ``configs`` is a list of ``(dem_si1000, circuit_noisy_template,
    detection_events)`` triples -- one per (round-count, basis) config in a
    patch's fit-set. Each config generally has a structurally different
    circuit/DEM (different round counts emit different detector layers), so
    the per-config moments and weights are precomputed once up front and the
    candidate ``NoiseModel`` is injected into EACH config's own
    ``circuit_noisy_template`` on every ``loss`` evaluation -- see the
    "Loss" section of
    ``docs/superpowers/specs/2026-07-09-25-param-fit-rung-design.md``.

    Each config's own ``dem_si1000`` supplies its own pairwise "window"
    (``pairs_from_dem``) -- the reference graph structure, same DEM the
    ``syndrome`` rung uses.

    ``n_workers`` bounds the persistent process pool used to parallelize the
    optimizer's per-iteration finite-difference gradient (see the pool setup
    below) -- defaults to ``os.cpu_count()``, capped at ``len(order) + 1``
    (there are never more independent gradient probe points than that to
    dispatch in one batch, so extra workers beyond that would idle).
    """
    from ising_sim2real.ingest.syndrome_dem import spins_from_detection_events

    # Column-chunk width for the batched bootstrap precompute below: bounds
    # the transient (n_shots, chunk) working array so the widest real configs
    # (up to ~16,762 co-occurring pairs) never materialize a multi-GB matrix
    # in one shot (50,000 shots x 16,762 pairs x 8 bytes ~= 6.7 GB unchunked).
    _BOOTSTRAP_CHUNK = 2000

    precomputed = []
    for dem_si1000, circuit_noisy_template, detection_events in configs:
        pairs = pairs_from_dem(dem_si1000)
        real_singles, real_pijs = real_moments(detection_events, pairs)

        spins = spins_from_detection_events(detection_events)

        single_weights = {}
        n_det = spins.shape[1]
        for start in range(0, n_det, _BOOTSTRAP_CHUNK):
            stop = min(start + _BOOTSTRAP_CHUNK, n_det)
            _means, stds = bootstrap_mean_std_batch(spins[:, start:stop], n_bootstrap=50, seed=start)
            for d, std in zip(range(start, stop), stds):
                single_weights[d] = 1.0 / (std ** 2 + 1e-8)

        pair_weights = {}
        if pairs:
            i_arr = np.array([i for i, _ in pairs], dtype=np.intp)
            j_arr = np.array([j for _, j in pairs], dtype=np.intp)
            n_pairs = len(pairs)
            for start in range(0, n_pairs, _BOOTSTRAP_CHUNK):
                stop = min(start + _BOOTSTRAP_CHUNK, n_pairs)
                block = spins[:, i_arr[start:stop]] * spins[:, j_arr[start:stop]]
                # seed offset by a large constant so single- and pair- chunk
                # seeds never collide across the two loops above.
                _means, stds = bootstrap_mean_std_batch(block, n_bootstrap=50, seed=100_000_000 + start)
                for pair, std in zip(pairs[start:stop], stds):
                    pair_weights[pair] = 1.0 / (std ** 2 + 1e-8)

        precomputed.append(
            (circuit_noisy_template, pairs, real_singles, real_pijs, single_weights, pair_weights)
        )

    noise_cls = type(init)
    x0, order = _params_to_vector(init)
    bounds = [(0.0, 1.0)] * len(order)

    # scipy's default (no `jac=`) finite-difference gradient needs ~len(order)+1
    # independent `loss()` calls per L-BFGS-B iteration (one base point plus one
    # forward-difference perturbation per free parameter) -- each `loss()` call
    # itself sums an expensive `inject_noise_model` + `detector_error_model` +
    # `predicted_moments` pass over EVERY config in `precomputed`, making the
    # gradient the dominant real-world cost (measured: single real patch fits
    # projected at 5-30 hours single-threaded). Every one of these probe points
    # is independent, so they are dispatched to a persistent worker pool below
    # instead of being computed one at a time on the calling process -- the
    # pool is created ONCE for the whole optimization (not per iteration) so
    # `precomputed` (per-config circuits/moments/weights) is only pickled to
    # each worker a single time (via `_init_worker`), not on every gradient
    # evaluation.
    n_points = len(order) + 1
    if n_workers is None:
        n_workers = os.cpu_count() or 1
    n_workers = max(1, min(n_workers, n_points))

    pool = ProcessPoolExecutor(
        max_workers=n_workers,
        # Pinned explicitly: this code is validated under macOS's "spawn"
        # default, but Linux (Gautschi) defaults to "fork", which forks a
        # process that may hold live BLAS/OpenMP thread state from the
        # precompute phase's matvecs -- an untested, classic fork-safety
        # hazard. "spawn" is slower to start workers but is the path this
        # module was actually tested against on every platform.
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(precomputed, order, str(ISING_CODE)),
    )
    try:
        def fun_and_grad(vec: "np.ndarray"):
            points = [vec]
            for i in range(len(vec)):
                perturbed = vec.copy()
                perturbed[i] += _FD_EPS
                points.append(perturbed)
            values = np.fromiter(
                pool.map(_worker_loss, points, chunksize=1), dtype=np.float64, count=len(points)
            )
            f0 = values[0]
            grad = (values[1:] - f0) / _FD_EPS
            return float(f0), grad

        # jac=True: `fun_and_grad` returns (value, gradient) together from ONE
        # batch of parallel probe evaluations, so L-BFGS-B's per-trial-point
        # (f, g) request never triggers two separate, redundant evaluation
        # rounds the way a plain `fun=`/`jac=` pair would.
        result = minimize(fun_and_grad, x0, method="L-BFGS-B", jac=True, bounds=bounds)
    finally:
        pool.shutdown(wait=True)

    return _vector_to_params(result.x, order, noise_cls)
