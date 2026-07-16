"""Unit tests for the syndrome-estimated DEM math primitives (arXiv:2606.11496
Appendix B). No network, no real data -- pure numpy on hand-constructed arrays.
"""

from __future__ import annotations

import numpy as np
import pytest
import stim

from ising_sim2real.ingest.syndrome_dem import (
    bootstrap_mean_std,
    estimate_dem_from_syndromes,
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


def _estimated_probs(dem: "stim.DetectorErrorModel") -> dict[frozenset[int], float]:
    out = {}
    for instr in dem.flattened():
        if instr.type == "error":
            support = frozenset(
                t.val for t in instr.targets_copy() if t.is_relative_detector_id()
            )
            out[support] = instr.args_copy()[0]
    return out


def test_estimate_recovers_graphlike_probabilities():
    dem_text = "error(0.05) D0 D1\nerror(0.03) D1 D2\nerror(0.01) D0\nerror(0.02) D2\n"
    dem = stim.DetectorErrorModel(dem_text)
    sampler = dem.compile_sampler(seed=1)
    dets, _obs, _ = sampler.sample(shots=500_000)

    estimated = estimate_dem_from_syndromes(dem, dets)
    probs = _estimated_probs(estimated)

    assert probs[frozenset({0, 1})] == pytest.approx(0.05, abs=0.005)
    assert probs[frozenset({1, 2})] == pytest.approx(0.03, abs=0.005)
    assert probs[frozenset({0})] == pytest.approx(0.01, abs=0.005)
    assert probs[frozenset({2})] == pytest.approx(0.02, abs=0.005)


def test_estimate_recovers_hyperedge_probabilities():
    dem_text = (
        "error(0.05) D0 D1\nerror(0.03) D1 D2\nerror(0.01) D0\n"
        "error(0.02) D2\nerror(0.02) D0 D1 ^ D2\n"
    )
    dem = stim.DetectorErrorModel(dem_text)
    sampler = dem.compile_sampler(seed=2)
    dets, _obs, _ = sampler.sample(shots=1_000_000)

    estimated = estimate_dem_from_syndromes(dem, dets)
    probs = _estimated_probs(estimated)

    assert probs[frozenset({0, 1})] == pytest.approx(0.05, abs=0.005)
    assert probs[frozenset({1, 2})] == pytest.approx(0.03, abs=0.005)
    assert probs[frozenset({0})] == pytest.approx(0.01, abs=0.005)
    assert probs[frozenset({2})] == pytest.approx(0.02, abs=0.005)
    assert probs[frozenset({0, 1, 2})] == pytest.approx(0.02, abs=0.005)


def test_estimate_preserves_dem_structure():
    # Same graph, same num_detectors/num_observables, only probabilities differ.
    dem_text = "error(0.05) D0 D1 L0\nerror(0.01) D0\n"
    dem = stim.DetectorErrorModel(dem_text)
    sampler = dem.compile_sampler(seed=9)
    dets, _obs, _ = sampler.sample(shots=100_000)

    estimated = estimate_dem_from_syndromes(dem, dets)
    assert estimated.num_detectors == dem.num_detectors
    assert estimated.num_observables == dem.num_observables

    # Logical-observable assignment must be inherited unchanged.
    original_logicals = {}
    for instr in dem.flattened():
        if instr.type == "error":
            support = frozenset(
                t.val for t in instr.targets_copy() if t.is_relative_detector_id()
            )
            logicals = frozenset(
                t.val for t in instr.targets_copy() if t.is_logical_observable_id()
            )
            original_logicals[support] = logicals
    for instr in estimated.flattened():
        if instr.type == "error":
            support = frozenset(
                t.val for t in instr.targets_copy() if t.is_relative_detector_id()
            )
            logicals = frozenset(
                t.val for t in instr.targets_copy() if t.is_logical_observable_id()
            )
            assert logicals == original_logicals[support]


def test_estimate_raises_on_ambiguous_shared_support():
    # Two events with the SAME detector support but DIFFERENT logical support
    # cannot be told apart from detector moments alone (paper Appendix A) --
    # this must fail loudly, not silently mis-split the combined rate.
    dem_text = "error(0.05) D0 D1 L0\nerror(0.03) D0 D1 L1\n"
    dem = stim.DetectorErrorModel(dem_text)
    sampler = dem.compile_sampler(seed=11)
    dets, _obs, _ = sampler.sample(shots=1000)

    with pytest.raises(NotImplementedError):
        estimate_dem_from_syndromes(dem, dets)


# --- Task 7: top-10 targeted bug-scenario tests ------------------------------
#
# Scenario 1 (spin sign convention: fired detector -> spin -1) is already
# covered verbatim by `test_spins_from_detection_events` above -- skipped here.


def test_nonempty_subsets_size_four_is_fifteen_distinct():
    # Scenario 2: the dataset's max event size is 4 -> 2**4 - 1 = 15 nonempty
    # subsets, including the full set and all four singletons.
    support = frozenset({1, 2, 3, 4})
    subsets = nonempty_subsets(support)
    assert len(subsets) == 15
    assert len(set(subsets)) == 15  # all distinct
    assert support in subsets  # the full set is included
    singletons = {s for s in subsets if len(s) == 1}
    assert singletons == {frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4})}


def test_reemission_preserves_separator_position():
    # Scenario 3: `estimate_dem_from_syndromes` re-emits `error` instructions
    # via `out.append("error", p[support], targets)`, reusing the ORIGINAL
    # `targets_copy()` list wholesale (not rebuilding it from the support
    # set) -- so a `^` separator's position must survive byte-for-byte, not
    # just the derived support set. D1/D4/D20 is a single event with support
    # {1,4,20}; if a future change rebuilt targets from the support set
    # instead of reusing the original list, the `^` token would silently
    # disappear (changing the decomposition hint for matching decoders).
    dem_text = "error(0.05) D1 D4 ^ D20\n"
    dem = stim.DetectorErrorModel(dem_text)
    sampler = dem.compile_sampler(seed=3)
    dets, _obs, _ = sampler.sample(shots=1000)

    estimated = estimate_dem_from_syndromes(dem, dets)

    def error_targets(d: "stim.DetectorErrorModel"):
        for instr in d.flattened():
            if instr.type == "error":
                return instr.targets_copy()
        raise AssertionError("no error instruction found")

    orig_targets = error_targets(dem)
    new_targets = error_targets(estimated)
    orig_tokens = [str(t) for t in orig_targets]
    new_tokens = [str(t) for t in new_targets]
    assert orig_tokens == ["D1", "D4", "^", "D20"]
    assert new_tokens == orig_tokens  # separator position (index 2) unchanged
    assert [t.is_separator() for t in new_targets] == [t.is_separator() for t in orig_targets]


def test_hierarchical_correction_direction_is_division_not_multiplication():
    # Scenario 4: the correction step for a subset event must DIVIDE by each
    # strict superset's q (`correction *= q[other_support] ** -1`), not
    # multiply. Verified against the existing hyperedge toy DEM
    # (same fixture as test_estimate_recovers_hyperedge_probabilities):
    # support {0} is a strict subset of both {0,1} and {0,1,2}, so its
    # correction combines both superset q-values. We recover q_{0,1} and
    # q_{0,1,2} from the code's own output (q = 1 - 2p), independently
    # recompute the moment for {0} from the same raw detection events, and
    # then compute p_e for {0} both the correct way (dividing) and the
    # deliberately-wrong way (multiplying). The two diverge by more than an
    # order of magnitude (0.0101 vs 0.1347, computed once via a companion
    # script) -- the actual code output must land on the correct one.
    dem_text = (
        "error(0.05) D0 D1\nerror(0.03) D1 D2\nerror(0.01) D0\n"
        "error(0.02) D2\nerror(0.02) D0 D1 ^ D2\n"
    )
    dem = stim.DetectorErrorModel(dem_text)
    sampler = dem.compile_sampler(seed=2)
    dets, _obs, _ = sampler.sample(shots=1_000_000)

    estimated = estimate_dem_from_syndromes(dem, dets)
    probs = _estimated_probs(estimated)

    q01 = 1.0 - 2.0 * probs[frozenset({0, 1})]
    q012 = 1.0 - 2.0 * probs[frozenset({0, 1, 2})]

    spins = spins_from_detection_events(dets)
    m0 = float(spins[:, 0].mean())
    assert m0 >= 0  # sanity: no regularization involved, fully deterministic

    q_correct = m0 * (q01**-1) * (q012**-1)
    q_wrong = m0 * q01 * q012  # the deliberately-wrong direction
    p_correct = (1.0 - q_correct) / 2.0
    p_wrong = (1.0 - q_wrong) / 2.0

    actual = probs[frozenset({0})]
    assert actual == pytest.approx(p_correct, abs=1e-9)
    assert abs(actual - p_wrong) > 0.05  # not vacuously close to the wrong direction


def test_r_e_negative_clamps_to_exactly_zero():
    # Scenario 5: a strongly, consistently negative singleton moment (95% of
    # 200 shots fire, spin -1) is a "resolved" negative sign under
    # `regularize_moment` (SNR >> 0.5), so it stays negative. For a size-1
    # event r_e IS that moment (only subset is itself, sign +1), so r_e < 0
    # -- verified independently: raw mean is exactly -0.9 here (190 fired /
    # 200 shots), which regularize_moment keeps as -0.9 since the bootstrap
    # std around a 95/5 split is tiny relative to the mean. This must clamp
    # to probability exactly 0.0 (not NaN, not complex, not left negative).
    n_shots = 200
    n_not_fired = 10
    det = np.ones((n_shots, 1), dtype=bool)  # default: fired -> spin -1
    det[:n_not_fired, 0] = False  # a few not fired -> spin +1
    spins = spins_from_detection_events(det)
    raw = float(spins[:, 0].mean())
    assert raw == -0.9  # confirms the fixture actually produces a negative raw moment
    regularized = regularize_moment(spins[:, 0].copy(), raw)
    assert regularized == -0.9  # confirms it stays negative (resolved, not floored)

    dem = stim.DetectorErrorModel("error(0.05) D0\n")
    estimated = estimate_dem_from_syndromes(dem, det)
    probs = _estimated_probs(estimated)
    assert probs[frozenset({0})] == 0.0


def test_p_e_out_of_range_clamps_to_exactly_zero():
    # Scenario 6: a hand-built joint distribution over two detectors where
    # r_e for the size-1 event {0} is NON-negative (0.16, so this does NOT
    # take the r_e<0 branch of scenario 5) but the correction pulled in from
    # the size-2 superset {0,1} is large enough to push the raw q_e/p_e
    # formula out of [0,1]. Verified with a companion search script: with
    # joint counts n(+1,+1)=20, n(+1,-1)=9, n(-1,+1)=6, n(-1,-1)=15 (N=50),
    # the moments are m0=0.16, m1=0.04, m01=0.4, giving r_e({0,1})=0.016 (in
    # range) but q_e({0}) = m0 / sqrt(r_e({0,1})) = 0.16/0.1265 = 1.265,
    # i.e. p_e({0}) = (1-1.265)/2 = -0.132 -- outside [0,1], so this must hit
    # `if not (0.0 <= p_e <= 1.0): p_e = 0.0` (not the r_e<0 branch, since
    # r_e({0}) = m0 = 0.16 >= 0 the whole time).
    counts = {(False, False): 20, (False, True): 9, (True, False): 6, (True, True): 15}
    rows = []
    for (v0, v1), n in counts.items():
        rows.extend([[v0, v1]] * n)
    det = np.array(rows, dtype=bool)
    assert det.shape == (50, 2)

    spins = spins_from_detection_events(det)
    m0 = float(spins[:, 0].mean())
    assert m0 == pytest.approx(0.16)
    assert m0 >= 0  # confirms r_e({0}) itself is non-negative -- isolates the p_e/q_e clamp

    dem = stim.DetectorErrorModel("error(0.05) D0\nerror(0.03) D0 D1\n")
    estimated = estimate_dem_from_syndromes(dem, det)
    probs = _estimated_probs(estimated)
    assert probs[frozenset({0})] == 0.0
    # The size-2 event itself is unaffected -- only the corrected singleton is.
    assert 0.0 < probs[frozenset({0, 1})] < 1.0


def test_bootstrap_mean_std_single_resample():
    # Scenario 7: n_bootstrap=1 runs without error. Documented actual
    # behavior (not a value the spec mandates): std is always exactly 0.0,
    # since a single resample produces one scalar mean and `.std()` of a
    # length-1 array is 0 by definition; mean is whatever that one
    # bootstrap draw produced (deterministic given the seed).
    values = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    mean, std = bootstrap_mean_std(values, n_bootstrap=1, seed=0)
    assert np.isfinite(mean)
    assert std == 0.0

    constant = np.full(10, 0.4)
    mean_c, std_c = bootstrap_mean_std(constant, n_bootstrap=1, seed=0)
    assert mean_c == 0.4
    assert std_c == 0.0


def test_estimate_preserves_instruction_count_and_order():
    # Scenario 8: the output DEM has the same total instruction COUNT and
    # relative ORDER as the input, including non-`error` instructions
    # (`detector(...)`) interspersed between error lines -- not just the
    # same detector graph.
    dem_text = (
        "detector(0,0) D0\n"
        "error(0.05) D0 D1\n"
        "detector(1,0) D1\n"
        "error(0.03) D1 D2\n"
        "detector(2,0) D2\n"
    )
    dem = stim.DetectorErrorModel(dem_text)
    sampler = dem.compile_sampler(seed=6)
    dets, _obs, _ = sampler.sample(shots=1000)

    estimated = estimate_dem_from_syndromes(dem, dets)

    orig = list(dem.flattened())
    new = list(estimated.flattened())
    assert len(new) == len(orig)
    assert [instr.type for instr in new] == [instr.type for instr in orig]


def test_duplicate_identical_support_and_logicals_does_not_raise():
    # Scenario 9: two `error` lines with the SAME support AND the SAME
    # logical-observable assignment (e.g. because the shipped DEM genuinely
    # merged two fault mechanisms into one line, appearing twice) must NOT
    # raise -- only a support seen with DIFFERENT logicals is ambiguous
    # (test_estimate_raises_on_ambiguous_shared_support above). This
    # exercises the `seen[support] != logicals` guard's implicit else
    # (identical logicals -> no raise).
    dem_text = "error(0.05) D0 D1 L0\nerror(0.05) D0 D1 L0\n"
    dem = stim.DetectorErrorModel(dem_text)
    sampler = dem.compile_sampler(seed=5)
    dets, _obs, _ = sampler.sample(shots=1000)

    estimated = estimate_dem_from_syndromes(dem, dets)  # must not raise
    error_instrs = [instr for instr in estimated.flattened() if instr.type == "error"]
    assert len(error_instrs) == 2
    # Both lines share one support -> both get the same estimated probability.
    assert error_instrs[0].args_copy()[0] == error_instrs[1].args_copy()[0]


def test_shots_column_mismatch_raises_indexerror():
    # Scenario 10: `detection_events` whose column count does NOT match
    # `dem_si1000.num_detectors` is not validated anywhere in this module
    # (by design -- no defensive checks for cases that can't happen in the
    # real call path). This pins the CURRENT failure mode (a numpy
    # IndexError from `spins[:, i]` inside the subset-moment loop) as a
    # regression tripwire, not a spec requirement. This test documents
    # existing behavior; no production code change is implied.
    dem = stim.DetectorErrorModel("error(0.05) D0 D1\n")  # needs 2 detector columns
    too_few = np.zeros((10, 1), dtype=bool)
    with pytest.raises(IndexError):
        estimate_dem_from_syndromes(dem, too_few)


# --- Task 7: byte-for-byte E2E regression -------------------------------------


def test_e2e_pipeline_pinned_exact_values():
    # Pins the WHOLE pipeline (sampling -> spins -> moments -> hierarchical
    # inversion -> re-emission) as a single deterministic unit, exact float
    # equality (not pytest.approx). The four expected values below were
    # captured by actually running this exact code once (seed=123, 10_000
    # shots on the plan's verified graphlike toy DEM) via a companion
    # script -- not guessed. Any future change that alters so much as the
    # last bit of a float here must consciously update this test.
    #
    # The events are sampled with numpy rather than dem.compile_sampler(seed=):
    # Stim only guarantees a seeded stream on the same machine AND the same Stim
    # version (SIMD width changes it), so exact floats pinned downstream of it
    # hold on the machine that captured them and fail elsewhere. numpy's PCG64
    # stream is guaranteed across platforms, and the estimator's own bootstrap is
    # already PCG64-seeded, which makes the whole pipeline portable-deterministic.
    dem_text = "error(0.05) D0 D1\nerror(0.03) D1 D2\nerror(0.01) D0\nerror(0.02) D2\n"
    dem = stim.DetectorErrorModel(dem_text)
    rng = np.random.default_rng(123)
    dets = np.zeros((10_000, 3), dtype=bool)
    for p, support in [(0.05, (0, 1)), (0.03, (1, 2)), (0.01, (0,)), (0.02, (2,))]:
        fired = rng.random(10_000) < p
        for d in support:
            dets[fired, d] ^= True

    estimated = estimate_dem_from_syndromes(dem, dets)
    probs = _estimated_probs(estimated)

    assert probs[frozenset({0, 1})] == 0.04904926761371853
    assert probs[frozenset({1, 2})] == 0.02748009793915457
    assert probs[frozenset({0})] == 0.010589551918168683
    assert probs[frozenset({2})] == 0.02150163619799117


def test_e2e_pipeline_no_hidden_nondeterminism():
    # Running estimate_dem_from_syndromes TWICE on the SAME already-sampled
    # detection_events array (no re-sampling) must be exactly reproducible
    # -- confirms the estimator itself (in particular regularize_moment's
    # bootstrap, which is seeded with a fixed default seed=0) has no hidden
    # source of nondeterminism independent of the initial Stim sampling.
    dem_text = "error(0.05) D0 D1\nerror(0.03) D1 D2\nerror(0.01) D0\nerror(0.02) D2\n"
    dem = stim.DetectorErrorModel(dem_text)
    sampler = dem.compile_sampler(seed=123)
    dets, _obs, _ = sampler.sample(shots=10_000)

    probs_a = _estimated_probs(estimate_dem_from_syndromes(dem, dets))
    probs_b = _estimated_probs(estimate_dem_from_syndromes(dem, dets))
    assert probs_a == probs_b
