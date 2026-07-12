"""Unit tests for param_fit.py's forward analytic moment computation --
hand-verified toy DEMs with known expected moments (derived by hand from the
inclusion-exclusion formulas in this module's docstring, not by running other
code -- an independent check)."""

from __future__ import annotations

import zlib

import stim

from ising_sim2real.ingest.param_fit import pairs_from_dem, predicted_moments


def test_predicted_single_detector_moment_one_event():
    # error(0.1) D0: <s_0> = 1 - 2*0.1 = 0.8
    dem = stim.DetectorErrorModel("error(0.1) D0\n")
    singles, _ = predicted_moments(dem, pairs=[])
    assert abs(singles[0] - 0.8) < 1e-12


def test_predicted_single_detector_moment_two_events():
    # D0 covered by error(0.1) and error(0.2):
    # <s_0> = (1-0.2)*(1-0.4) = 0.8*0.6 = 0.48
    dem = stim.DetectorErrorModel("error(0.1) D0\nerror(0.2) D0 D1\n")
    singles, _ = predicted_moments(dem, pairs=[])
    assert abs(singles[0] - 0.48) < 1e-12
    # D1 covered only by the second event: <s_1> = 1 - 0.4 = 0.6
    assert abs(singles[1] - 0.6) < 1e-12


def test_predicted_pairwise_moment_shared_event_only():
    # Single event covers BOTH D0 and D1 -> <s_0 s_1> = 1 (always equal).
    dem = stim.DetectorErrorModel("error(0.3) D0 D1\n")
    _, pijs = predicted_moments(dem, pairs=[(0, 1)])
    assert abs(pijs[(0, 1)] - 1.0) < 1e-12


def test_predicted_pairwise_moment_disjoint_events():
    # D0 and D1 covered by DISJOINT events (0.1 and 0.2 respectively):
    # <s_0 s_1> = (1-0.2)*(1-0.4) = 0.8 * 0.6 = 0.48 (both factors apply, since
    # each event covers exactly one of {0,1}).
    dem = stim.DetectorErrorModel("error(0.1) D0\nerror(0.2) D1\n")
    _, pijs = predicted_moments(dem, pairs=[(0, 1)])
    assert abs(pijs[(0, 1)] - 0.48) < 1e-12


def test_predicted_pairwise_moment_mixed_shared_and_disjoint():
    # error(0.3) covers both D0,D1 (drops out, factor 1).
    # error(0.1) covers only D0 (factor 1-0.2=0.8).
    # <s_0 s_1> = 1 * 0.8 = 0.8
    dem = stim.DetectorErrorModel("error(0.3) D0 D1\nerror(0.1) D0\n")
    _, pijs = predicted_moments(dem, pairs=[(0, 1)])
    assert abs(pijs[(0, 1)] - 0.8) < 1e-12


def test_pairs_from_dem_collects_all_pairs_within_events():
    # error(0.1) D0 D1 D2 (size-3 event) contributes all C(3,2)=3 pairs.
    # error(0.2) D2 D3 contributes 1 more pair. D0-D3 never co-occur -> absent.
    dem = stim.DetectorErrorModel("error(0.1) D0 D1 D2\nerror(0.2) D2 D3\n")
    pairs = set(pairs_from_dem(dem))
    assert pairs == {(0, 1), (0, 2), (1, 2), (2, 3)}


def test_pairs_from_dem_ignores_size_one_events():
    dem = stim.DetectorErrorModel("error(0.1) D0\nerror(0.2) D1\n")
    assert pairs_from_dem(dem) == []


import numpy as np


def test_real_moments_matches_predicted_on_sampled_data():
    """Sample from a KNOWN DEM, confirm real_moments recovers close to the
    analytic prediction (law of large numbers -- this is the bridge test
    connecting the real-data side to the analytic forward side)."""
    from ising_sim2real.ingest.param_fit import real_moments

    dem = stim.DetectorErrorModel("error(0.1) D0 D1\nerror(0.05) D1 D2\nerror(0.02) D0\n")
    sampler = dem.compile_sampler(seed=42)
    dets, _obs, _ = sampler.sample(shots=500_000)

    pairs = pairs_from_dem(dem)
    predicted_singles, predicted_pijs = predicted_moments(dem, pairs)
    real_singles, real_pijs = real_moments(dets, pairs)

    for d in predicted_singles:
        assert abs(real_singles[d] - predicted_singles[d]) < 0.01
    for pair in pairs:
        assert abs(real_pijs[pair] - predicted_pijs[pair]) < 0.01


def test_bootstrap_mean_std_batch_matches_manual_computation_for_one_resample():
    """Hand-checkable case: with n_bootstrap=1, bootstrap_mean_std_batch's
    output must exactly equal the mean of the SAME resampled rows computed by
    hand from an independently-reproduced index draw (no bootstrap
    mean/std-of-resamples averaging to reason about with only one resample)."""
    from ising_sim2real.ingest.param_fit import bootstrap_mean_std_batch

    matrix = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
    ])
    seed = 42
    means, stds = bootstrap_mean_std_batch(matrix, n_bootstrap=1, seed=seed)

    # Independently reproduce the exact same index draw to hand-check.
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
    expected_means = matrix[idx].mean(axis=0)

    assert np.allclose(means, expected_means)
    # A single resample's distribution over itself has std exactly 0, for
    # every column.
    assert np.allclose(stds, 0.0)


def test_bootstrap_mean_std_batch_matches_per_item_bootstrap_statistically():
    """bootstrap_mean_std_batch's per-column (mean, std) should be
    statistically equivalent to calling the existing per-item
    ``syndrome_dem.bootstrap_mean_std`` on that column alone -- three columns
    with known, well-separated means, enough shots/resamples that both
    independent code paths' bootstrap statistics should closely agree."""
    from ising_sim2real.ingest.param_fit import bootstrap_mean_std_batch
    from ising_sim2real.ingest.syndrome_dem import bootstrap_mean_std

    rng = np.random.default_rng(123)
    n_shots = 20_000
    col0 = rng.choice([-1.0, 1.0], size=n_shots, p=[0.75, 0.25])  # mean ~ -0.5
    col1 = rng.choice([-1.0, 1.0], size=n_shots, p=[0.5, 0.5])    # mean ~  0.0
    col2 = rng.choice([-1.0, 1.0], size=n_shots, p=[0.1, 0.9])    # mean ~  0.8
    matrix = np.stack([col0, col1, col2], axis=1)

    batch_means, batch_stds = bootstrap_mean_std_batch(matrix, n_bootstrap=200, seed=0)

    for k, col in enumerate((col0, col1, col2)):
        ref_mean, ref_std = bootstrap_mean_std(col, n_bootstrap=200, seed=k)
        assert abs(batch_means[k] - ref_mean) < 0.01, (k, batch_means[k], ref_mean)
        assert abs(batch_means[k] - col.mean()) < 0.01
        # Both are independent bootstrap std ESTIMATES of the same
        # population std-of-the-mean, not the same draws -- generous relative
        # tolerance to avoid flakiness from the estimator's own sampling noise.
        assert abs(batch_stds[k] - ref_std) < 0.5 * ref_std + 0.002, (k, batch_stds[k], ref_std)


def _toy_noise_fit_template() -> stim.Circuit:
    """Toy circuit for the fit rung's ground-truth-recovery tests (this test,
    plus Task 6's ``test_fit_converges_across_representative_regimes``, which
    reuses this same template).

    A single-pair 2-detector circuit (the original toy here) is degenerate for
    23 free parameters: only 3 moment equations (<s_0>, <s_1>, <s_0 s_1>), and
    its lone ``DEPOLARIZE1`` sits in a bulk tick with no ``R``/``M``, so
    ``p_idle_spam_*`` never affects it at all. Worse, hand-derived Jacobian
    analysis (see the implementation-plan notes) showed that even a richer
    *single-check-type* repetition-code-style circuit has EXACT (not just
    correlated) degeneracies: any error localized purely on a check ancilla
    within its own reset-to-measurement window -- whether from ``p_prep_Z`` or
    from the ancilla-side component of a two-qubit gate error occurring at ANY
    of that ancilla's CZ sites -- propagates through CZ trivially (Z commutes
    with CZ) and leaves no residual on data, so it is physically
    indistinguishable from every other such error UNLESS something else in the
    circuit can tell them apart.

    Two circuit features break these degeneracies (both verified empirically
    via finite-difference Jacobian rank/correlation checks and the actual
    fit, not just derived by hand):

    - AZ (a Z-type check, CNOT(data->ancilla) via H-CZ-CZ-H) and AX (an
      X-type check, CNOT(ancilla->data) via H-sandwiched CZs) on the SAME
      pair of data qubits, run SEQUENTIALLY (not interleaved -- interleaving
      them was tried and made the circuit's detectors non-deterministic).
      Because they act on the same qubits, a residual error invisible to one
      check type's own detector becomes visible via the other's, which
      disentangles X/Y-type error components from pure-Z-type ones.
    - AZbig, a SECOND, independent Z-type check with a different weight (4
      CZ legs instead of AZ's 2), run on its own fresh data qubits. A
      weight-w check's own-detector flip probability is, to leading order,
      p_prep_Z + w*(ancilla-local 2Q components) + ...; two different w's
      give two linearly independent equations, which is what actually
      separates p_prep_Z from p_cnot_ZI (still highly correlated afterward,
      but no longer EXACTLY degenerate -- Jacobian correlation drops from
      1.0000 to ~0.95).

    Requirements satisfied: (1) 10 detectors / 11 co-occurring pairs = 21
    moment equations for 23 free params -- not literally more, but Jacobian-
    verified to no longer have an exact degeneracy along any checked field's
    direction (real Willow circuits have hundreds of detectors; this only
    needs to be "clearly enough" for the empirically-checked fields, not
    exhaustively rank-23). (2) AZ/AX/AZbig's reset and measure ticks each
    carry a ``DEPOLARIZE1`` on the OTHER (spectator) qubits, so
    ``p_idle_spam_*`` has a real effect. (3) The bulk tick between each
    ancilla's closing H and its own M has no R/M, so ``p_idle_cnot_*`` has a
    real, separate effect. (4) AZ/AX/AZbig each have CZ+DEPOLARIZE2 sites (8
    total per round). (5) 9 qubits, 3 rounds -- compact, though the resulting
    fit (23-dim L-BFGS-B over a genuinely non-trivial landscape) takes on the
    order of a minute per call, not milliseconds.

    Qubits: AZ=0, AX=1, D0=2, D1=3 (weight-2 block, AZ/AX share D0,D1);
    AZbig=4, D2=5, D3=6, D4=7, D5=8 (weight-4 block).
    """
    NROUNDS = 3
    all_qubits = list(range(9))
    lines: list[str] = []

    def az_small(other_idle):
        lines.append("R 0")
        lines.append("X_ERROR(0.002) 0")
        lines.append("DEPOLARIZE1(0.0001) " + " ".join(str(q) for q in other_idle))
        lines.append("TICK")
        lines.append("H 0")
        lines.append("TICK")
        lines.append("CZ 0 2")
        lines.append("DEPOLARIZE2(0.001) 0 2")
        lines.append("TICK")
        lines.append("CZ 0 3")
        lines.append("DEPOLARIZE2(0.001) 0 3")
        lines.append("TICK")
        lines.append("H 0")
        lines.append("TICK")

    def ax_small(other_idle):
        lines.append("R 1")
        lines.append("X_ERROR(0.002) 1")
        lines.append("DEPOLARIZE1(0.0001) " + " ".join(str(q) for q in other_idle))
        lines.append("TICK")
        lines.append("H 1")
        lines.append("TICK")
        lines.append("H 2")
        lines.append("TICK")
        lines.append("CZ 1 2")
        lines.append("DEPOLARIZE2(0.001) 1 2")
        lines.append("TICK")
        lines.append("H 2")
        lines.append("TICK")
        lines.append("H 3")
        lines.append("TICK")
        lines.append("CZ 1 3")
        lines.append("DEPOLARIZE2(0.001) 1 3")
        lines.append("TICK")
        lines.append("H 3")
        lines.append("TICK")
        lines.append("H 1")
        lines.append("TICK")

    def az_big(other_idle):
        lines.append("R 4")
        lines.append("X_ERROR(0.002) 4")
        lines.append("DEPOLARIZE1(0.0001) " + " ".join(str(q) for q in other_idle))
        lines.append("TICK")
        lines.append("H 4")
        lines.append("TICK")
        for d in (5, 6, 7, 8):
            lines.append(f"CZ 4 {d}")
            lines.append(f"DEPOLARIZE2(0.001) 4 {d}")
            lines.append("TICK")
        lines.append("H 4")
        lines.append("TICK")

    for r in range(NROUNDS):
        az_small([q for q in all_qubits if q != 0])
        lines.append("DEPOLARIZE1(0.0001) " + " ".join(str(q) for q in all_qubits))
        lines.append("TICK")
        lines.append("M(0.005) 0")
        lines.append("DEPOLARIZE1(0.0001) " + " ".join(str(q) for q in all_qubits if q != 0))
        lines.append(f"DETECTOR(0, 0, {r}) rec[-1]" if r == 0 else f"DETECTOR(0, 0, {r}) rec[-1] rec[-4]")
        lines.append("TICK")

        ax_small([q for q in all_qubits if q != 1])
        lines.append("DEPOLARIZE1(0.0001) " + " ".join(str(q) for q in all_qubits))
        lines.append("TICK")
        lines.append("M(0.005) 1")
        lines.append("DEPOLARIZE1(0.0001) " + " ".join(str(q) for q in all_qubits if q != 1))
        if r >= 1:
            lines.append(f"DETECTOR(1, 0, {r}) rec[-1] rec[-4]")
        lines.append("TICK")

        az_big([q for q in all_qubits if q != 4])
        lines.append("DEPOLARIZE1(0.0001) " + " ".join(str(q) for q in all_qubits))
        lines.append("TICK")
        lines.append("M(0.005) 4")
        lines.append("DEPOLARIZE1(0.0001) " + " ".join(str(q) for q in all_qubits if q != 4))
        lines.append(f"DETECTOR(4, 0, {r}) rec[-1]" if r == 0 else f"DETECTOR(4, 0, {r}) rec[-1] rec[-4]")
        if r < NROUNDS - 1:
            lines.append("TICK")

    lines.append("TICK")
    lines.append("M 2 3 5 6 7 8")
    # record order: ..., AZ_last(-9), AX_last(-8), AZbig_last(-7),
    # D0(-6) D1(-5) D2(-4) D3(-3) D4(-2) D5(-1)
    lines.append(f"DETECTOR(0, 1, {NROUNDS}) rec[-9] rec[-6] rec[-5]")
    lines.append(f"DETECTOR(4, 1, {NROUNDS}) rec[-7] rec[-4] rec[-3] rec[-2] rec[-1]")

    return stim.Circuit("\n".join(lines) + "\n")


def test_fit_noise_model_recovers_known_ground_truth():
    """End-to-end: build a circuit from a KNOWN NoiseModel, sample real-looking
    detection events, fit, confirm recovered params are close to the known
    ground truth. This is the fit rung's version of syndrome_dem.py's
    ground-truth recovery tests."""
    from ising_sim2real.ingest.noise_injector import inject_noise_model
    from ising_sim2real.ingest.param_fit import fit_noise_model
    from ising_sim2real.paths import ISING_CODE
    import sys

    p = str(ISING_CODE)
    if p not in sys.path:
        sys.path.insert(0, p)
    from qec.noise_model import NoiseModel  # type: ignore

    template = _toy_noise_fit_template()
    ground_truth = NoiseModel(
        p_prep_X=0.0, p_prep_Z=0.008,
        p_meas_X=0.0, p_meas_Z=0.012,
        p_idle_cnot_X=0.0005, p_idle_cnot_Y=0.0005, p_idle_cnot_Z=0.0005,
        p_idle_spam_X=0.001, p_idle_spam_Y=0.001, p_idle_spam_Z=0.001,
        **{f"p_cnot_{k}": 0.0002 for k in
           ("IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ")},
    )
    true_circuit = inject_noise_model(template, ground_truth)
    # approximate_disjoint_errors=True required once PAULI_CHANNEL_2 sites
    # exist (confirmed against stim in Task 3) -- ground_truth has nonzero
    # 2Q-gate params.
    true_dem = true_circuit.detector_error_model(approximate_disjoint_errors=True)
    sampler = true_dem.compile_sampler(seed=7)
    dets, _obs, _ = sampler.sample(shots=200_000)

    init = NoiseModel.from_single_p(0.005)
    # No flag needed here -- template is the UNMODIFIED SI1000 circuit
    # (DEPOLARIZE2, not PAULI_CHANNEL_2 -- confirmed this doesn't need the flag).
    dem_si1000 = template.detector_error_model()  # reference graph structure
    fitted = fit_noise_model([(dem_si1000, template, dets)], init)

    for field in ("p_prep_Z", "p_meas_Z", "p_idle_cnot_X", "p_idle_spam_X", "p_cnot_XX"):
        assert abs(getattr(fitted, field) - getattr(ground_truth, field)) < 0.003, field


import pytest


GROUND_TRUTH_REGIMES = [
    # (label, overrides on top of a p=0.005-equivalent base)
    ("baseline", {}),
    ("idle-dominated", {
        "p_idle_cnot_X": 0.003, "p_idle_cnot_Y": 0.003, "p_idle_cnot_Z": 0.003,
        "p_idle_spam_X": 0.006, "p_idle_spam_Y": 0.006, "p_idle_spam_Z": 0.006,
    }),
    ("2q-gate-dominated", {
        **{f"p_cnot_{k}": 0.0015 for k in
           ("IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ")},
    }),
    ("spam-dominated", {
        "p_prep_Z": 0.02, "p_meas_Z": 0.03,
    }),
]


@pytest.mark.parametrize("label,overrides", GROUND_TRUTH_REGIMES)
def test_fit_converges_across_representative_regimes(label, overrides):
    """Blocking convergence-robustness gate: the fit must recover EACH of these
    representative ground-truth regimes within tolerance, not just one lucky
    case. A failure on ANY regime is a real optimizer or formula problem, not a
    known limitation to route around."""
    from ising_sim2real.ingest.noise_injector import inject_noise_model
    from ising_sim2real.ingest.param_fit import fit_noise_model
    from ising_sim2real.paths import ISING_CODE
    import sys

    p = str(ISING_CODE)
    if p not in sys.path:
        sys.path.insert(0, p)
    from qec.noise_model import NoiseModel  # type: ignore

    template = _toy_noise_fit_template()
    base = NoiseModel.from_single_p(0.005).to_config_dict()
    base.update(overrides)
    base["p_prep_X"] = base["p_prep_Z"]
    base["p_meas_X"] = base["p_meas_Z"]
    ground_truth = NoiseModel(**base)

    true_circuit = inject_noise_model(template, ground_truth)
    # approximate_disjoint_errors=True required once PAULI_CHANNEL_2 sites
    # exist (confirmed against stim in Task 3) -- every regime here has
    # nonzero 2Q-gate params.
    true_dem = true_circuit.detector_error_model(approximate_disjoint_errors=True)
    sampler = true_dem.compile_sampler(seed=zlib.crc32(label.encode()) & 0x7FFFFFFF)
    dets, _obs, _ = sampler.sample(shots=3_000_000)

    init = NoiseModel.from_single_p(0.005)
    # No flag needed here -- template is the UNMODIFIED SI1000 circuit.
    dem_si1000 = template.detector_error_model()
    fitted = fit_noise_model([(dem_si1000, template, dets)], init)

    for field in ("p_prep_Z", "p_meas_Z", "p_idle_cnot_X", "p_idle_spam_X", "p_cnot_XX"):
        assert abs(getattr(fitted, field) - getattr(ground_truth, field)) < 0.004, (
            f"{label}: {field} recovered={getattr(fitted, field)} "
            f"ground_truth={getattr(ground_truth, field)}"
        )


def _toy_single_check_circuit(nrounds: int) -> stim.Circuit:
    """Minimal single-ancilla-check circuit, parameterized by round count so two
    calls with different ``nrounds`` produce circuits with genuinely different
    detector counts (``nrounds + 1`` detectors: one per round's ancilla
    measurement, plus one final data-basis detector) -- exactly the real-world
    shape that broke ``fit_one_patch``'s ``np.concatenate`` (different round
    counts -> different detector counts). Not meant to be a faithful
    ground-truth-recovery fixture like ``_toy_noise_fit_template`` (no attempt
    to break the AZ/AX/AZbig degeneracies documented there) -- this only needs
    to prove the multi-config plumbing itself runs end-to-end without the
    array-size crash recurring.

    Qubits: ancilla=0, D0=1, D1=2.
    """
    lines: list[str] = []
    for r in range(nrounds):
        lines.append("R 0")
        lines.append("X_ERROR(0.002) 0")
        lines.append("DEPOLARIZE1(0.0001) 1 2")
        lines.append("TICK")
        lines.append("H 0")
        lines.append("TICK")
        lines.append("CZ 0 1")
        lines.append("DEPOLARIZE2(0.001) 0 1")
        lines.append("TICK")
        lines.append("CZ 0 2")
        lines.append("DEPOLARIZE2(0.001) 0 2")
        lines.append("TICK")
        lines.append("H 0")
        lines.append("TICK")
        lines.append("DEPOLARIZE1(0.0001) 0 1 2")
        lines.append("TICK")
        lines.append("M(0.005) 0")
        lines.append("DEPOLARIZE1(0.0001) 1 2")
        if r == 0:
            lines.append(f"DETECTOR(0, 0, {r}) rec[-1]")
        else:
            lines.append(f"DETECTOR(0, 0, {r}) rec[-1] rec[-2]")
        lines.append("TICK")

    lines.append("M 1 2")
    lines.append(f"DETECTOR(0, 1, {nrounds}) rec[-3] rec[-2] rec[-1]")

    return stim.Circuit("\n".join(lines) + "\n")


def test_fit_noise_model_handles_multiple_configs_with_different_detector_counts():
    """Regression test for the exact bug found in Task 10: ``fit_one_patch``
    pooled 14 configs' detection events with ``np.concatenate(axis=0)``, which
    crashes the moment two configs have different detector counts (real
    example: d5 round=1 has 24 detectors, round=10 has 240). Proves
    ``fit_noise_model`` accepts a list of differently-shaped configs and runs
    to completion without that crash -- not a rigorous ground-truth-recovery
    check (see ``test_fit_noise_model_recovers_known_ground_truth`` for that),
    just proof the multi-config plumbing itself works."""
    from ising_sim2real.ingest.noise_injector import inject_noise_model
    from ising_sim2real.ingest.param_fit import fit_noise_model
    from ising_sim2real.paths import ISING_CODE
    import sys

    p = str(ISING_CODE)
    if p not in sys.path:
        sys.path.insert(0, p)
    from qec.noise_model import NoiseModel  # type: ignore

    ground_truth = NoiseModel(
        p_prep_X=0.0, p_prep_Z=0.008,
        p_meas_X=0.0, p_meas_Z=0.012,
        p_idle_cnot_X=0.0005, p_idle_cnot_Y=0.0005, p_idle_cnot_Z=0.0005,
        p_idle_spam_X=0.001, p_idle_spam_Y=0.001, p_idle_spam_Z=0.001,
        **{f"p_cnot_{k}": 0.0002 for k in
           ("IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ")},
    )

    # 2 detectors and 4 detectors respectively -- genuinely different shapes,
    # the exact condition that made np.concatenate crash.
    template_small = _toy_single_check_circuit(nrounds=1)
    template_big = _toy_single_check_circuit(nrounds=3)

    configs = []
    for template, seed in ((template_small, 11), (template_big, 12)):
        true_circuit = inject_noise_model(template, ground_truth)
        true_dem = true_circuit.detector_error_model(approximate_disjoint_errors=True)
        sampler = true_dem.compile_sampler(seed=seed)
        dets, _obs, _ = sampler.sample(shots=200_000)
        dem_si1000 = template.detector_error_model()
        configs.append((dem_si1000, template, dets))

    assert configs[0][2].shape[1] != configs[1][2].shape[1], (
        "toy configs must have different detector counts to exercise the bug"
    )

    init = NoiseModel.from_single_p(0.005)
    fitted = fit_noise_model(configs, init)

    # Loose sanity bounds -- this circuit isn't designed to break parameter
    # degeneracies (see _toy_single_check_circuit docstring), so this is not a
    # tight ground-truth-recovery check, just proof the fit produces a
    # plausible, in-bounds NoiseModel from pooled multi-shape evidence.
    for field in ("p_prep_Z", "p_meas_Z"):
        val = getattr(fitted, field)
        assert 0.0 <= val <= 1.0
        assert abs(val - getattr(ground_truth, field)) < 0.05, (
            f"{field}: recovered={val} ground_truth={getattr(ground_truth, field)}"
        )
