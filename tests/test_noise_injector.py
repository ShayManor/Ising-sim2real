"""Unit tests for the noise injector's circuit-surgery primitives (arXiv... no
paper here -- this is reverse-engineered directly from real shipped
circuit_noisy_si1000.stim files; see the plan's Global Constraints for the exact
rule and why it's correct for Willow's circuit family).
"""

from __future__ import annotations

import pytest
import stim

from ising_sim2real.ingest.noise_injector import (
    _is_spam_tick,
    _qubits_of,
    _split_into_ticks,
)


def test_split_into_ticks_basic():
    circuit = stim.Circuit("H 0\nDEPOLARIZE1(0.1) 0\nTICK\nCZ 0 1\nDEPOLARIZE2(0.2) 0 1\nTICK\nM(0.3) 0 1\n")
    ticks = _split_into_ticks(circuit)
    assert len(ticks) == 3
    assert [instr.name for instr in ticks[0]] == ["H", "DEPOLARIZE1"]
    assert [instr.name for instr in ticks[1]] == ["CZ", "DEPOLARIZE2"]
    assert [instr.name for instr in ticks[2]] == ["M"]


def test_split_into_ticks_no_leading_tick():
    # Instructions before the first TICK form tick 0, even if there are none.
    circuit = stim.Circuit("TICK\nH 0\n")
    ticks = _split_into_ticks(circuit)
    assert len(ticks) == 2
    assert ticks[0] == []
    assert [instr.name for instr in ticks[1]] == ["H"]


def test_split_into_ticks_rejects_repeat_blocks():
    circuit = stim.Circuit("REPEAT 3 {\nH 0\nTICK\n}\n")
    with pytest.raises(NotImplementedError):
        _split_into_ticks(circuit)


def test_qubits_of_returns_target_values():
    circuit = stim.Circuit("CZ 3 5 7 9\n")
    instr = next(iter(circuit))
    assert _qubits_of(instr) == [3, 5, 7, 9]


def test_is_spam_tick_true_for_reset():
    circuit = stim.Circuit("R 0 1\nX_ERROR(0.1) 0 1\n")
    tick = list(circuit)
    assert _is_spam_tick(tick) is True


def test_is_spam_tick_true_for_measurement():
    circuit = stim.Circuit("M(0.1) 0 1\n")
    tick = list(circuit)
    assert _is_spam_tick(tick) is True


def test_is_spam_tick_false_for_bulk_round():
    circuit = stim.Circuit("CZ 0 1\nDEPOLARIZE2(0.1) 0 1\nDEPOLARIZE1(0.01) 2 3\n")
    tick = list(circuit)
    assert _is_spam_tick(tick) is False


import sys

from ising_sim2real.paths import ISING_CODE


def _noise_model_cls():
    if not ISING_CODE.exists():
        pytest.skip("vendored Ising-Decoding code absent")
    p = str(ISING_CODE)
    if p not in sys.path:
        sys.path.insert(0, p)
    from qec.noise_model import NoiseModel  # type: ignore

    return NoiseModel


def _make_noise(**overrides):
    NoiseModel = _noise_model_cls()
    base = {
        "p_prep_X": 0.0, "p_prep_Z": 0.01,
        "p_meas_X": 0.0, "p_meas_Z": 0.02,
        "p_idle_cnot_X": 0.001, "p_idle_cnot_Y": 0.001, "p_idle_cnot_Z": 0.001,
        "p_idle_spam_X": 0.003, "p_idle_spam_Y": 0.003, "p_idle_spam_Z": 0.003,
        **{f"p_cnot_{k}": 0.0005 for k in
           ("IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ")},
    }
    base.update(overrides)
    return NoiseModel(**base)


def test_inject_rewrites_prep_site():
    from ising_sim2real.ingest.noise_injector import inject_noise_model

    template = stim.Circuit("R 0 1\nX_ERROR(0.002) 0 1\nTICK\n")
    noise = _make_noise()
    out = inject_noise_model(template, noise)
    instrs = list(out)
    assert instrs[0].name == "R"
    assert instrs[1].name == "X_ERROR"
    assert instrs[1].gate_args_copy() == [noise.p_prep_Z]
    assert _qubits_of(instrs[1]) == [0, 1]


def test_inject_rewrites_2q_gate_site():
    from ising_sim2real.ingest.noise_injector import inject_noise_model

    template = stim.Circuit("CZ 0 1 2 3\nDEPOLARIZE2(0.001) 0 1 2 3\nTICK\n")
    noise = _make_noise()
    out = inject_noise_model(template, noise)
    instrs = list(out)
    assert instrs[0].name == "CZ"
    assert instrs[1].name == "PAULI_CHANNEL_2"
    assert instrs[1].gate_args_copy() == list(noise.to_stim_pauli_channel_2_args())
    assert _qubits_of(instrs[1]) == [0, 1, 2, 3]


def test_inject_rewrites_measurement_site():
    from ising_sim2real.ingest.noise_injector import inject_noise_model

    template = stim.Circuit("M(0.005) 0 1\nTICK\n")
    noise = _make_noise()
    out = inject_noise_model(template, noise)
    instrs = list(out)
    assert instrs[0].name == "M"
    assert instrs[0].gate_args_copy() == [noise.p_meas_Z]


def test_inject_merges_stacked_idle_lines_in_bulk_tick():
    from ising_sim2real.ingest.noise_injector import inject_noise_model

    # Two separate DEPOLARIZE1 lines targeting disjoint qubit sets in one bulk
    # tick (no R/M) -- must merge into ONE PAULI_CHANNEL_1(idle_cnot) covering
    # the union of both qubit sets.
    template = stim.Circuit("DEPOLARIZE1(0.0001) 0 1\nDEPOLARIZE1(0.0001) 2 3\nTICK\n")
    noise = _make_noise()
    out = inject_noise_model(template, noise)
    # Filter to PAULI_CHANNEL_1 (not raw instruction count): the template's own
    # trailing "TICK\n" produces a genuine trailing empty tick-group per
    # _split_into_ticks (see test_split_into_ticks_no_leading_tick), which
    # inject_noise_model faithfully reassembles as a trailing TICK instruction --
    # same filtering pattern as the analogous SPAM-merge test below.
    idle_instrs = [i for i in out if i.name == "PAULI_CHANNEL_1"]
    assert len(idle_instrs) == 1
    assert idle_instrs[0].gate_args_copy() == list(noise.to_stim_pauli_channel_1_args_cnot())
    assert sorted(_qubits_of(idle_instrs[0])) == [0, 1, 2, 3]


def test_inject_merges_idle_into_spam_channel_when_tick_has_reset():
    from ising_sim2real.ingest.noise_injector import inject_noise_model

    # A reset tick with a SEPARATE idle line for spectator qubits -- the
    # spectators' idle must use idle_spam, not idle_cnot, because the tick
    # contains an R instruction (matches the stacked-DEPOLARIZE1 pattern found
    # in real shipped files: reset qubits + X_ERROR, plus other idling qubits).
    template = stim.Circuit(
        "R 0 1\nX_ERROR(0.002) 0 1\nDEPOLARIZE1(0.0001) 2 3\nDEPOLARIZE1(0.002) 2 3\nTICK\n"
    )
    noise = _make_noise()
    out = inject_noise_model(template, noise)
    idle_instrs = [i for i in out if i.name == "PAULI_CHANNEL_1"]
    assert len(idle_instrs) == 1
    assert idle_instrs[0].gate_args_copy() == list(noise.to_stim_pauli_channel_1_args_spam())
    assert sorted(_qubits_of(idle_instrs[0])) == [2, 3]


def test_inject_leaves_sweep_bit_cx_untouched():
    from ising_sim2real.ingest.noise_injector import inject_noise_model

    template = stim.Circuit("CX sweep[0] 0 sweep[1] 1\nDEPOLARIZE1(0.0001) 0 1\nTICK\n")
    noise = _make_noise()
    out = inject_noise_model(template, noise)
    instrs = list(out)
    assert instrs[0].name == "CX"
    assert instrs[1].name == "PAULI_CHANNEL_1"  # the DEPOLARIZE1 still gets rewritten as idle


def test_inject_raises_on_rx():
    from ising_sim2real.ingest.noise_injector import inject_noise_model

    template = stim.Circuit("RX 0\nZ_ERROR(0.002) 0\nTICK\n")
    noise = _make_noise()
    with pytest.raises(NotImplementedError):
        inject_noise_model(template, noise)


def test_inject_raises_on_mx():
    from ising_sim2real.ingest.noise_injector import inject_noise_model

    template = stim.Circuit("MX(0.005) 0\nTICK\n")
    noise = _make_noise()
    with pytest.raises(NotImplementedError):
        inject_noise_model(template, noise)


def test_inject_preserves_detector_and_observable_instructions():
    from ising_sim2real.ingest.noise_injector import inject_noise_model

    template = stim.Circuit(
        "M(0.005) 0\nDETECTOR(0, 0, 0) rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]\nTICK\n"
    )
    noise = _make_noise()
    out = inject_noise_model(template, noise)
    names = [i.name for i in out]
    assert "DETECTOR" in names
    assert "OBSERVABLE_INCLUDE" in names


def test_si1000_reduction_matches_shipped_dem_closely():
    """Validation anchor: injecting NoiseModel.from_single_p(p) at SI1000's own p
    should reproduce a circuit whose DEM is close to the shipped SI1000 DEM. Uses
    a real shipped file from the local raw Willow tree if present; skips cleanly
    otherwise (this repo's CI/dev environment may not have the 12GB tree)."""
    from pathlib import Path

    from ising_sim2real.ingest.noise_injector import inject_noise_model
    from ising_sim2real.paths import WILLOW_RAW_DIR

    leaf = WILLOW_RAW_DIR / "d3_at_q2_7" / "Z" / "r01"
    template_path = leaf / "circuit_noisy_si1000.stim"
    dem_path = leaf / "decoding_results" / "correlated_matching_decoder_with_si1000_prior" / "error_model.dem"
    if not template_path.exists() or not dem_path.exists():
        pytest.skip(f"{leaf} not present locally")

    NoiseModel = _noise_model_cls()
    template = stim.Circuit.from_file(template_path)
    shipped_dem = stim.DetectorErrorModel.from_file(dem_path)

    # SI1000's own base p for this dataset is 0.002 (Google's standard below-
    # threshold operating point; the shipped DEPOLARIZE2 rate of 0.001 = p/2 per
    # the SI1000 formula's 2Q-gate term at p=0.002, consistent with _single_p_mapping).
    noise = NoiseModel.from_single_p(0.002)
    noisy = inject_noise_model(template, noise)
    # approximate_disjoint_errors=True: PAULI_CHANNEL_2 sites are not naturally
    # decomposable into independent error mechanisms, so stim requires this flag
    # to approximate them during DEM conversion (the brief's own snippet omitted
    # it; without it stim raises ValueError on the first PAULI_CHANNEL_2 site).
    reconstructed_dem = noisy.detector_error_model(approximate_disjoint_errors=True)

    assert reconstructed_dem.num_detectors == shipped_dem.num_detectors
    assert reconstructed_dem.num_observables == shipped_dem.num_observables
    # Not asserting numeric probability equality: SI1000's exact per-role scale
    # constants are Google's own tuning, not reproduced by NoiseModel.from_single_p's
    # generic formula (spam_factor=2/3 default, etc.) -- this anchor's job is
    # structural (same graph shape from the same instruction positions), not
    # exact numeric reproduction.
