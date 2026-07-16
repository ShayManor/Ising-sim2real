"""Unit tests for the RQ3 sensitivity-sweep model generator and the fit rung's
FITTED_MODELS_DIR override -- pure JSON/dict logic, no decode or model load."""

from __future__ import annotations

import json

import pytest

from ising_sim2real.paths import ISING_CODE
from scripts.gen_sensitivity_models import (
    FREE_PARAMS,
    load_baseline_models,
    overestimate,
    perturb_param,
    write_set,
)

# A minimal but complete 25-key model dict (values chosen to hit both the x2 and
# the +floor branch of the overestimate rule).
BASE = {
    **{p: 0.0 for p in FREE_PARAMS},
    "p_prep_X": 0.0, "p_meas_X": 0.0,
    "p_cnot_IX": 0.006,  # healthy fit -> x2 branch
    "p_prep_Z": 0.0,     # zero fit -> +floor branch
    "p_meas_Z": 0.004,
}


def test_free_params_count_is_23():
    assert len(FREE_PARAMS) == 23
    assert "p_prep_X" not in FREE_PARAMS and "p_meas_X" not in FREE_PARAMS


def test_overestimate_picks_larger_of_x2_and_floor():
    assert overestimate(0.006, 2.0, 1e-3, 0.5) == pytest.approx(0.012)   # x2 wins
    assert overestimate(0.0, 2.0, 1e-3, 0.5) == pytest.approx(1e-3)      # floor wins
    assert overestimate(0.4, 2.0, 1e-3, 0.5) == pytest.approx(0.5)       # cap clamps


def test_perturb_only_touches_target_param():
    out = perturb_param(BASE, "p_cnot_IX", 2.0, 1e-3, 0.5)
    assert out["p_cnot_IX"] == pytest.approx(0.012)
    for k, v in BASE.items():
        if k != "p_cnot_IX":
            assert out[k] == v  # everything else unchanged


def test_perturb_ties_x_copy_for_spam():
    out = perturb_param(BASE, "p_meas_Z", 2.0, 1e-3, 0.5)
    assert out["p_meas_Z"] == pytest.approx(0.008)
    assert out["p_meas_X"] == out["p_meas_Z"]  # tied along


def test_generate_and_reload_roundtrips(tmp_path):
    write_set(tmp_path / "baseline", {"d3_q4_5": BASE})
    reloaded = load_baseline_models(tmp_path / "baseline")
    assert reloaded["d3_q4_5"] == BASE


@pytest.mark.skipif(not ISING_CODE.exists(), reason="vendored Ising-Decoding code absent")
def test_fitted_models_dir_override(tmp_path, monkeypatch):
    from ising_sim2real.ingest import synthetic
    (tmp_path / "d3_q4_5.json").write_text(json.dumps(BASE))
    monkeypatch.setenv("FITTED_MODELS_DIR", str(tmp_path))
    nm = synthetic._load_fitted_noise_model("d3_q4_5")
    assert nm.p_cnot_IX == pytest.approx(0.006)
    assert nm.p_meas_Z == pytest.approx(0.004)


@pytest.mark.skipif(not ISING_CODE.exists(), reason="vendored Ising-Decoding code absent")
def test_explicit_models_dir_beats_env(tmp_path, monkeypatch):
    """2x2 decomposition contract: an explicit models_dir loads the prior from a
    different dir than FITTED_MODELS_DIR (the sampling source)."""
    from ising_sim2real.ingest import synthetic
    source, prior = tmp_path / "source", tmp_path / "prior"
    source.mkdir(); prior.mkdir()
    (source / "d3_q4_5.json").write_text(json.dumps(BASE))
    (prior / "d3_q4_5.json").write_text(json.dumps({**BASE, "p_meas_Z": 0.099}))
    monkeypatch.setenv("FITTED_MODELS_DIR", str(source))
    # no models_dir -> env (source)
    assert synthetic._load_fitted_noise_model("d3_q4_5").p_meas_Z == pytest.approx(0.004)
    # explicit models_dir (prior) overrides the env
    nm = synthetic._load_fitted_noise_model("d3_q4_5", models_dir=str(prior))
    assert nm.p_meas_Z == pytest.approx(0.099)
