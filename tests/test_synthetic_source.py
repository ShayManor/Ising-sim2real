"""Sanity gate for the synthetic noise rungs (spec S7): DEM layout, sample shapes,
and mwpm on si1000-sampled events landing below the real-row mwpm LER.

Needs network access to the published HF dataset (circuit + shipped DEM fetch);
skips if unreachable.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from ising_sim2real.decoders.pymatching_decoder import PyMatchingDecoder
from ising_sim2real.ingest.hf import discover_configs_hf
from ising_sim2real.ingest.synthetic import build_rung_dem, sample_config
from ising_sim2real.metrics import logical_error_per_cycle, logical_error_rate

REAL_ALL = Path(__file__).resolve().parent.parent / "results" / "willow_real" / "eval_all.csv"


@pytest.fixture(scope="module")
def cfg():
    try:
        configs = discover_configs_hf()
    except Exception as exc:  # noqa: BLE001 -- network/auth flake, not a code bug
        pytest.skip(f"HF dataset unreachable: {exc}")
    for c in configs:
        if c.distance == 5 and c.basis == "Z" and c.rounds == 10:
            return c
    pytest.skip("no d5/Z/r10 config found")


def test_si1000_rung_dem_matches_shipped_layout(cfg):
    data = sample_config(cfg, rung="si1000", p=2e-3, shots=500, seed=1234)
    assert data.detectors.shape == (500, data.dem_si1000.num_detectors)
    assert data.observables.shape == (500, data.dem_si1000.num_observables)
    assert data.dem_si1000.num_detectors == data.circuit.num_detectors


def test_uniform_rung_keeps_graph_shape(cfg):
    si1000 = sample_config(cfg, rung="si1000", p=2e-3, shots=10, seed=1).dem_si1000
    uniform_dem = build_rung_dem("uniform", si1000, p=2e-3)
    assert uniform_dem.num_detectors == si1000.num_detectors
    assert uniform_dem.num_observables == si1000.num_observables

    data = sample_config(cfg, rung="uniform", p=2e-3, shots=500, seed=1234)
    assert data.detectors.shape == (500, si1000.num_detectors)
    assert data.observables.shape == (500, si1000.num_observables)


def test_unknown_rung_rejected():
    with pytest.raises(ValueError):
        build_rung_dem("bogus", None, 2e-3)  # type: ignore[arg-type]


def test_si1000_mwpm_below_real_row(cfg):
    """Synthetic si1000-sampled events must decode cleaner than real hardware."""
    if not REAL_ALL.exists():
        pytest.skip(f"{REAL_ALL} not present locally")

    data = sample_config(cfg, rung="si1000", p=2e-3, shots=5000, seed=1234)
    res = PyMatchingDecoder.from_dem(data.dem_si1000).decode_batch(data.detectors)
    ler = logical_error_rate(res.predictions, data.observables)
    assert 0.0 <= ler < 0.5
    perc = logical_error_per_cycle(ler, cfg.rounds)

    real_perc = []
    with REAL_ALL.open(newline="") as f:
        for row in csv.DictReader(f):
            if (row["decoder"] == "mwpm" and int(row["distance"]) == cfg.distance
                    and row["basis"] == cfg.basis and int(row["rounds"]) == cfg.rounds):
                try:
                    real_perc.append(float(row["ler_per_cycle"]))
                except ValueError:
                    continue
    if not real_perc:
        pytest.skip("no matching real-row mwpm rows to compare against")
    assert perc <= min(real_perc), (
        f"synthetic si1000 mwpm per-cycle LER {perc} not below real-row {min(real_perc)}"
    )


def test_sampling_is_seed_reproducible(cfg):
    a = sample_config(cfg, rung="si1000", p=2e-3, shots=200, seed=42)
    b = sample_config(cfg, rung="si1000", p=2e-3, shots=200, seed=42)
    assert np.array_equal(a.detectors, b.detectors)
    assert np.array_equal(a.observables, b.observables)


def test_syndrome_rung_dem_matches_shipped_layout(cfg):
    data = sample_config(cfg, rung="syndrome", p=2e-3, shots=500, seed=1234)
    assert data.detectors.shape == (500, data.dem_si1000.num_detectors)
    assert data.observables.shape == (500, data.dem_si1000.num_observables)
    assert data.dem_si1000.num_detectors == data.circuit.num_detectors


def test_syndrome_rung_probabilities_in_unit_interval(cfg):
    data = sample_config(cfg, rung="syndrome", p=2e-3, shots=10, seed=1234)
    for instr in data.dem_si1000.flattened():
        if instr.type == "error":
            prob = instr.args_copy()[0]
            assert 0.0 <= prob <= 1.0


def test_syndrome_rung_requires_detection_events():
    with pytest.raises(ValueError):
        build_rung_dem("syndrome", None, 2e-3)  # type: ignore[arg-type]


def test_cli_accepts_syndrome_rung_choice(monkeypatch):
    from ising_sim2real.eval import runner as runner_module
    # Tests the REAL parser inside runner.main() (not a duplicate/copy of it),
    # without touching the network: monkeypatch evaluate() to a stub that
    # returns no rows, so main() runs its full argparse validation (including
    # the `choices=(...)` check on --rung) and then exits cleanly via the
    # "no configs matched" path (return code 1) instead of ever calling
    # discover_configs_hf(). If "syndrome" were not a valid --rung choice,
    # argparse would raise SystemExit(2) before evaluate() is ever reached.
    monkeypatch.setattr(runner_module, "evaluate", lambda args: [])
    rc = runner_module.main(["--rung", "syndrome", "--source", "synth"])
    assert rc == 1  # "no configs matched the filters" -- proves args parsed fine


def test_syndrome_mwpm_below_real_row(cfg):
    """Syndrome-estimated-DEM-sampled events must decode cleaner than real
    hardware -- same S7 validation gate as the si1000/uniform rungs.
    """
    if not REAL_ALL.exists():
        pytest.skip(f"{REAL_ALL} not present locally")

    data = sample_config(cfg, rung="syndrome", p=2e-3, shots=5000, seed=1234)
    res = PyMatchingDecoder.from_dem(data.dem_si1000).decode_batch(data.detectors)
    ler = logical_error_rate(res.predictions, data.observables)
    assert 0.0 <= ler < 0.5
    perc = logical_error_per_cycle(ler, cfg.rounds)

    real_perc = []
    with REAL_ALL.open(newline="") as f:
        for row in csv.DictReader(f):
            if (row["decoder"] == "mwpm" and int(row["distance"]) == cfg.distance
                    and row["basis"] == cfg.basis and int(row["rounds"]) == cfg.rounds):
                try:
                    real_perc.append(float(row["ler_per_cycle"]))
                except ValueError:
                    continue
    if not real_perc:
        pytest.skip("no matching real-row mwpm rows to compare against")
    assert perc <= min(real_perc), (
        f"synthetic syndrome mwpm per-cycle LER {perc} not below real-row {min(real_perc)}"
    )


def test_syndrome_sampling_is_seed_reproducible(cfg):
    a = sample_config(cfg, rung="syndrome", p=2e-3, shots=200, seed=42)
    b = sample_config(cfg, rung="syndrome", p=2e-3, shots=200, seed=42)
    assert np.array_equal(a.detectors, b.detectors)
    assert np.array_equal(a.observables, b.observables)

    # Task 7: also confirm the estimated-DEM output itself (not just the
    # resampled shots) is bit-identical run-to-run on real data -- pins that
    # estimate_dem_from_syndromes has no hidden nondeterminism (e.g. an
    # unseeded bootstrap call) independent of the sampling step.
    a_probs = np.array(
        [instr.args_copy()[0] for instr in a.dem_si1000.flattened() if instr.type == "error"]
    )
    b_probs = np.array(
        [instr.args_copy()[0] for instr in b.dem_si1000.flattened() if instr.type == "error"]
    )
    assert np.array_equal(a_probs, b_probs)


def test_fit_rung_requires_patch():
    with pytest.raises(ValueError):
        build_rung_dem("fit", None, 2e-3)  # type: ignore[arg-type]


def test_fit_rung_dem_matches_shipped_layout(cfg):
    from ising_sim2real.ingest.willow import patch_key

    fitted_dir = Path(__file__).resolve().parents[1] / "results" / "fitted_noise_models"
    key = patch_key(cfg.distance, cfg.orientation)
    if not (fitted_dir / f"{key}.json").exists():
        pytest.skip(f"no fitted model for patch {key} -- run scripts/fit_noise_models.py first")
    data = sample_config(cfg, rung="fit", p=2e-3, shots=500, seed=1234)
    assert data.detectors.shape == (500, data.dem_si1000.num_detectors)
    assert data.observables.shape == (500, data.dem_si1000.num_observables)


def test_cli_accepts_fit_rung_choice(monkeypatch):
    from ising_sim2real.eval import runner as runner_module
    monkeypatch.setattr(runner_module, "evaluate", lambda args: [])
    rc = runner_module.main(["--rung", "fit", "--source", "synth"])
    assert rc == 1


def test_fit_mwpm_reasonably_close_to_real_row(cfg):
    """Validation gate for the fit rung: unlike the purely-synthetic rungs
    (which must decode CLEANER than real hardware), a rung whose noise model is
    FIT to be realistic should land close to the real row, not necessarily below
    it -- see the design spec's Validation Gate section."""
    from ising_sim2real.ingest.willow import patch_key

    key = patch_key(cfg.distance, cfg.orientation)
    fitted_path = Path(__file__).resolve().parents[1] / "results" / "fitted_noise_models" / f"{key}.json"
    if not fitted_path.exists():
        pytest.skip(f"{fitted_path} not present -- run scripts/fit_noise_models.py --patch {cfg.orientation} first")
    if not REAL_ALL.exists():
        pytest.skip(f"{REAL_ALL} not present locally")
    if cfg.rounds not in (110, 130, 150, 170, 190, 210, 230, 250):
        pytest.skip("fit rung validation only meaningful on eval-set (held-out) rounds")

    data = sample_config(cfg, rung="fit", p=2e-3, shots=5000, seed=1234)
    res = PyMatchingDecoder.from_dem(data.dem_si1000).decode_batch(data.detectors)
    ler = logical_error_rate(res.predictions, data.observables)
    assert 0.0 <= ler < 0.5
    perc = logical_error_per_cycle(ler, cfg.rounds)

    real_perc = []
    with REAL_ALL.open(newline="") as f:
        for row in csv.DictReader(f):
            if (row["decoder"] == "mwpm" and int(row["distance"]) == cfg.distance
                    and row["basis"] == cfg.basis and int(row["rounds"]) == cfg.rounds):
                try:
                    real_perc.append(float(row["ler_per_cycle"]))
                except ValueError:
                    continue
    if not real_perc:
        pytest.skip("no matching real-row mwpm rows to compare against")
    # "Reasonably close" per the design spec: same order of magnitude, not a
    # strict inequality like the idealized synthetic rungs.
    ratio = perc / min(real_perc) if min(real_perc) > 0 else float("inf")
    assert 0.1 <= ratio <= 10.0, (
        f"fit-rung mwpm per-cycle LER {perc} vs real-row {min(real_perc)} "
        f"(ratio {ratio}) -- outside the 'reasonably close' band"
    )


def test_fit_sampling_is_seed_reproducible(cfg):
    from ising_sim2real.ingest.willow import patch_key

    key = patch_key(cfg.distance, cfg.orientation)
    fitted_path = Path(__file__).resolve().parents[1] / "results" / "fitted_noise_models" / f"{key}.json"
    if not fitted_path.exists():
        pytest.skip(f"{fitted_path} not present -- run scripts/fit_noise_models.py --patch {cfg.orientation} first")
    a = sample_config(cfg, rung="fit", p=2e-3, shots=200, seed=42)
    b = sample_config(cfg, rung="fit", p=2e-3, shots=200, seed=42)
    assert np.array_equal(a.detectors, b.detectors)
    assert np.array_equal(a.observables, b.observables)
