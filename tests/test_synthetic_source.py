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
