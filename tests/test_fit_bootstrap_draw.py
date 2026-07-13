import json

import pytest

from scripts import fit_bootstrap_draw


PATCHES = [("q1_2", 3), ("q3_4", 3), ("q5_6", 5), ("q6_7", 7)]  # 4 fake patches


def test_tile_decodes_draw_and_patch():
    # index 0..3 -> draw 0; index 4..7 -> draw 1; patch cycles every len(patches)
    assert fit_bootstrap_draw.tile(0, PATCHES) == (0, "q1_2", 3)
    assert fit_bootstrap_draw.tile(3, PATCHES) == (0, "q6_7", 7)
    assert fit_bootstrap_draw.tile(4, PATCHES) == (1, "q1_2", 3)
    assert fit_bootstrap_draw.tile(9, PATCHES) == (2, "q3_4", 3)


def test_run_index_writes_named_json(tmp_path, monkeypatch):
    monkeypatch.setattr(fit_bootstrap_draw, "_all_patches", lambda repo: PATCHES)
    monkeypatch.setattr(fit_bootstrap_draw, "OUT_ROOT", tmp_path)
    captured = {}

    def fake_fit(orientation, distance, repo=None, draw=None):
        captured["args"] = (orientation, distance, draw)
        return {"p_cnot_XX": 0.001, "p_meas_Z": 0.02}

    monkeypatch.setattr(fit_bootstrap_draw, "fit_one_patch", fake_fit)

    fit_bootstrap_draw.run_index(4, repo="fake")

    assert captured["args"] == ("q1_2", 3, 1)  # draw 1, first patch
    out = tmp_path / "draw_01" / "d3_q1_2.json"
    assert out.exists()
    assert json.loads(out.read_text())["p_meas_Z"] == 0.02


def test_resume_skips_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(fit_bootstrap_draw, "_all_patches", lambda repo: PATCHES)
    monkeypatch.setattr(fit_bootstrap_draw, "OUT_ROOT", tmp_path)
    out = tmp_path / "draw_00" / "d3_q1_2.json"
    out.parent.mkdir(parents=True)
    out.write_text('{"kept": true}')

    def boom(*a, **k):
        raise AssertionError("fit_one_patch must not run on resume-skip")

    monkeypatch.setattr(fit_bootstrap_draw, "fit_one_patch", boom)

    fit_bootstrap_draw.run_index(0, repo="fake", resume=True)

    assert json.loads(out.read_text()) == {"kept": True}  # untouched
