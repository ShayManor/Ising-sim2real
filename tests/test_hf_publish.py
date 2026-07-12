"""Unit test for the si1000_noisy publish-bundle staging logic (offline, no
network -- exercises build_bundle() against a temp directory standing in for
the local raw Willow tree).
"""

from __future__ import annotations

from pathlib import Path

from ising_sim2real.ingest.willow import WillowConfig


def _make_fake_leaf(root: Path, cfg, stem: str) -> Path:
    from scripts.publish_hf_dataset import config_stem
    assert config_stem(cfg) == stem
    leaf = root / f"d{cfg.distance}_at_{cfg.orientation}" / cfg.basis / f"r{cfg.rounds:02d}"
    leaf.mkdir(parents=True)
    (leaf / "circuit_ideal.stim").write_text("QUBIT_COORDS(0, 0) 0\nM 0\n")
    (leaf / "circuit_noisy_si1000.stim").write_text("QUBIT_COORDS(0, 0) 0\nM(0.005) 0\n")
    decoding = leaf / "decoding_results" / "correlated_matching_decoder_with_si1000_prior"
    decoding.mkdir(parents=True)
    (decoding / "error_model.dem").write_text("error(0.01) D0\n")
    return leaf


def test_build_bundle_stages_si1000_noisy_circuit(tmp_path, monkeypatch):
    from scripts.publish_hf_dataset import build_bundle

    cfg = WillowConfig(distance=3, basis="Z", rounds=1, orientation="q4_5")
    stem = f"d{cfg.distance}_at_{cfg.orientation}__{cfg.basis}__r{cfg.rounds:03d}"
    raw_root = tmp_path / "raw"
    _make_fake_leaf(raw_root, cfg, stem)

    monkeypatch.setattr("ising_sim2real.paths.WILLOW_RAW_DIR", raw_root)
    import scripts.publish_hf_dataset as mod
    monkeypatch.setattr(mod, "WILLOW_RAW_DIR", raw_root)

    staging = tmp_path / "staging"
    n = build_bundle(staging, [cfg])

    assert n == 1
    noisy_path = staging / "circuits" / f"{stem}.si1000_noisy.stim"
    assert noisy_path.exists()
    assert noisy_path.read_text() == "QUBIT_COORDS(0, 0) 0\nM(0.005) 0\n"
    # Existing artifacts must still be staged unchanged.
    assert (staging / "circuits" / f"{stem}.stim").exists()
    assert (staging / "dems" / f"{stem}.si1000.dem.gz").exists()
