#!/usr/bin/env python3
"""Fit one 25-param NoiseModel per Willow patch (14 total) from real detection-
event statistics, and write the results to
results/fitted_noise_models/<patch_key>.json.

Run ONCE, locally, before any RUNG=fit synthetic-eval submission (this is data
science / fitting, not raw-data production, per CLAUDE.md's "SLURM jobs compute
RAW DATA ONLY" split -- this script is NOT a SLURM job).

Held-out split (per patch, avoids circularity between fitting and the ladder's
eval-set synthetic data -- see docs/superpowers/specs/2026-07-09-25-param-fit-rung-design.md):
    fit-set rounds:  {1, 10, 13, 30, 50, 70, 90}     (both bases pooled)
    eval-set rounds: {110, 130, 150, 170, 190, 210, 230, 250}

Usage:
    python scripts/fit_noise_models.py                  # all 14 patches
    python scripts/fit_noise_models.py --patch q4_5      # patches with this
                                                          # orientation (may be
                                                          # >1 distance, e.g. q6_7)
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib

from ising_sim2real.ingest.hf import (
    DEFAULT_HF_REPO,
    discover_configs_hf,
    fetch_si1000_noisy_circuit,
    load_config_from_hf,
)
from ising_sim2real.ingest.param_fit import fit_noise_model, resample_shots
from ising_sim2real.ingest.willow import WillowConfig, patch_key
from ising_sim2real.paths import ISING_CODE, REPO_ROOT

STD_ROUNDS = (1, 10, 13, 30, 50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250)
FIT_ROUNDS = (1, 10, 13, 30, 50, 70, 90)
EVAL_ROUNDS = (110, 130, 150, 170, 190, 210, 230, 250)

OUT_DIR = REPO_ROOT / "results" / "fitted_noise_models"


def _all_patches(repo: str = DEFAULT_HF_REPO) -> list[tuple[str, int]]:
    """(orientation, distance) pairs from the live HF config listing. May
    contain duplicate orientations across distances (e.g. "q6_7" at both d3 and
    d7) -- callers must key artifacts by patch_key(distance, orientation),
    never orientation alone."""
    seen: set[tuple[str, int]] = set()
    for cfg in discover_configs_hf(repo):
        seen.add((cfg.orientation, cfg.distance))
    return sorted(seen)


def fit_set_configs_for_patch(orientation: str, distance: int) -> list[WillowConfig]:
    return [
        WillowConfig(distance=distance, basis=b, rounds=r, orientation=orientation)
        for b in ("X", "Z")
        for r in FIT_ROUNDS
    ]


def _ensure_noise_model_cls():
    p = str(ISING_CODE)
    if p not in sys.path:
        sys.path.insert(0, p)
    from qec.noise_model import NoiseModel  # type: ignore

    return NoiseModel


def fit_one_patch(orientation: str, distance: int, repo: str = DEFAULT_HF_REPO,
                  draw: int | None = None) -> dict:
    """Fit against the patch's fit-set configs' real detection events, one
    circuit/DEM/detection-events triple per config (different round counts
    have structurally different circuits and detector counts -- each config
    needs its OWN template and DEM, not a shared one), return the fitted
    NoiseModel's canonical parameter dict. When ``draw`` is set, each config's detectors are shot-resampled first -- the joint-bootstrap refit."""
    NoiseModel = _ensure_noise_model_cls()
    configs = fit_set_configs_for_patch(orientation, distance)

    per_config = []
    for cfg in configs:
        data = load_config_from_hf(cfg, repo=repo)
        template = fetch_si1000_noisy_circuit(cfg, repo=repo)
        detectors = data.detectors
        if draw is not None:
            basis_int = 0 if cfg.basis == "X" else 1
            detectors = resample_shots(detectors, [draw, distance, cfg.rounds, basis_int,
                                                    zlib.crc32(orientation.encode())])
        per_config.append((data.dem_si1000, template, detectors))

    init = NoiseModel.from_single_p(0.002)
    fitted = fit_noise_model(per_config, init)
    return fitted.canonical_parameters()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--patch", default=None,
                     help="fit only patches with this orientation label "
                          "(debugging; may match >1 distance, e.g. q6_7)")
    ap.add_argument("--repo", default=DEFAULT_HF_REPO)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_patches = _all_patches(args.repo)
    targets = [(o, d) for o, d in all_patches if args.patch is None or o == args.patch]
    if not targets:
        raise SystemExit(f"no patches matched --patch={args.patch!r}")

    for orientation, distance in targets:
        key = patch_key(distance, orientation)
        print(f"fitting {key} ...")
        params = fit_one_patch(orientation, distance, repo=args.repo)
        out_path = OUT_DIR / f"{key}.json"
        out_path.write_text(json.dumps(params, sort_keys=True, indent=2))
        print(f"  -> {out_path}")

    print(f"done: {len(targets)} patch(es) fitted -> {OUT_DIR}")


if __name__ == "__main__":
    main()
