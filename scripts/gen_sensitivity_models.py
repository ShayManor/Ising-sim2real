#!/usr/bin/env python3
"""Generate perturbed noise-model sets for the RQ3 per-parameter sensitivity sweep.

For each of the 23 FREE parameters of the fitted 25-param NoiseModel, write a full
set of 14 per-patch model JSONs into ``results/sensitivity/models/<param>/`` with
that ONE parameter overestimated and every other parameter left at its fitted
value. A ``baseline/`` set (the unperturbed fits, copied verbatim) is also written
so the sweep's comparison is apples-to-apples on the same reduced config set.

This is data-prep, not a SLURM job (CLAUDE.md's raw-data/stats split): run it
locally, rsync ``results/sensitivity/models`` to the cluster, then submit
``slurm/eval_sensitivity.sbatch`` which points the fit rung at each set via
``FITTED_MODELS_DIR``.

Perturbation ("overestimate parameter k"): the fits drive many parameters to
exactly 0 (a plain x2 would be a no-op there), so the overestimate is

    perturbed = min(cap, max(fitted * MULT, fitted + FLOOR))

i.e. "x2, or at least +FLOOR, whichever is the larger overestimate", clamped to
CAP. For a param the fit left near zero this bumps it to a small-but-real value
(a genuine mischaracterization); for a healthily-fit param it is a clean x2.

p_prep_X and p_meas_X are structurally tied to their _Z counterparts (Willow's
circuits have no RX/MX -- see the fit-rung design spec), so they are NOT swept
independently; when p_prep_Z / p_meas_Z is perturbed its X copy is tied along.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ising_sim2real.paths import REPO_ROOT

BASELINE_DIR = REPO_ROOT / "results" / "fitted_noise_models"

# Canonical 25 NoiseModel parameter names.
CNOT = [f"p_cnot_{k}" for k in
        ("IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI", "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ")]
IDLE_CNOT = ["p_idle_cnot_X", "p_idle_cnot_Y", "p_idle_cnot_Z"]
IDLE_SPAM = ["p_idle_spam_X", "p_idle_spam_Y", "p_idle_spam_Z"]
# 23 free params: everything except the two tied X-basis SPAM knobs.
FREE_PARAMS = CNOT + IDLE_CNOT + IDLE_SPAM + ["p_prep_Z", "p_meas_Z"]
# When a free param is perturbed, its tied X copy follows.
TIED = {"p_prep_Z": "p_prep_X", "p_meas_Z": "p_meas_X"}


def overestimate(v: float, mult: float, floor: float, cap: float) -> float:
    return min(cap, max(v * mult, v + floor))


def load_baseline_models(baseline_dir: Path) -> dict[str, dict]:
    models = {}
    for path in sorted(baseline_dir.glob("*.json")):
        models[path.stem] = json.loads(path.read_text())
    if not models:
        raise SystemExit(f"no baseline models in {baseline_dir}")
    return models


def write_set(out_dir: Path, models: dict[str, dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for patch, params in models.items():
        (out_dir / f"{patch}.json").write_text(json.dumps(params, sort_keys=True, indent=2))


def perturb_param(params: dict, param: str, mult: float, floor: float, cap: float) -> dict:
    out = dict(params)
    out[param] = overestimate(params.get(param, 0.0), mult, floor, cap)
    if param in TIED:
        out[TIED[param]] = out[param]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "sensitivity" / "models")
    ap.add_argument("--mult", type=float, default=2.0, help="multiplicative overestimate factor")
    ap.add_argument("--floor", type=float, default=1e-3, help="minimum additive overestimate")
    ap.add_argument("--cap", type=float, default=0.5, help="clamp perturbed probabilities to this max")
    args = ap.parse_args()

    baseline = load_baseline_models(args.baseline_dir)
    write_set(args.out / "baseline", baseline)

    manifest = {"mult": args.mult, "floor": args.floor, "cap": args.cap,
                "patches": sorted(baseline), "params": {}}
    for param in FREE_PARAMS:
        perturbed = {patch: perturb_param(p, param, args.mult, args.floor, args.cap)
                     for patch, p in baseline.items()}
        write_set(args.out / param, perturbed)
        manifest["params"][param] = {
            patch: {"baseline": baseline[patch].get(param, 0.0), "perturbed": perturbed[patch][param]}
            for patch in baseline
        }
    (args.out / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2))

    print(f"wrote baseline + {len(FREE_PARAMS)} perturbed sets "
          f"({len(baseline)} patches each) -> {args.out}")
    print(f"  mult={args.mult} floor={args.floor} cap={args.cap}")


if __name__ == "__main__":
    main()
