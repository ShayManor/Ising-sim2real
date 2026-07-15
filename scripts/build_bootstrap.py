"""Joint-bootstrap analysis (LOCAL, per CLAUDE.md's raw-data/stats split).

Reads the B=40 per-draw eval CSVs + param JSON sets produced by
slurm/fit_bootstrap.sbatch (refit) and slurm/eval_sensitivity.sbatch pointed at
results/bootstrap/models (eval). Reports:
  1. the 23x23 parameter covariance/correlation (pooled within-patch) -> heatmap,
  2. the per-(decoder,distance) joint LER band across draws,
  3. rank-flip probability vs the point-estimate fit: overall, the mwpm<->bplsd
     pair, and top-1 (tesseract) stability -- near-ties gated by
     build_sensitivity/build_selection's bootstrap machinery.
No cluster compute.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from scripts.build_sensitivity import _label, aggregate  # reuse loaders/agg
from scripts.gen_sensitivity_models import FREE_PARAMS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
MODELS = os.path.join(RESULTS, "bootstrap", "models")
EVAL = os.path.join(RESULTS, "bootstrap", "eval")
PAIR = ("mwpm", "bplsd")


def param_covariance(vectors: "np.ndarray") -> "np.ndarray":
    """(k,k) covariance of an (n_samples, k) matrix (columns = params)."""
    return np.cov(vectors, rowvar=False)


def centered_pooled_matrix(draws_by_patch: dict, params: list[str]) -> "np.ndarray":
    """Stack every patch's per-draw param vectors, mean-centered per patch, so
    the pooled covariance reflects within-patch sampling covariance, not
    between-patch level differences."""
    blocks = []
    for _patch, draws in draws_by_patch.items():
        block = np.array([[d.get(p, 0.0) for p in params] for d in draws], dtype=float)
        blocks.append(block - block.mean(axis=0, keepdims=True))
    return np.vstack(blocks)


def ordering(agg: dict, distance: str) -> list[str]:
    """Decoders at `distance` (the distance STRING as it appears in the key),
    ascending by mean per-cycle LER (best first). `build_sensitivity.aggregate`
    returns {(label, distance_str): mean_lpc_float}, so the value IS the LER
    float -- not a cell dict."""
    rows = [(dec, lpc) for (dec, d), lpc in agg.items() if d == distance]
    return [dec for dec, _ in sorted(rows, key=lambda t: t[1])]


def pair_flip_rate(draw_orderings: list[list[str]], baseline: list[str],
                   a: str, b: str) -> float:
    """Fraction of draws whose a-vs-b order differs from `baseline`."""
    def before(order, x, y):
        return order.index(x) < order.index(y)
    base = before(baseline, a, b)
    flips = sum(1 for o in draw_orderings if a in o and b in o and before(o, a, b) != base)
    present = sum(1 for o in draw_orderings if a in o and b in o)
    return flips / present if present else 0.0


def _load_draw_params() -> dict:
    """{patch_key: [param_dict per draw]} across results/bootstrap/models/draw_*/."""
    by_patch: dict[str, list[dict]] = {}
    for draw_dir in sorted(glob.glob(os.path.join(MODELS, "draw_*"))):
        for jf in sorted(glob.glob(os.path.join(draw_dir, "*.json"))):
            key = os.path.splitext(os.path.basename(jf))[0]
            by_patch.setdefault(key, []).append(json.loads(open(jf).read()))
    return by_patch


def _load_rows(csv_dir: str) -> list[dict]:
    import csv
    path = os.path.join(csv_dir, "eval_all.csv")
    with open(path) as f:
        return list(csv.DictReader(f))


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- parameter covariance ----
    by_patch = _load_draw_params()
    n_draws = max((len(v) for v in by_patch.values()), default=0)
    print(f"loaded {len(by_patch)} patches x up to {n_draws} draws")
    if n_draws < 10:
        print("WARNING: <10 draws present; covariance not reported")
    else:
        m = centered_pooled_matrix(by_patch, FREE_PARAMS)
        cov = param_covariance(m)
        sd = np.sqrt(np.diag(cov))
        corr = cov / np.outer(sd, sd)
        corr[~np.isfinite(corr)] = 0.0
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(FREE_PARAMS)))
        ax.set_yticks(range(len(FREE_PARAMS)))
        ax.set_xticklabels(FREE_PARAMS, rotation=90, fontsize=6)
        ax.set_yticklabels(FREE_PARAMS, fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        cov_png = os.path.join(RESULTS, "bootstrap", "covariance.png")
        fig.savefig(cov_png, dpi=150)
        print(f"  -> {cov_png}")

    # ---- rank-flip vs point-estimate fit ----
    # Baseline = the point-estimate fit at the SAME operating point the draws use
    # (RQ3's r30 fit-rung baseline). results/willow_synth_fit is the full-ladder
    # run (r1..r250) whose high-round rows saturate to LER 0.5 and inflate the
    # per-decoder mean, so it is NOT comparable to the r30-only draws.
    draw_dirs = sorted(glob.glob(os.path.join(EVAL, "draw_*")))
    draw_aggs = [aggregate(_load_rows(d)) for d in draw_dirs]
    base_dir = os.path.join(RESULTS, "sensitivity", "baseline")
    baseline_agg = aggregate(_load_rows(base_dir)) if os.path.isdir(base_dir) else (draw_aggs[0] if draw_aggs else {})
    dists = sorted({d for (_dec, d) in baseline_agg}, key=int)
    print(f"\n{'dist':<6}{'top1-stable':<14}{'mwpm<->bplsd flip':<20}{'any-flip'}")
    for d in dists:
        base_order = ordering(baseline_agg, d)
        draw_orders = [ordering(a, d) for a in draw_aggs]
        top1 = sum(1 for o in draw_orders if o and o[0] == base_order[0]) / max(len(draw_orders), 1)
        pflip = pair_flip_rate(draw_orders, base_order, *PAIR)
        anyflip = sum(1 for o in draw_orders if o != base_order) / max(len(draw_orders), 1)
        print(f"d{d:<5}{top1:<14.2f}{pflip:<20.2f}{anyflip:.2f}")


if __name__ == "__main__":
    main()
