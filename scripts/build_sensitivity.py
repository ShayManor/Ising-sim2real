"""RQ3 analysis: which noise parameter, when mischaracterized (overestimated),
flips the decoder ranking?

Reads the sweep's per-setting results (results/sensitivity/<setting>/, where
<setting> is 'baseline' or a perturbed param name), ranks the decoder panel by
mean per-cycle LER per distance, and for each perturbed param measures how far
its ranking has moved from the baseline-fit ranking (Kendall tau + pairwise
inversions). Also reports how each perturbation shifts agreement with real
Willow. Prints a table ranked by disruption and writes a bar-chart figure.

LOCAL step only (CLAUDE.md: "SLURM jobs compute RAW DATA ONLY").
"""

from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict

from scipy.stats import kendalltau

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
SWEEP = os.path.join(RESULTS, "sensitivity")
REAL_DIR = os.path.join(RESULTS, "willow_real")


def _label(row: dict) -> str:
    d = row["decoder"]
    return f"ising-{row['model']}" if d == "ising" else d


def _load_rows(path_dir: str) -> list[dict]:
    merged = os.path.join(path_dir, "eval_all.csv")
    if os.path.exists(merged):
        with open(merged, newline="") as fh:
            return list(csv.DictReader(fh))
    rows: list[dict] = []
    for f in sorted(glob.glob(os.path.join(path_dir, "*.csv"))):
        if os.path.basename(f) == "eval_all.csv":
            continue
        with open(f, newline="") as fh:
            rows.extend(list(csv.DictReader(fh)))
    return rows


def aggregate(rows: list[dict], min_r: int = 2, max_r: int = 250) -> dict[tuple[str, str], float]:
    """(decoder, distance) -> mean per-cycle LER over the rounds present."""
    acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        try:
            rounds = int(r["rounds"])
            lpc = float(r["ler_per_cycle"])
        except (ValueError, KeyError):
            continue
        if rounds < min_r or rounds > max_r or lpc != lpc:
            continue
        acc[(_label(r), r["distance"])].append(lpc)
    return {k: sum(v) / len(v) for k, v in acc.items()}


def _per_distance(agg: dict[tuple[str, str], float]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for (dec, dist), v in agg.items():
        out[dist][dec] = v
    return out


def inversions(order_a: list[str], order_b: list[str]) -> int:
    """Number of decoder pairs ranked in opposite order between two orderings."""
    pos_b = {d: i for i, d in enumerate(order_b)}
    n = 0
    for i in range(len(order_a)):
        for j in range(i + 1, len(order_a)):
            if pos_b[order_a[i]] > pos_b[order_a[j]]:
                n += 1
    return n


def compare(setting: dict[str, float], baseline: dict[str, float], dist: str):
    """Compare one setting's decoder ranking to baseline at one distance.
    Returns (tau, n_inversions, top1_changed, n_decoders) over shared decoders."""
    s_by = _per_distance(setting).get(dist, {})
    b_by = _per_distance(baseline).get(dist, {})
    shared = sorted(set(s_by) & set(b_by))
    if len(shared) < 3:
        return None
    s_vals = [s_by[d] for d in shared]
    b_vals = [b_by[d] for d in shared]
    tau, _ = kendalltau(b_vals, s_vals)
    b_order = sorted(shared, key=lambda d: b_by[d])
    s_order = sorted(shared, key=lambda d: s_by[d])
    return tau, inversions(b_order, s_order), b_order[0] != s_order[0], len(shared)


def main() -> None:
    settings = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(SWEEP, "*"))
        if os.path.isdir(p) and os.path.basename(p) != "models"
    )
    if "baseline" not in settings:
        raise SystemExit(f"no baseline set under {SWEEP}/ (found: {settings})")

    aggs = {s: aggregate(_load_rows(os.path.join(SWEEP, s))) for s in settings}
    baseline = aggs["baseline"]
    real = aggregate(_load_rows(REAL_DIR)) if os.path.isdir(REAL_DIR) else {}
    dists = sorted({d for (_dec, d) in baseline}, key=int)

    def tau_vs_real(agg) -> dict[str, float]:
        out = {}
        for d in dists:
            a = _per_distance(agg).get(d, {})
            r = _per_distance(real).get(d, {})
            shared = sorted(set(a) & set(r))
            if len(shared) >= 3:
                out[d], _ = kendalltau([r[x] for x in shared], [a[x] for x in shared])
        return out

    base_real = tau_vs_real(baseline)

    print(f"baseline-fit ranking vs real (Kendall tau): "
          + "  ".join(f"d{d}={base_real.get(d, float('nan')):.3f}" for d in dists))
    print(f"\n{'param':<16}" + "".join(f"  d{d}:tau/inv" for d in dists) + "   disruption")
    print("-" * (16 + 12 * len(dists) + 12))

    scored = []
    for s in settings:
        if s == "baseline":
            continue
        cells, total_inv = [], 0
        for d in dists:
            c = compare(aggs[s], baseline, d)
            if c is None:
                cells.append(f"  {'-':>9}")
                continue
            tau, inv, _top1, _n = c
            total_inv += inv
            cells.append(f"  {tau:+.2f}/{inv:<2d}")
        scored.append((total_inv, s, cells))

    for total_inv, s, cells in sorted(scored, reverse=True):
        print(f"{s:<16}" + "".join(cells) + f"   {total_inv:>3d} inv")

    _make_figure(scored, dists)


def _make_figure(scored, dists) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not available; skipping figure)")
        return
    scored = sorted(scored)  # ascending so most-disruptive is at the top of a barh
    params = [s for _inv, s, _c in scored]
    vals = [inv for inv, _s, _c in scored]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.32 * len(params))))
    colors = ["#c0392b" if v > 0 else "#bdc3c7" for v in vals]
    ax.barh(params, vals, color=colors)
    ax.set_xlabel("ranking inversions vs baseline fit (summed over distances)")
    ax.set_title("RQ3: parameter mischaracterization vs decoder ranking")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = os.path.join(SWEEP, "sensitivity.png")
    fig.savefig(out, dpi=150)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
