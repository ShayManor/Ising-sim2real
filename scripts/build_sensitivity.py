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

from scripts.build_selection import bootstrap_prefers, load_cells

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
# SWEEP_DIR lets the same analysis run over a multi-round / underestimate sweep
# (e.g. results/sensitivity_r10) instead of the default r30 sweep.
SWEEP = os.environ.get("SWEEP_DIR", os.path.join(RESULTS, "sensitivity"))
REAL_DIR = os.path.join(RESULTS, "willow_real")


def significant_inversions(base_counts, pert_counts, base_by, pert_by, dist) -> int:
    """Of the decoder pairs that swap order between baseline and a perturbed
    setting, how many are REAL swaps -- both orderings bootstrap-significant (95%
    CI excludes a tie), not near-ties. Filters the reviewer's 'CI includes a tie'
    case out of the raw inversion count."""
    b, s = base_by.get(dist, {}), pert_by.get(dist, {})
    shared = sorted(set(b) & set(s))
    n = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            di, dj = shared[i], shared[j]
            if (b[di] < b[dj]) == (s[di] < s[dj]):
                continue  # not swapped
            rb = bootstrap_prefers(base_counts, di, dj, dist)
            rs = bootstrap_prefers(pert_counts, di, dj, dist)
            if rb and rs and (rb[1] > 0 or rb[2] < 0) and (rs[1] > 0 or rs[2] < 0):
                n += 1
    return n


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
    # per-config counts for bootstrap-gating the inversions (near-tie filter)
    counts = {s: load_cells(os.path.join(SWEEP, s))[1] for s in settings}
    base_by = _per_distance(baseline)

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
    print(f"\n{'param':<16}" + "".join(f"  d{d}:tau/inv(sig)" for d in dists)
          + "   disruption (sig=bootstrap-real swaps)")
    print("-" * (16 + 16 * len(dists) + 12))

    scored = []
    for s in settings:
        if s == "baseline":
            continue
        cells, total_inv, total_sig = [], 0, 0
        pert_by = _per_distance(aggs[s])
        for d in dists:
            c = compare(aggs[s], baseline, d)
            if c is None:
                cells.append(f"  {'-':>13}")
                continue
            tau, inv, _top1, _n = c
            sig = significant_inversions(counts["baseline"], counts[s],
                                         base_by, pert_by, d) if inv else 0
            total_inv += inv
            total_sig += sig
            cells.append(f"  {tau:+.2f}/{inv}({sig})")
        scored.append((total_inv, total_sig, s, cells))

    for total_inv, total_sig, s, cells in sorted(scored, reverse=True):
        print(f"{s:<16}" + "".join(cells) + f"   {total_inv:>2d} inv {total_sig:>2d} sig")

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
    params = [s for _inv, _sig, s, _c in scored]
    vals = [inv for inv, _sig, _s, _c in scored]
    sigs = [sig for _inv, sig, _s, _c in scored]
    fig, ax = plt.subplots(figsize=(6, max(3, 0.32 * len(params))))
    # red = at least one bootstrap-real swap; grey = swaps but all near-ties; light = none
    colors = ["#c0392b" if sg > 0 else "#95a5a6" if v > 0 else "#ecf0f1"
              for v, sg in zip(vals, sigs)]
    ax.barh(params, vals, color=colors)
    ax.set_xlabel("ranking inversions vs baseline fit (red = bootstrap-real swap)")
    ax.set_title("RQ3: parameter mischaracterization vs decoder ranking")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = os.path.join(SWEEP, "sensitivity.png")
    fig.savefig(out, dpi=150)
    print(f"\nfigure -> {out}")


if __name__ == "__main__":
    main()
