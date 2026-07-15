"""Assemble the noise-fidelity ladder (real Willow + synthetic rungs) and compare
decoder rankings.

Loads the merged eval_all.csv from each rung's results dir, aggregates mean
per-cycle LER by (decoder, distance) over a matched round window, prints one table
per rung, and reports Kendall tau / Spearman rank agreement between each synthetic
rung's decoder ranking and the real-Willow ranking, per distance.

This is the LOCAL analysis step (CLAUDE.md: "SLURM jobs compute RAW DATA ONLY").
No decode/model work happens here -- just CSV aggregation and rank stats.
"""

from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict

from scipy.stats import kendalltau, spearmanr

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")

# rung name -> results subdir. Order matters (fidelity ladder, low to high).
RUNGS = [
    ("uniform", "willow_synth_uniform"),
    ("si1000", "willow_synth_si1000"),
    ("fit", "willow_synth_fit"),
    ("syndrome", "willow_synth_syndrome"),
    ("real", "willow_real"),
]

MIN_ROUNDS = 2
MAX_ROUNDS = 30  # matched window: the synthetic rungs were capped here (spec S5)


def _label(row: dict) -> str:
    d = row["decoder"]
    return f"ising-{row['model']}" if d == "ising" else d


def load_eval_all(rung_dir: str) -> list[dict]:
    path = os.path.join(RESULTS, rung_dir, "eval_all.csv")
    if not os.path.exists(path):
        # fall back to merging shard CSVs directly (rung dir has no eval_all.csv yet)
        rows: list[dict] = []
        for f in sorted(glob.glob(os.path.join(RESULTS, rung_dir, "*.csv"))):
            if os.path.basename(f) == "eval_all.csv":
                continue
            with open(f, newline="") as fh:
                rows.extend(list(csv.DictReader(fh)))
        return rows
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def aggregate(rows: list[dict]) -> dict[tuple[str, str], list[float]]:
    """(decoder, distance) -> list of per-cycle LERs in the matched round window."""
    agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        try:
            rounds = int(r["rounds"])
            lpc = float(r["ler_per_cycle"])
        except (ValueError, KeyError):
            continue
        if rounds < MIN_ROUNDS or rounds > MAX_ROUNDS or lpc != lpc:
            continue
        agg[(_label(r), r["distance"])].append(lpc)
    return agg


def print_table(name: str, agg: dict[tuple[str, str], list[float]]) -> None:
    decoders = sorted({k[0] for k in agg})
    dists = sorted({k[1] for k in agg}, key=int)
    print(f"\n=== {name}: mean per-cycle LER, rounds {MIN_ROUNDS}-{MAX_ROUNDS} ===")
    print(f"{'decoder':<16}" + "".join(f"  d{d:<16}" for d in dists))
    print("-" * (16 + 18 * len(dists)))
    for dec in decoders:
        cells = []
        for d in dists:
            vals = agg.get((dec, d))
            cells.append(f"{sum(vals) / len(vals):.5f} (n={len(vals):>3})" if vals else "-")
        print(f"{dec:<16}" + "".join(f"  {c:<16}" for c in cells))


def rank_agreement(real: dict, synth: dict, synth_name: str) -> None:
    dists = sorted({k[1] for k in real} & {k[1] for k in synth}, key=int)
    print(f"\n=== rank agreement: {synth_name} vs real ===")
    for d in dists:
        decoders = sorted(
            {k[0] for k in real if k[1] == d} & {k[0] for k in synth if k[1] == d}
        )
        if len(decoders) < 3:
            print(f"  d{d}: too few shared decoders ({len(decoders)}) to rank")
            continue
        real_vals = [sum(real[(dec, d)]) / len(real[(dec, d)]) for dec in decoders]
        synth_vals = [sum(synth[(dec, d)]) / len(synth[(dec, d)]) for dec in decoders]
        tau, tau_p = kendalltau(real_vals, synth_vals)
        rho, rho_p = spearmanr(real_vals, synth_vals)
        print(f"  d{d} ({len(decoders)} decoders): "
              f"Kendall tau={tau:.3f} (p={tau_p:.3f})  Spearman rho={rho:.3f} (p={rho_p:.3f})")


def _tau_by_distance(real: dict, synth: dict) -> dict[str, float]:
    """{distance -> Kendall tau} of a synthetic rung's decoder ranking vs real,
    over the decoders/distances the two share (same basis as rank_agreement)."""
    out: dict[str, float] = {}
    for d in sorted({k[1] for k in real} & {k[1] for k in synth}, key=int):
        decoders = sorted(
            {k[0] for k in real if k[1] == d} & {k[0] for k in synth if k[1] == d}
        )
        if len(decoders) < 3:
            continue
        rv = [sum(real[(dec, d)]) / len(real[(dec, d)]) for dec in decoders]
        sv = [sum(synth[(dec, d)]) / len(synth[(dec, d)]) for dec in decoders]
        tau, _ = kendalltau(rv, sv)
        out[d] = tau
    return out


# Fidelity ladder, low to high; label = how it reads on the x-axis.
_LADDER = [("uniform", "uniform"), ("si1000", "SI1000"),
           ("fit", "25-param fit"), ("syndrome", "syndrome DEM")]
# Okabe-Ito, colorblind-safe: one fixed hue per code distance.
_DIST_COLOR = {"3": "#0072B2", "5": "#E69F00", "7": "#009E73"}


def make_figure(tables: dict, out: str) -> None:
    """Central deliverable: rank agreement with real Willow (Kendall tau) as the
    synthetic noise model climbs the fidelity ladder, one line per code distance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ladder = [(n, lbl) for n, lbl in _LADDER if n in tables]
    taus = {n: _tau_by_distance(tables["real"], tables[n]) for n, _ in ladder}
    dists = sorted({d for t in taus.values() for d in t}, key=int)
    x = list(range(len(ladder)))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axhline(1.0, color="#bbbbbb", lw=1, ls="--", zorder=0)
    for d in dists:
        y = [taus[n].get(d, float("nan")) for n, _ in ladder]
        ax.plot(x, y, "-o", color=_DIST_COLOR.get(d, "#333333"), lw=2, ms=7, label=f"d={d}")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in ladder])
    ax.set_xlabel("synthetic noise fidelity  →")
    ax.set_ylabel(r"Kendall $\tau$ vs real Willow")
    ax.set_ylim(0.5, 1.03)
    ax.grid(axis="y", color="#eeeeee", lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\nfigure -> {out}")


def main() -> None:
    tables = {}
    for rung, rung_dir in RUNGS:
        rows = load_eval_all(rung_dir)
        if not rows:
            print(f"[skip] {rung}: no data at results/{rung_dir}/")
            continue
        agg = aggregate(rows)
        tables[rung] = agg
        print_table(rung, agg)

    if "real" in tables:
        for rung, _ in RUNGS:
            if rung != "real" and rung in tables:
                rank_agreement(tables["real"], tables[rung], rung)
        make_figure(tables, os.path.join(RESULTS, "ladder.png"))


if __name__ == "__main__":
    main()
