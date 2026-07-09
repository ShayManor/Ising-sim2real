"""Merge the real-Willow per-shard eval CSVs and summarize Ising vs the classical panel.

Reads every per-shard CSV under results/willow_real/, concatenates them into one
eval_all.csv, and prints a decoder x distance table of mean per-cycle LER over the
matched round window (2 <= rounds <= 70, where every decoder has coverage).
"""

from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(HERE, "results", "willow_real")
COLS = ["distance", "basis", "orientation", "rounds", "decoder", "model",
        "shots", "n_errors", "ler", "ler_per_cycle", "decode_seconds", "note"]


def main() -> None:
    rows = []
    for path in sorted(glob.glob(os.path.join(SRC, "*.csv"))):
        base = os.path.basename(path)
        if base == "eval_all.csv":
            continue
        # beliefmatching has two sources: the dedicated capped job (per-round, 10k shots,
        # names end _r{N}.csv -- consistent with the rest of the slow panel) and the main
        # job's redundant per-patch shards (all-rounds, 50k). Keep only the per-round ones
        # so the comparison uses one consistent beliefmatching estimate.
        if base.startswith("beliefmatching_") and "_r" not in base:
            continue
        with open(path, newline="") as fh:
            rows.extend(list(csv.DictReader(fh)))

    merged = os.path.join(SRC, "eval_all.csv")
    with open(merged, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})
    print(f"merged {len(rows)} rows -> {merged}\n")

    # decoder label folds ising model into the name so fast/accurate are distinct.
    def label(r: dict) -> str:
        d = r["decoder"]
        return f"ising-{r['model']}" if d == "ising" else d

    # mean per-cycle LER over the matched window 2..70 rounds.
    agg: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        try:
            rounds = int(r["rounds"])
            lpc = float(r["ler_per_cycle"])
        except (ValueError, KeyError):
            continue
        if rounds < 2 or rounds > 70 or lpc != lpc:  # skip r<2 and NaN
            continue
        agg[(label(r), r["distance"])].append(lpc)

    decoders = sorted({k[0] for k in agg})
    dists = sorted({k[1] for k in agg}, key=int)

    print("Mean per-cycle LER on REAL Willow, rounds 2..70 (n configs)")
    print(f"{'decoder':<16}" + "".join(f"  d{d:<14}" for d in dists))
    print("-" * (16 + 17 * len(dists)))
    for dec in decoders:
        cells = []
        for d in dists:
            vals = agg.get((dec, d))
            cells.append(f"{sum(vals)/len(vals):.5f} (n={len(vals):>3})" if vals else f"{'-':<15}")
        print(f"{dec:<16}" + "".join(f"  {c:<15}" for c in cells))


if __name__ == "__main__":
    main()
