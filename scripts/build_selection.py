"""Constrained decoder selection + deployment regret (LOCAL analysis).

Motivation: on unconstrained min-LER the top decoder (tesseract) never changes
across sources, so "synthetic picks the wrong decoder" is false as stated. But
tesseract is offline-only (~11 ms/shot median, >0.6 s/shot at d7); MWPM is ~0.02
ms/shot. Under a latency budget the feasible panel shrinks and the choice can
flip. This script asks, per source M and latency budget B:

    d*(M, B) = argmin_{d : latency(d) <= B} LER_M(d)         (who M tells you to pick)
    regret(M, B) = LER_real(d*(M,B)) - min_{d: lat<=B} LER_real(d)

regret == 0 iff source M selects the decoder real would have picked under B.

CRITICAL: per-cycle LER means are only comparable across decoders on a COMMON
config set. Decoders cover different (basis, patch, rounds) configs, so this
aggregates every decoder over the matched set of configs present for ALL feasible
decoders (per source, per distance). Comparing unmatched means silently biases the
"best decoder" toward whichever decoder was evaluated on the easier configs.

Reads the four synthetic rungs + real from results/. Latency from real (the
deployment target), median ms/shot per (decoder, distance). No cluster compute.
"""

from __future__ import annotations

import csv
import glob
import os
import statistics
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
REAL_DIR = os.path.join(RESULTS, "willow_real")

SOURCES = [
    ("uniform", "willow_synth_uniform"),
    ("si1000", "willow_synth_si1000"),
    ("fit", "willow_synth_fit"),
    ("syndrome", "willow_synth_syndrome"),
    ("real", "willow_real"),
]
BUDGETS = [("0.1ms", 0.1), ("1ms", 1.0), ("10ms", 10.0), ("offline", None)]


def _label(dec: str, model: str) -> str:
    return f"ising-{model}" if dec == "ising" else dec


def load_cells(path_dir: str):
    """-> (ler[(dec,dist)][config]=lpc, counts[(dec,dist)][config]=(n_err,shots),
    latency[(dec,dist)]=median ms/shot). config = (basis, patch, rounds), read from
    the CSV COLUMNS (patch = the ``orientation`` column) so it works on both the
    merged eval_all.csv and per-shard files, and per-cycle means can be matched
    across decoders on a common config set."""
    merged = os.path.join(path_dir, "eval_all.csv")
    if os.path.exists(merged):
        files = [merged]
    else:
        files = [f for f in sorted(glob.glob(os.path.join(path_dir, "*.csv")))
                 if os.path.basename(f) != "eval_all.csv"]
    ler: dict[tuple[str, str], dict[tuple, float]] = defaultdict(dict)
    counts: dict[tuple[str, str], dict[tuple, tuple]] = defaultdict(dict)
    lat_acc: dict[tuple[str, str], list[float]] = defaultdict(list)
    for f in files:
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                try:
                    lpc = float(row["ler_per_cycle"])
                    rnd = int(row["rounds"])
                    dist, basis, patch = row["distance"], row["basis"], row["orientation"]
                except (ValueError, KeyError):
                    continue
                dec = _label(row["decoder"], row.get("model", ""))
                cfg = (basis, patch, rnd)
                if lpc == lpc:
                    ler[(dec, dist)][cfg] = lpc
                try:
                    ne, sh = int(row["n_errors"]), int(row["shots"])
                    counts[(dec, dist)][cfg] = (ne, sh)
                    if sh > 0:
                        lat_acc[(dec, dist)].append(
                            float(row["decode_seconds"]) / sh * 1000.0)
                except (ValueError, KeyError):
                    pass
    latency = {k: statistics.median(v) for k, v in lat_acc.items()}
    return ler, counts, latency


def _per_cycle(ne: int, sh: int, rnd: int) -> float:
    ler = ne / sh
    return 1 - (1 - ler) ** (1.0 / rnd) if rnd > 0 else ler


def bootstrap_prefers(counts, a: str, b: str, dist: str, nboot: int = 4000):
    """Matched-config binomial bootstrap of mean-per-cycle-LER(a) - LER(b).
    Returns (delta, lo, hi, n_cfgs) or None. delta<0 => a better; CI excluding 0 =>
    significant. Conservative (treats shots as independent binomial, no pairing)."""
    ca, cb = counts.get((a, dist), {}), counts.get((b, dist), {})
    cfgs = sorted(set(ca) & set(cb))
    if not cfgs:
        return None
    A = np.array([ca[c] for c in cfgs], float)   # (n, 2) = n_err, shots
    B = np.array([cb[c] for c in cfgs], float)
    rn = np.array([c[2] for c in cfgs], float)
    n = len(cfgs)
    obs = (np.array([_per_cycle(*ca[c], c[2]) for c in cfgs]).mean()
           - np.array([_per_cycle(*cb[c], c[2]) for c in cfgs]).mean())
    rng = np.random.default_rng(20260712)
    diffs = np.empty(nboot)
    for i in range(nboot):
        idx = rng.integers(0, n, n)
        nea = rng.binomial(A[idx, 1].astype(int), A[idx, 0] / A[idx, 1])
        neb = rng.binomial(B[idx, 1].astype(int), B[idx, 0] / B[idx, 1])
        lca = 1 - (1 - nea / A[idx, 1]) ** (1.0 / rn[idx])
        lcb = 1 - (1 - neb / B[idx, 1]) ** (1.0 / rn[idx])
        diffs[i] = lca.mean() - lcb.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return obs, lo, hi, n


def matched_means(ler, decs: list[str], dist: str) -> dict[str, float]:
    """Mean per-cycle LER for each decoder over configs present for ALL `decs`."""
    cfg_sets = [set(ler.get((d, dist), {})) for d in decs]
    if not cfg_sets or not all(cfg_sets):
        return {}
    common = set.intersection(*cfg_sets)
    if not common:
        return {}
    return {d: sum(ler[(d, dist)][c] for c in common) / len(common) for d in decs}


def main() -> None:
    data = {name: load_cells(os.path.join(RESULTS, d))
            for name, d in SOURCES if os.path.isdir(os.path.join(RESULTS, d))}
    real_ler, real_counts, real_lat = data["real"]
    lat_by = defaultdict(dict)
    for (dec, dist), v in real_lat.items():
        lat_by[dist][dec] = v
    dists = sorted({d for (_dec, d) in real_ler}, key=int)
    synth_names = [n for n, _d in SOURCES if n in data and n != "real"]

    print("Latency (real, median ms/shot):")
    decs_all = sorted({dec for (dec, _d) in real_lat})
    print(f"  {'decoder':<16}" + "".join(f"d{d:>7}" for d in dists))
    for dec in decs_all:
        print(f"  {dec:<16}" + "".join(
            f"{lat_by[d].get(dec, float('nan')):8.3f}" for d in dists))

    failures = []
    for dist in dists:
        lat = lat_by.get(dist, {})
        # shared universe: decoders on real AND on every synth source (excludes mwpm-rl)
        panels = [{dec for (dec, dd) in data[n][0] if dd == dist} for n in synth_names]
        real_decs = {dec for (dec, dd) in real_ler if dd == dist}
        shared = sorted(set.intersection(real_decs, *panels)) if panels else []
        print(f"\n=== d{dist} ===  shared panel: {', '.join(shared)}"
              + ("  (+ real-only mwpm-rl, footnote)" if "mwpm-rl" in real_decs else ""))
        print(f"  {'budget':<9}{'nfeas':<7}"
              + "".join(f"{n:>12}" for n in synth_names) + f"{'real-best':>12}")
        for bname, bval in BUDGETS:
            feas = [d for d in shared if bval is None or lat.get(d, float("inf")) <= bval]
            rmeans = matched_means(real_ler, feas, dist)
            real_best = min(rmeans, key=lambda d: rmeans[d]) if rmeans else None
            picks, regrets = [], []
            for sname in synth_names:
                smeans = matched_means(data[sname][0], feas, dist)
                pick = min(smeans, key=lambda d: smeans[d]) if smeans else None
                picks.append(pick)
                if pick and real_best:
                    reg = rmeans[pick] - rmeans[real_best]
                    regrets.append(reg)
                    if pick != real_best and reg > 0:
                        failures.append((dist, bname, sname, pick, real_best, reg))
                else:
                    regrets.append(None)
            pcells = "".join(
                f"{(p + ('*' if p != real_best else '')) if p else '-':>12}" for p in picks)
            print(f"  {bname:<9}{len(feas):<7}{pcells}{(real_best or '-'):>12}")
            print(f"  {'  regret1e3':<16}"
                  + "".join(f"{r * 1e3:>12.3f}" if r is not None else f"{'-':>12}"
                            for r in regrets))

    print("\n" + "=" * 64)
    if not failures:
        print("No selection failures under any budget on matched configs -- the "
              "ranking is robust even when latency-constrained.")
        return
    print(f"CANDIDATE SELECTION FAILURES ({len(failures)}), bootstrap-gated "
          f"(95% CI, matched configs). A failure is REAL only if BOTH sides are "
          f"significant: real prefers real-best AND source prefers its pick.")
    for dist, b, s, pick, rbest, reg in failures:
        r = bootstrap_prefers(real_counts, rbest, pick, dist)          # real side
        sc = data[s][1]
        sv = bootstrap_prefers(sc, pick, rbest, dist)                  # source side
        r_sig = r and r[2] < 0                    # LER(rbest)-LER(pick) < 0 on real
        s_sig = sv and (sv[2] < 0)                # LER(pick)-LER(rbest) < 0 on source
        verdict = "REAL" if (r_sig and s_sig) else "NOT SIGNIFICANT (near-tie)"
        print(f"\n  d{dist} @ {b}: {s} picks {pick}, real-best {rbest}; "
              f"regret={reg * 1e3:.3f}e-3/cycle  ->  {verdict}")
        if r:
            print(f"    real:   LER({rbest})-LER({pick}) = {r[0]*1e3:+.3f}e-3 "
                  f"CI[{r[1]*1e3:+.3f},{r[2]*1e3:+.3f}] (n={r[3]})")
        if sv:
            print(f"    {s+':':<8} LER({pick})-LER({rbest}) = {sv[0]*1e3:+.3f}e-3 "
                  f"CI[{sv[1]*1e3:+.3f},{sv[2]*1e3:+.3f}] (n={sv[3]})")


if __name__ == "__main__":
    main()
