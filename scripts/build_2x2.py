"""2x2 prior/channel decomposition (LOCAL analysis of slurm/eval_2x2.sbatch).

For each swept param and the fit rung, four cells (L = mean per-cycle LER):

    L(t,t)   baseline           -> results/sensitivity/baseline
    L(t',t') matched perturbed  -> results/sensitivity/<param>
    L(t,t')  source t, prior t' -> results/twobytwo/S-baseline__P-<param>   (off-diag)
    L(t',t)  source t', prior t -> results/twobytwo/S-<param>__P-baseline   (off-diag)

    Delta_channel = L(t',t') - L(t,t)   (the noise genuinely got worse)
    Delta_prior   = L(t,t')  - L(t,t)   (decoder prior mischaracterized, syndromes fixed)

Because the off-diagonal L(t,t') cell samples from the SAME source (baseline) with the
SAME seed/rounds as L(t,t), the two share syndromes exactly, so Delta_prior isolates the
pure prior effect. This answers the reviewer's question: is the ranking shift from RQ3
driven by the channel or by the decoder's prior? Reports per-decoder deltas and, for the
decision-relevant mwpm<->bplsd pair, whether the swap is channel- or prior-driven.

Reuses build_selection's matched-config machinery. LOCAL only.
"""

from __future__ import annotations

import os

from scripts.build_selection import load_cells, matched_means

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(HERE, "results")
SENS = os.path.join(RESULTS, "sensitivity")
TWOBYTWO = os.path.join(RESULTS, "twobytwo")

PAIR = ("mwpm", "bplsd")  # the only pair RQ3 ever flips


def _params_present() -> list[str]:
    """Params with both off-diagonal cells on disk."""
    if not os.path.isdir(TWOBYTWO):
        return []
    cells = {d for d in os.listdir(TWOBYTWO) if os.path.isdir(os.path.join(TWOBYTWO, d))}
    params = set()
    for c in cells:
        if c.startswith("S-baseline__P-"):
            p = c[len("S-baseline__P-"):]
            if f"S-{p}__P-baseline" in cells:
                params.add(p)
    return sorted(params)


def main() -> None:
    params = _params_present()
    if not params:
        raise SystemExit(f"no 2x2 cells under {TWOBYTWO}/ yet (pull from scratch first)")

    base = load_cells(os.path.join(SENS, "baseline"))[0]  # L(t,t)
    dists = sorted({d for (_dec, d) in base}, key=int)

    print(f"{'param':<16}{'dist':<5}{'decoder':<14}"
          f"{'L(t,t)':>9}{'D_chan':>9}{'D_prior':>9}{'chan%':>7}")
    print("-" * 69)
    for p in params:
        tptp = load_cells(os.path.join(SENS, p))[0]                       # L(t',t')
        ttp = load_cells(os.path.join(TWOBYTWO, f"S-baseline__P-{p}"))[0]  # L(t,t')
        for dist in dists:
            # decoders present in all three relevant sets at this distance
            decs = sorted(
                {d for (d, dd) in base if dd == dist}
                & {d for (d, dd) in tptp if dd == dist}
                & {d for (d, dd) in ttp if dd == dist}
            )
            if not decs:
                continue
            m_tt = matched_means(base, decs, dist)
            m_tptp = matched_means(tptp, decs, dist)
            m_ttp = matched_means(ttp, decs, dist)
            for dec in decs:
                d_chan = m_tptp[dec] - m_tt[dec]
                d_prior = m_ttp[dec] - m_tt[dec]
                tot = abs(d_chan) + abs(d_prior)
                frac = 100 * abs(d_chan) / tot if tot else float("nan")
                print(f"{p:<16}{dist:<5}{dec:<14}"
                      f"{m_tt[dec] * 1e3:>9.3f}{d_chan * 1e3:>9.3f}"
                      f"{d_prior * 1e3:>9.3f}{frac:>7.0f}")
            # mwpm<->bplsd crossing attribution
            a, b = PAIR
            if a in m_tt and b in m_tt:
                _pair_attribution(p, dist, m_tt, m_tptp, m_ttp, a, b)
        print()


def _pair_attribution(p, dist, m_tt, m_tptp, m_ttp, a, b) -> None:
    def order(m):
        return f"{a}<{b}" if m[a] < m[b] else f"{b}<{a}"
    base_ord, chan_ord, prior_ord = order(m_tt), order(m_tptp), order(m_ttp)
    if base_ord != chan_ord or base_ord != prior_ord:
        chan = " channel-flips" if base_ord != chan_ord else ""
        prior = " prior-flips" if base_ord != prior_ord else ""
        print(f"    -> {a}/{b} @ d{dist}: base[{base_ord}] "
              f"perturbed[{chan_ord}]{chan}{prior}")


if __name__ == "__main__":
    main()
