"""Run the decoder panel on Willow data and record LER per cycle.

The harness iterates Willow configurations, derives detection events through the
ingest pipeline, and scores each requested decoder with the same yardstick
(``metrics.logical_error_rate`` / ``logical_error_per_cycle``). Results stream to a
CSV and a tqdm progress bar.

Decoders
--------
``mwpm``        PyMatching (MWPM) off the shipped SI1000 DEM. The validated
                classical baseline (CLAUDE.md step 2/3).
``mwpm-rl``     PyMatching off the shipped RL-optimized DEM.
``ising``       NVIDIA Ising pre-decoder + PyMatching on the cleaned residual
                (RQ4). NOT validated against the baseline -- that is the
                experiment; always read its number next to ``mwpm``.

Run it (uv)
-----------
    uv run ising-eval --distances 3 --bases Z --rounds 10,30 --shots 5000
    uv run ising-eval --decoders mwpm,ising --model fast --limit 8
    uv run ising-eval                      # full panel, all 420 configs, all shots
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from ising_sim2real.device import describe_device, resolve_device
from ising_sim2real.decoders.pymatching_decoder import PyMatchingDecoder
from ising_sim2real.ingest.dataset import discover_configs
from ising_sim2real.ingest.detectors import measurements_to_detectors
from ising_sim2real.ingest.willow import WillowConfig, load_run
from ising_sim2real.metrics import logical_error_per_cycle, logical_error_rate
from ising_sim2real.paths import OUTPUTS_DIR, WILLOW_RAW_DIR

ALL_DECODERS = ("mwpm", "mwpm-rl", "ising")


@dataclass
class EvalRow:
    """One (configuration, decoder) result."""

    distance: int
    basis: str
    orientation: str
    rounds: int
    decoder: str
    model: str
    shots: int
    ler: float
    ler_per_cycle: float
    decode_seconds: float
    note: str = ""


def _select_configs(args) -> list[WillowConfig]:
    configs = discover_configs(args.data_dir)
    if args.distances:
        configs = [c for c in configs if c.distance in args.distances]
    if args.bases:
        configs = [c for c in configs if c.basis in args.bases]
    if args.rounds:
        configs = [c for c in configs if c.rounds in args.rounds]
    if args.max_rounds is not None:
        configs = [c for c in configs if c.rounds <= args.max_rounds]
    if args.limit is not None:
        configs = configs[: args.limit]
    return configs


def _score(predictions: np.ndarray, observables: np.ndarray, rounds: int) -> tuple[float, float]:
    ler = logical_error_rate(predictions, observables)
    return ler, logical_error_per_cycle(ler, rounds)


def _load_config(args, cfg: WillowConfig):
    """Fetch one config's decode inputs from the local tree or the HF dataset.

    Returns an object exposing ``circuit``, ``detectors``, ``observables``,
    ``dem_si1000`` and ``dem_rl`` (HFConfigData shape), so the eval loop is
    source-agnostic.
    """
    if args.source == "hf":
        from ising_sim2real.ingest.hf import load_config_from_hf

        return load_config_from_hf(cfg, repo=args.hf_repo)

    from ising_sim2real.ingest.hf import HFConfigData

    run = load_run(args.data_dir, cfg)
    det = measurements_to_detectors(run.circuit, run.measurements, sweep_bits=run.sweep_bits)
    return HFConfigData(
        circuit=run.circuit,
        detectors=det.detectors,
        observables=det.observables,
        dem_si1000=run.dem_si1000,
        dem_rl=run.dem_rl,
    )


def evaluate(args) -> list[EvalRow]:
    """Run the requested decoders over the selected configs; return result rows."""
    configs = _select_configs(args)
    device = resolve_device(args.device)

    want_ising = "ising" in args.decoders
    model = None
    model_info = None
    if want_ising:
        # Imported lazily so the classical-only path needs neither torch-heavy
        # vendored code nor model weights.
        from ising_sim2real.ising.loader import load_ising_model
        from ising_sim2real.ising.predecoder import IsingPreDecoder

        model, model_info = load_ising_model(args.model, device=device)

    rows: list[EvalRow] = []
    bar = tqdm(configs, unit="cfg", dynamic_ncols=True, disable=args.no_progress)
    for cfg in bar:
        bar.set_description(f"d{cfg.distance} {cfg.basis} {cfg.orientation} r{cfg.rounds}")
        data = _load_config(args, cfg)
        n = data.detectors.shape[0] if args.shots is None else min(args.shots, data.detectors.shape[0])
        dets = data.detectors[:n]
        obs = data.observables[:n]

        postfix: dict[str, str] = {}

        if "mwpm" in args.decoders and data.dem_si1000 is not None:
            res = PyMatchingDecoder.from_dem(data.dem_si1000).decode_batch(dets)
            ler, perc = _score(res.predictions, obs, cfg.rounds)
            rows.append(EvalRow(cfg.distance, cfg.basis, cfg.orientation, cfg.rounds,
                                "mwpm", "-", n, ler, perc, res.seconds))
            postfix["mwpm/c"] = f"{perc:.4f}"

        if "mwpm-rl" in args.decoders and data.dem_rl is not None:
            res = PyMatchingDecoder.from_dem(data.dem_rl).decode_batch(dets)
            ler, perc = _score(res.predictions, obs, cfg.rounds)
            rows.append(EvalRow(cfg.distance, cfg.basis, cfg.orientation, cfg.rounds,
                                "mwpm-rl", "-", n, ler, perc, res.seconds))

        if want_ising:
            if cfg.rounds < IsingPreDecoder.MIN_ROUNDS:
                rows.append(EvalRow(cfg.distance, cfg.basis, cfg.orientation, cfg.rounds,
                                    "ising", model_info.name, n, float("nan"), float("nan"),
                                    0.0, note=f"skipped: rounds<{IsingPreDecoder.MIN_ROUNDS}"))
            else:
                dec = IsingPreDecoder(model, data.circuit, cfg.basis, cfg.distance, cfg.rounds,
                                      device, rotation=args.rotation, syn_noise=args.syn_noise)
                res = dec.decode_batch(dets)
                ler, perc = _score(res.predictions, obs, cfg.rounds)
                rows.append(EvalRow(cfg.distance, cfg.basis, cfg.orientation, cfg.rounds,
                                    "ising", model_info.name, n, ler, perc, res.seconds,
                                    note=f"R={model_info.receptive_field},rot={args.rotation}"))
                postfix["ising/c"] = f"{perc:.4f}"

        if postfix:
            bar.set_postfix(postfix)
    bar.close()
    return rows


def _write_csv(rows: list[EvalRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def _int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _str_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decoders", type=_str_list, default=list(ALL_DECODERS),
                   help=f"comma list from {ALL_DECODERS} (default: all)")
    p.add_argument("--model", choices=("fast", "accurate"), default="fast",
                   help="Ising model variant (default: fast)")
    p.add_argument("--distances", type=_int_list, default=None, help="e.g. 3,5,7 (default: all)")
    p.add_argument("--bases", type=_str_list, default=None, help="X,Z (default: both)")
    p.add_argument("--rounds", type=_int_list, default=None, help="exact round counts, e.g. 10,30")
    p.add_argument("--max-rounds", type=int, default=None, help="only configs with rounds <= this")
    p.add_argument("--shots", type=int, default=None, help="subsample N shots/config (default: all 50000)")
    p.add_argument("--rotation", default="XV", help="Ising code rotation: XV|XH|ZV|ZH (default: XV)")
    p.add_argument("--syn-noise", type=float, default=1e-3, help="uniform noise for the residual matcher DEM")
    p.add_argument("--device", default="cpu", help="auto|cpu|cuda|cuda:N|mps (default: cpu — fastest here)")
    p.add_argument("--source", choices=("local", "hf"), default="local",
                   help="local Willow tree (default) or the published HF dataset")
    p.add_argument("--hf-repo", default="ShayManor/willow-surface-code-detection-events",
                   help="HF dataset repo when --source hf")
    p.add_argument("--data-dir", type=Path, default=WILLOW_RAW_DIR)
    p.add_argument("--out", type=Path, default=OUTPUTS_DIR / "eval_results.csv")
    p.add_argument("--limit", type=int, default=None, help="cap number of configs (smoke test)")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)

    bad = set(args.decoders) - set(ALL_DECODERS)
    if bad:
        p.error(f"unknown decoders {sorted(bad)}; choose from {ALL_DECODERS}")

    print(f"device: {describe_device(resolve_device(args.device))}  decoders: {args.decoders}", file=sys.stderr)
    rows = evaluate(args)
    if not rows:
        print("no configs matched the filters.", file=sys.stderr)
        return 1
    _write_csv(rows, args.out)

    # Console summary: mean per-cycle LER by decoder.
    print(f"\nwrote {len(rows)} rows -> {args.out}")
    for dec in args.decoders:
        vals = [r.ler_per_cycle for r in rows if r.decoder == dec and not np.isnan(r.ler_per_cycle)]
        if vals:
            print(f"  {dec:8s} mean LER/cycle = {np.mean(vals):.5f}  over {len(vals)} configs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
