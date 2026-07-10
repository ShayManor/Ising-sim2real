"""Run the decoder panel on Willow data and record LER per cycle.

The harness iterates Willow configurations, derives detection events through the
ingest pipeline, and scores each requested decoder with the same yardstick
(``metrics.logical_error_rate`` / ``logical_error_per_cycle``). Results stream to a
CSV and a tqdm progress bar.

Decoders
--------
``mwpm``            PyMatching (MWPM) off the shipped SI1000 DEM. The validated
                    classical baseline (CLAUDE.md step 2/3).
``mwpm-rl``         PyMatching off the shipped RL-optimized DEM.
``beliefmatching``  BP-informed correlated matching off the SI1000 DEM (Libra
                    family) -- the strong classical baseline.
``bposd``           BP + ordered-statistics decoding (``ldpc``) off the SI1000 DEM.
``bplsd``           BP + localized-statistics decoding (``ldpc``) off the SI1000 DEM.
``tesseract``       Tesseract A*-search most-likely-error decoder off the SI1000 DEM.
``ising``           NVIDIA Ising pre-decoder + PyMatching on the cleaned residual
                    (RQ4). NOT validated against the baseline -- that is the
                    experiment; always read its number next to ``mwpm``.

The ``beliefmatching``/``bposd``/``bplsd``/``tesseract`` panel needs the optional
``decoders`` extra (``uv sync --extra decoders``); they are imported lazily so a
classical/ising-only run does not require it.

Run it (uv)
-----------
    uv run ising-eval --distances 3 --bases Z --rounds 10,30 --shots 5000
    uv run ising-eval --decoders mwpm,ising --model fast --limit 8
    uv run ising-eval --decoders ising --model custom --model-id 2 --checkpoint path/to/trained.pt
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
from ising_sim2real.metrics import logical_error_outcomes, logical_error_per_cycle
from ising_sim2real.paths import OUTPUTS_DIR, WILLOW_RAW_DIR

ALL_DECODERS = ("mwpm", "mwpm-rl", "beliefmatching", "bposd", "bplsd", "tesseract", "ising")

# DEM-based panel decoders beyond the two PyMatching baselines. Each scores off the
# shipped SI1000 DEM (like ``mwpm``), so panel differences are algorithmic, not prior
# differences. Resolved lazily from "module:attr" so the optional ``decoders`` extra
# is only needed when one is requested.
_PANEL_DECODERS = {
    "beliefmatching": "ising_sim2real.decoders.beliefmatching_decoder:BeliefMatchingDecoder",
    "bposd": "ising_sim2real.decoders.ldpc_decoder:BpOsdDecoder",
    "bplsd": "ising_sim2real.decoders.ldpc_decoder:BpLsdDecoder",
    "tesseract": "ising_sim2real.decoders.tesseract_decoder:TesseractDecoder",
}


def _resolve(spec: str):
    import importlib

    module, attr = spec.split(":")
    return getattr(importlib.import_module(module), attr)


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
    n_errors: int
    ler: float
    ler_per_cycle: float
    decode_seconds: float
    note: str = ""


def _select_configs(args) -> list[WillowConfig]:
    if args.source in ("hf", "synth"):
        from ising_sim2real.ingest.hf import discover_configs_hf

        configs = discover_configs_hf(args.hf_repo)
    else:
        configs = discover_configs(args.data_dir)
    if args.distances:
        configs = [c for c in configs if c.distance in args.distances]
    if args.bases:
        configs = [c for c in configs if c.basis in args.bases]
    if args.patches:
        configs = [c for c in configs if c.orientation in args.patches]
    if args.rounds:
        configs = [c for c in configs if c.rounds in args.rounds]
    if args.max_rounds is not None:
        configs = [c for c in configs if c.rounds <= args.max_rounds]
    if args.limit is not None:
        configs = configs[: args.limit]
    return configs


def _score(predictions: np.ndarray, observables: np.ndarray, rounds: int):
    """Return (per-shot outcomes, total LER, per-cycle LER) for one decode."""
    outcomes = logical_error_outcomes(predictions, observables)
    ler = float(np.mean(outcomes))
    return outcomes, ler, logical_error_per_cycle(ler, rounds)


def _record(store, cfg: WillowConfig, decoder: str, n: int, outcomes: np.ndarray) -> None:
    """Stash a shard's per-shot outcome vector (bit-packed) for later NPZ dump."""
    if store is None:
        return
    key = f"d{cfg.distance}|{cfg.basis}|{cfg.orientation}|r{cfg.rounds}|{decoder}|n{n}"
    store[key] = np.packbits(np.asarray(outcomes, dtype=bool))


def _load_config(args, cfg: WillowConfig):
    """Fetch one config's decode inputs from the local tree or the HF dataset.

    Returns an object exposing ``circuit``, ``detectors``, ``observables``,
    ``dem_si1000`` and ``dem_rl`` (HFConfigData shape), so the eval loop is
    source-agnostic.
    """
    if args.source == "hf":
        from ising_sim2real.ingest.hf import load_config_from_hf

        return load_config_from_hf(cfg, repo=args.hf_repo)

    if args.source == "synth":
        from ising_sim2real.ingest.synthetic import sample_config

        return sample_config(
            cfg, rung=args.rung, p=args.synth_p, shots=args.synth_shots,
            seed=args.seed, repo=args.hf_repo,
        )

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

        model, model_info = load_ising_model(
            args.model, device=device, checkpoint=args.checkpoint, model_id=args.model_id
        )

    rows: list[EvalRow] = []
    # Per-shot outcome vectors (bit-packed), keyed per (config, decoder), so every
    # downstream statistic can be recomputed locally without re-decoding. None unless
    # --outcomes-dir is set.
    outcomes_store: dict | None = {} if args.outcomes_dir else None
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
            outcomes, ler, perc = _score(res.predictions, obs, cfg.rounds)
            rows.append(EvalRow(cfg.distance, cfg.basis, cfg.orientation, cfg.rounds,
                                "mwpm", "-", n, int(outcomes.sum()), ler, perc, res.seconds))
            _record(outcomes_store, cfg, "mwpm", n, outcomes)
            postfix["mwpm/c"] = f"{perc:.4f}"

        if "mwpm-rl" in args.decoders and data.dem_rl is not None:
            res = PyMatchingDecoder.from_dem(data.dem_rl).decode_batch(dets)
            outcomes, ler, perc = _score(res.predictions, obs, cfg.rounds)
            rows.append(EvalRow(cfg.distance, cfg.basis, cfg.orientation, cfg.rounds,
                                "mwpm-rl", "-", n, int(outcomes.sum()), ler, perc, res.seconds))
            _record(outcomes_store, cfg, "mwpm-rl", n, outcomes)

        for dname, spec in _PANEL_DECODERS.items():
            if dname in args.decoders and data.dem_si1000 is not None:
                res = _resolve(spec).from_dem(data.dem_si1000).decode_batch(dets)
                outcomes, ler, perc = _score(res.predictions, obs, cfg.rounds)
                rows.append(EvalRow(cfg.distance, cfg.basis, cfg.orientation, cfg.rounds,
                                    dname, "-", n, int(outcomes.sum()), ler, perc, res.seconds))
                _record(outcomes_store, cfg, dname, n, outcomes)
                postfix[f"{dname[:4]}/c"] = f"{perc:.4f}"

        if want_ising:
            if cfg.rounds < IsingPreDecoder.MIN_ROUNDS:
                rows.append(EvalRow(cfg.distance, cfg.basis, cfg.orientation, cfg.rounds,
                                    "ising", model_info.name, n, 0, float("nan"), float("nan"),
                                    0.0, note=f"skipped: rounds<{IsingPreDecoder.MIN_ROUNDS}"))
            else:
                dec = IsingPreDecoder(model, data.circuit, cfg.basis, cfg.distance, cfg.rounds,
                                      device, rotation=args.rotation, syn_noise=args.syn_noise)
                try:
                    res = dec.decode_batch(dets)
                except Exception as exc:  # noqa: BLE001 -- Willow->CSS order open (RQ4)
                    # The residual matcher lives in the model's CSS MemoryCircuit
                    # layout; mapping Willow's XZZX detectors into that order is the
                    # open step-4 problem. Record it instead of crashing the shard.
                    rows.append(EvalRow(cfg.distance, cfg.basis, cfg.orientation, cfg.rounds,
                                        "ising", model_info.name, n, 0, float("nan"), float("nan"),
                                        0.0, note=f"ising-failed: {type(exc).__name__}"))
                else:
                    outcomes, ler, perc = _score(res.predictions, obs, cfg.rounds)
                    rows.append(EvalRow(cfg.distance, cfg.basis, cfg.orientation, cfg.rounds,
                                        "ising", model_info.name, n, int(outcomes.sum()), ler, perc,
                                        res.seconds,
                                        note=f"R={model_info.receptive_field},rot={args.rotation}"))
                    _record(outcomes_store, cfg, "ising", n, outcomes)
                    postfix["ising/c"] = f"{perc:.4f}"

        if postfix:
            bar.set_postfix(postfix)
    bar.close()

    if outcomes_store is not None and outcomes_store:
        args.outcomes_dir.mkdir(parents=True, exist_ok=True)
        npz_path = args.outcomes_dir / (args.out.stem + ".outcomes.npz")
        np.savez_compressed(npz_path, **outcomes_store)
        print(f"wrote per-shot outcomes for {len(outcomes_store)} shards -> {npz_path}",
              file=sys.stderr)
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
    p.add_argument("--model", choices=("fast", "accurate", "custom"), default="fast",
                   help="Ising model variant (default: fast). 'custom' needs --checkpoint and --model-id.")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="explicit weights path (overrides the default fast/accurate lookup)")
    p.add_argument("--model-id", type=int, default=None,
                   help="public model id 1-5 for the checkpoint's architecture (required with --model custom)")
    p.add_argument("--distances", type=_int_list, default=None, help="e.g. 3,5,7 (default: all)")
    p.add_argument("--bases", type=_str_list, default=None, help="X,Z (default: both)")
    p.add_argument("--patches", type=_str_list, default=None,
                   help="orientation labels, e.g. q6_7,q4_5 (default: all patches for the distance)")
    p.add_argument("--rounds", type=_int_list, default=None, help="exact round counts, e.g. 10,30")
    p.add_argument("--max-rounds", type=int, default=None, help="only configs with rounds <= this")
    p.add_argument("--shots", type=int, default=None, help="subsample N shots/config (default: all 50000)")
    p.add_argument("--rotation", default="XV", help="Ising code rotation: XV|XH|ZV|ZH (default: XV)")
    p.add_argument("--syn-noise", type=float, default=1e-3, help="uniform noise for the residual matcher DEM")
    p.add_argument("--device", default="cpu", help="auto|cpu|cuda|cuda:N|mps (default: cpu — fastest here)")
    p.add_argument("--source", choices=("local", "hf", "synth"), default="local",
                   help="local Willow tree (default), the published HF dataset, or a "
                        "sampled synthetic noise rung")
    p.add_argument("--hf-repo", default="ShayManor/willow-surface-code-detection-events",
                   help="HF dataset repo when --source hf or synth (circuit+DEM fetch)")
    p.add_argument("--rung", choices=("uniform", "si1000", "syndrome", "fit"), default=None,
                   help="synthetic noise rung (required when --source synth)")
    p.add_argument("--synth-p", type=float, default=2e-3,
                   help="uniform-rung per-error probability (default: 2e-3)")
    p.add_argument("--synth-shots", type=int, default=20000,
                   help="shots to SAMPLE per synthetic config (default: 20000)")
    p.add_argument("--seed", type=int, default=1234,
                   help="base seed for synthetic sampling (default: 1234)")
    p.add_argument("--data-dir", type=Path, default=WILLOW_RAW_DIR)
    p.add_argument("--out", type=Path, default=OUTPUTS_DIR / "eval_results.csv")
    p.add_argument("--outcomes-dir", type=Path, default=None,
                   help="also dump per-shot logical-error outcomes (bit-packed .npz) here, "
                        "one file per --out CSV, so bootstrap/rank-CIs can be computed locally")
    p.add_argument("--limit", type=int, default=None, help="cap number of configs (smoke test)")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)

    bad = set(args.decoders) - set(ALL_DECODERS)
    if bad:
        p.error(f"unknown decoders {sorted(bad)}; choose from {ALL_DECODERS}")
    if args.model == "custom" and (args.checkpoint is None or args.model_id is None):
        p.error("--model custom requires both --checkpoint and --model-id")
    if args.source == "synth" and args.rung is None:
        p.error("--source synth requires --rung uniform|si1000|syndrome|fit")

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
