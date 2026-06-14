#!/usr/bin/env python3
"""Run the PyMatching baseline on Google Willow hardware data.

Reports logical error rate per cycle for every available (distance, basis,
orientation, rounds) configuration found under the data directory.
Ground truth is obs_flips_actual.b8; the DEM is the bundled SI1000-prior
error_model.dem from each run's decoding_results/ subdirectory.

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --data-dir data/google_105Q_surface_code_d3_d5_d7
    python scripts/run_baseline.py --distance 3 --basis X
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ising_sim2real.decoders.base import PyMatchingDecoder  # noqa: E402
from ising_sim2real.ingest.dataset import iter_configs       # noqa: E402
from ising_sim2real.ingest.willow import load_run            # noqa: E402
from ising_sim2real.paths import DATA_DIR                    # noqa: E402


def _find_willow_dir(data_dir: Path) -> Path | None:
    """Return the first google_* subdirectory, or data_dir itself if it has configs."""
    candidate = next(data_dir.glob("google_*"), None)
    if candidate is not None and candidate.is_dir():
        return candidate
    # data_dir might already be the google_* directory
    if any(data_dir.glob("d*_at_*/*/r*/metadata.json")):
        return data_dir
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Root data directory (default: repo data/); auto-detects google_* subdir",
    )
    parser.add_argument("--distance", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--basis", nargs="+", default=["X", "Z"])
    args = parser.parse_args()

    willow_dir = _find_willow_dir(args.data_dir)
    if willow_dir is None:
        print(f"No Willow data found under {args.data_dir}")
        print("Download from Zenodo DOI 10.5281/zenodo.13273331")
        return 1

    header = (
        f"{'d':>3}  {'T':>4}  {'basis':>5}  {'orient':>8}  "
        f"{'shots':>7}  {'errors':>7}  {'LER/cycle':>12}  {'decode_ms':>10}"
    )
    print(header)
    print("-" * len(header))

    found = 0
    for cfg in iter_configs(willow_dir, tuple(args.distance), tuple(args.basis)):
        run = load_run(willow_dir, cfg)
        if run.dem_si1000 is None:
            print(
                f"  d={cfg.distance} {cfg.basis} r={cfg.rounds} {cfg.orientation}: "
                f"no DEM found, skipping"
            )
            continue

        decoder = PyMatchingDecoder.from_dem(run.dem_si1000)
        result = decoder.decode_batch(run.detection_data.detectors)

        shots = run.detection_data.observables.shape[0]
        errors = int(
            (result.predictions != run.detection_data.observables).any(axis=1).sum()
        )
        ler_per_cycle = errors / (shots * cfg.rounds)

        # Guardrail: always report code distance d and rounds T together.
        print(
            f"{cfg.distance:>3}  {cfg.rounds:>4}  {cfg.basis:>5}  {cfg.orientation:>8}  "
            f"{shots:>7}  {errors:>7}  {ler_per_cycle:>12.5%}  {result.seconds * 1e3:>10.1f}"
        )
        found += 1

    if found == 0:
        print(f"No configs matched distance={args.distance} basis={args.basis}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
