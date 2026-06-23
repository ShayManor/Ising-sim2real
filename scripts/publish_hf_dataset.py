#!/usr/bin/env python3
"""Run the Willow ingest pipeline and publish the result as a HF dataset.

For every one of the 420 Willow configurations this:

1. Loads the raw inputs (circuit_ideal.stim, measurements.b8, sweep_bits.b8).
2. Derives detection events + observable flips via Stim m2d (the ingest pipeline).
3. Validation gate: asserts the derived arrays equal the dataset's shipped
   detection_events.b8 / obs_flips_actual.b8 byte-for-byte. Refuses to publish a
   config whose ingest does not reproduce the shipped ground truth.
4. Writes one zstd-compressed Parquet shard per config (one row per shot).
5. Uploads the staging folder to a Hugging Face dataset repo.

Parquet schema (consistent across all shards):
    distance     int16
    basis        string   ("X" | "Z")
    rounds       int16
    orientation  string   (patch label, e.g. "q4_5")
    shot         int32    (shot index within the config)
    detectors    list<bool>   (length == num_detectors for that config)
    observable   bool         (the logical flip to predict against)

Usage:
    python scripts/publish_hf_dataset.py --limit 4 --no-upload   # quick local check
    python scripts/publish_hf_dataset.py                          # full run + upload
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ising_sim2real.ingest.dataset import discover_configs
from ising_sim2real.ingest.detectors import measurements_to_detectors
from ising_sim2real.ingest.willow import (
    _RL_PATHWAY,
    _SI1000_PATHWAY,
    WillowConfig,
    config_dir,
    load_run,
    load_shipped_detection_data,
)
from ising_sim2real.paths import OUTPUTS_DIR, WILLOW_RAW_DIR

DEFAULT_REPO_ID = "ShayManor/willow-surface-code-detection-events"

# Arrow schema shared by every shard so the shards load as one logical dataset.
SCHEMA = pa.schema(
    [
        pa.field("distance", pa.int16()),
        pa.field("basis", pa.string()),
        pa.field("rounds", pa.int16()),
        pa.field("orientation", pa.string()),
        pa.field("shot", pa.int32()),
        pa.field("detectors", pa.list_(pa.bool_())),
        pa.field("observable", pa.bool_()),
    ]
)


def config_stem(cfg: WillowConfig) -> str:
    """Shared stem identifying a config across parquet shards and bundle files."""
    return f"d{cfg.distance}_at_{cfg.orientation}__{cfg.basis}__r{cfg.rounds:03d}"


def shard_name(cfg: WillowConfig) -> str:
    """Stable, sortable filename for a config's Parquet shard."""
    return f"{config_stem(cfg)}.parquet"


def build_bundle(staging: Path, configs: list[WillowConfig]) -> int:
    """Stage the decoding bundle: ideal circuit + shipped DEMs (gzipped) per config.

    These are the inputs the decoder panel needs that the detection-event parquet
    does not carry: the circuit (for the Ising lattice layout) and the shipped
    SI1000 / RL detector error models (for the MWPM baselines). DEMs are text and
    gzip ~10x, so they ship compressed and are decompressed on load.
    """
    circuits = staging / "circuits"
    dems = staging / "dems"
    for d in (circuits, dems):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    n = 0
    for i, cfg in enumerate(configs, 1):
        leaf = config_dir(WILLOW_RAW_DIR, cfg)
        stem = config_stem(cfg)
        shutil.copyfile(leaf / "circuit_ideal.stim", circuits / f"{stem}.stim")
        for pathway, tag in ((_SI1000_PATHWAY, "si1000"), (_RL_PATHWAY, "rl")):
            src = leaf / "decoding_results" / pathway / "error_model.dem"
            if not src.exists():
                continue
            with open(src, "rb") as fin, gzip.open(dems / f"{stem}.{tag}.dem.gz", "wb") as fout:
                shutil.copyfileobj(fin, fout)
        n += 1
        if i % 50 == 0 or i == len(configs):
            print(f"[{i:3d}/{len(configs)}] bundled {stem}")
    return n


def build_table(cfg: WillowConfig, detectors: np.ndarray, observable: np.ndarray) -> pa.Table:
    """Pack one config's (shots, num_detectors) detection field into an Arrow table."""
    shots, num_detectors = detectors.shape

    # list<bool>: a flat bool values buffer + uniform offsets (every row is the
    # same width for a given config). Built directly from numpy, no Python lists.
    flat = pa.array(np.ascontiguousarray(detectors).reshape(-1), type=pa.bool_())
    offsets = pa.array(
        np.arange(0, (shots + 1) * num_detectors, num_detectors, dtype=np.int32)
    )
    det_list = pa.ListArray.from_arrays(offsets, flat)

    return pa.table(
        {
            "distance": pa.array(np.full(shots, cfg.distance, dtype=np.int16)),
            "basis": pa.array([cfg.basis] * shots, type=pa.string()),
            "rounds": pa.array(np.full(shots, cfg.rounds, dtype=np.int16)),
            "orientation": pa.array([cfg.orientation] * shots, type=pa.string()),
            "shot": pa.array(np.arange(shots, dtype=np.int32)),
            "detectors": det_list,
            "observable": pa.array(observable.reshape(-1), type=pa.bool_()),
        },
        schema=SCHEMA,
    )


def ingest_config(cfg: WillowConfig) -> pa.Table:
    """Run the ingest pipeline for one config and apply the validation gate."""
    run = load_run(WILLOW_RAW_DIR, cfg)
    derived = measurements_to_detectors(run.circuit, run.measurements, sweep_bits=run.sweep_bits)
    shipped = load_shipped_detection_data(WILLOW_RAW_DIR, cfg)

    # Validation gate: ingest must reproduce the shipped ground truth exactly.
    if not np.array_equal(derived.detectors, shipped.detectors):
        raise AssertionError(f"{shard_name(cfg)}: derived detectors != shipped")
    if not np.array_equal(derived.observables, shipped.observables):
        raise AssertionError(f"{shard_name(cfg)}: derived observables != shipped")

    return build_table(cfg, derived.detectors, derived.observables)


def write_card(staging: Path, repo_id: str, n_configs: int) -> None:
    """Write the dataset card (README.md) into the staging folder."""
    card = f"""---
license: apache-2.0
task_categories:
  - other
tags:
  - quantum-error-correction
  - surface-code
  - willow
pretty_name: Willow Surface-Code Detection Events
configs:
  - config_name: default
    data_files: data/*.parquet
---

# Willow Surface-Code Detection Events (ingested)

Detection events and logical-observable flips derived from Google's Willow
below-threshold surface-code dataset (Zenodo
[10.5281/zenodo.13273331](https://doi.org/10.5281/zenodo.13273331)), rotated
surface code at distances 3, 5, 7 in X and Z memory.

Each row is one experimental **shot**. Detection events are derived from the raw
device measurement records with Stim's measurement-to-detector converter, using
the per-shot sweep bits, and validated to reproduce the dataset's shipped
`detection_events.b8` / `obs_flips_actual.b8` **byte-for-byte** for all
{n_configs} configurations.

## Columns

| column | type | meaning |
|---|---|---|
| `distance` | int16 | code distance d (3, 5, 7) |
| `basis` | string | `X` or `Z` memory |
| `rounds` | int16 | number of QEC cycles |
| `orientation` | string | patch location label on the chip (e.g. `q4_5`) |
| `shot` | int32 | shot index within the configuration |
| `detectors` | list&lt;bool&gt; | which detectors fired (length `num_detectors`, `= 2 * rounds * (d^2-1)/2`) |
| `observable` | bool | ground-truth logical flip the decoder must predict |

One Parquet shard per `(distance, orientation, basis, rounds)` configuration under
`data/`. The decoder sees only `detectors`; `observable` is the held-out answer key.

## Decoding bundle

Alongside the detection events, each config ships the inputs a decoder panel needs,
keyed by the same `<stem>` (`d{{D}}_at_{{orient}}__{{basis}}__r{{rounds:03d}}`):

- `circuits/<stem>.stim` — the ideal (noiseless) annotated circuit.
- `dems/<stem>.si1000.dem.gz` — shipped SI1000 detector error model (gzipped).
- `dems/<stem>.rl.dem.gz` — shipped RL-optimized detector error model (gzipped).

With these, the whole evaluation runs off this dataset with no local copy of the
12 GB Willow tree.

## Load

```python
from datasets import load_dataset
ds = load_dataset("{repo_id}", split="train")     # detection events
```
"""
    (staging / "README.md").write_text(card)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--limit", type=int, default=None, help="process only the first N configs")
    ap.add_argument("--no-upload", action="store_true", help="build Parquet shards but skip the HF upload")
    ap.add_argument("--private", action="store_true", help="create the HF repo as private")
    ap.add_argument("--bundle", action="store_true",
                    help="instead of detection events, build+upload the decoding bundle "
                         "(ideal circuits + shipped DEMs) additively to the same repo")
    args = ap.parse_args()

    configs = discover_configs(WILLOW_RAW_DIR)
    if args.limit is not None:
        configs = configs[: args.limit]

    if args.bundle:
        return _run_bundle(args, configs)

    staging = OUTPUTS_DIR / "hf_dataset"
    data_dir = staging / "data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for i, cfg in enumerate(configs, 1):
        table = ingest_config(cfg)
        out = data_dir / shard_name(cfg)
        pq.write_table(table, out, compression="zstd", use_dictionary=["basis", "orientation"])
        total_rows += table.num_rows
        print(f"[{i:3d}/{len(configs)}] {out.name}  rows={table.num_rows}  "
              f"detectors={len(table['detectors'][0])}  size={out.stat().st_size/1e6:.1f}MB")

    write_card(staging, args.repo_id, len(configs))
    shard_bytes = sum(p.stat().st_size for p in data_dir.glob("*.parquet"))
    print(f"\nValidation gate passed for all {len(configs)} configs.")
    print(f"Wrote {len(configs)} shards, {total_rows:,} rows, {shard_bytes/1e9:.2f} GB to {staging}")

    if args.no_upload:
        print("--no-upload: skipping HF upload.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    print(f"Uploading {staging} -> https://huggingface.co/datasets/{args.repo_id} ...")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(staging),
        commit_message="Add ingested Willow detection events (all 420 configs)",
    )
    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")


def _run_bundle(args, configs: list[WillowConfig]) -> None:
    staging = OUTPUTS_DIR / "hf_bundle"
    n = build_bundle(staging, configs)
    nbytes = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
    print(f"\nStaged decoding bundle for {n} configs ({nbytes/1e6:.0f} MB) at {staging}")

    if args.no_upload:
        print("--no-upload: skipping HF upload.")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=args.private, exist_ok=True)
    print(f"Uploading bundle (circuits/ + dems/) -> https://huggingface.co/datasets/{args.repo_id} ...")
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(staging),
        commit_message="Add decoding bundle: ideal circuits + shipped SI1000/RL DEMs (gzipped)",
    )
    print(f"Done: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
