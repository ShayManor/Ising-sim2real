"""Load configurations from the published Hugging Face dataset.

Mirror of the local ingest path, but sourcing everything from
``ShayManor/willow-surface-code-detection-events`` so a run needs no local copy
of the 12 GB Willow tree. Per config it pulls three small artifacts:

    data/<stem>.parquet         detection events + observable flips (already ingested)
    circuits/<stem>.stim        ideal circuit (for the Ising lattice layout)
    dems/<stem>.{si1000,rl}.dem.gz   shipped DEMs (for the MWPM baselines)

``<stem>`` is ``d{D}_at_{orient}__{basis}__r{rounds:03d}`` -- the same key the
publish script writes. Files are fetched lazily through the HF hub cache, so only
the configs a run touches are downloaded.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
import stim

from ising_sim2real.ingest.willow import WillowConfig

DEFAULT_HF_REPO = "ShayManor/willow-surface-code-detection-events"

# data/<stem>.parquet where <stem> == d{D}_at_{orientation}__{basis}__r{rounds:03d}
_STEM_RE = re.compile(r"^d(\d+)_at_(.+)__([XZ])__r(\d+)$")


def discover_configs_hf(repo: str = DEFAULT_HF_REPO) -> list[WillowConfig]:
    """Enumerate configs from the HF dataset's parquet listing (no local tree).

    The ``--source hf`` path must not touch the local Willow directory at all, so
    config discovery comes from the repo file list rather than ``data_dir``.
    """
    from huggingface_hub import HfApi

    configs: list[WillowConfig] = []
    for f in HfApi().list_repo_files(repo, repo_type="dataset"):
        if not (f.startswith("data/") and f.endswith(".parquet")):
            continue
        m = _STEM_RE.match(f[len("data/"):-len(".parquet")])
        if m is None:
            continue
        configs.append(
            WillowConfig(
                distance=int(m.group(1)),
                orientation=m.group(2),
                basis=m.group(3),
                rounds=int(m.group(4)),
            )
        )
    if not configs:
        raise FileNotFoundError(
            f"No data/*.parquet shards found on {repo}; is the dataset published?"
        )
    return sorted(configs, key=lambda c: (c.distance, c.orientation, c.basis, c.rounds))


@dataclass
class HFConfigData:
    """Everything the decoder panel needs for one config, sourced from HF."""

    circuit: stim.Circuit
    detectors: np.ndarray              # (shots, num_detectors), bool
    observables: np.ndarray            # (shots, num_observables), bool
    dem_si1000: stim.DetectorErrorModel | None
    dem_rl: stim.DetectorErrorModel | None


def _stem(cfg: WillowConfig) -> str:
    return f"d{cfg.distance}_at_{cfg.orientation}__{cfg.basis}__r{cfg.rounds:03d}"


def _columns_to_numpy(table) -> tuple[np.ndarray, np.ndarray]:
    """Decode the parquet shard to numpy WITHOUT going through Python lists.

    ``detectors`` is a list<bool> column of uniform width per config; reading it
    via ``.to_pylist()`` materializes ~50k Python lists and is ~160x slower than
    reshaping the Arrow child buffer directly (32s vs 0.2s on a d5/r250 shard).
    """
    det = table["detectors"].combine_chunks()
    flat = det.values.to_numpy(zero_copy_only=False).astype(bool, copy=False)
    offsets = det.offsets.to_numpy()
    width = int(offsets[1] - offsets[0]) if len(offsets) > 1 else 0
    detectors = flat.reshape(len(det), width)
    observables = (
        table["observable"].combine_chunks().to_numpy(zero_copy_only=False)
        .astype(bool, copy=False).reshape(-1, 1)
    )
    return detectors, observables


def _download(repo: str, path: str) -> str | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        return hf_hub_download(repo_id=repo, filename=path, repo_type="dataset")
    except EntryNotFoundError:
        return None


def _load_dem_gz(path: str | None) -> stim.DetectorErrorModel | None:
    if path is None:
        return None
    with gzip.open(path, "rt") as f:
        return stim.DetectorErrorModel(f.read())


def load_config_from_hf(cfg: WillowConfig, repo: str = DEFAULT_HF_REPO) -> HFConfigData:
    """Fetch one configuration's decode inputs from the HF dataset."""
    stem = _stem(cfg)

    table = pq.read_table(_download(repo, f"data/{stem}.parquet"))
    detectors, observables = _columns_to_numpy(table)

    circuit_path = _download(repo, f"circuits/{stem}.stim")
    if circuit_path is None:
        raise FileNotFoundError(
            f"circuits/{stem}.stim missing on {repo}; upload the decoding bundle "
            "with `python scripts/publish_hf_dataset.py --bundle`."
        )
    circuit = stim.Circuit.from_file(circuit_path)

    return HFConfigData(
        circuit=circuit,
        detectors=detectors,
        observables=observables,
        dem_si1000=_load_dem_gz(_download(repo, f"dems/{stem}.si1000.dem.gz")),
        dem_rl=_load_dem_gz(_download(repo, f"dems/{stem}.rl.dem.gz")),
    )
