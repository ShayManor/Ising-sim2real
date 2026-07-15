#!/usr/bin/env python3
"""Fit ONE (draw, patch) tile of the joint bootstrap: resample the patch's
fit-set shots for this draw, refit all 23 free params, and write the result to
results/bootstrap/models/draw_{draw}/<patch_key>.json.

Runs as one task of the slurm/fit_bootstrap.sbatch array (index 0..B*NPATCH-1).
NOT a local script -- fits are heavy (see CLAUDE.md's no-local-heavy-runs rule).

    python -m scripts.fit_bootstrap_draw --index 0          # one tile
    python -m scripts.fit_bootstrap_draw --index 4 --resume # skip if JSON exists
"""

from __future__ import annotations

import argparse
import json

from ising_sim2real.ingest.hf import DEFAULT_HF_REPO
from ising_sim2real.ingest.willow import patch_key
from ising_sim2real.paths import REPO_ROOT
from scripts.fit_noise_models import _all_patches, fit_one_patch

OUT_ROOT = REPO_ROOT / "results" / "bootstrap" / "models"


def _patches(repo: str = DEFAULT_HF_REPO) -> list[tuple[str, int]]:
    """Index-decoding patch list. Reads ``OUT_ROOT/patches.json`` if present so
    array tiles need NO HuggingFace listing API call -- the per-tile
    ``list_repo_files`` was 429-ing at 16-way concurrency, and this makes the
    tile path ``HF_HUB_OFFLINE``-safe. Otherwise queries HF once and caches it;
    pre-generate the cache online (one call) before launching an offline array."""
    cache = OUT_ROOT / "patches.json"
    if cache.exists():
        return [(o, d) for o, d in json.loads(cache.read_text())]
    patches = _all_patches(repo)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(patches))
    return patches


def tile(index: int, patches: list[tuple[str, int]]) -> tuple[int, str, int]:
    """Array index -> (draw, orientation, distance). Patches cycle fastest so
    every draw is a contiguous block of NPATCH tasks."""
    npatch = len(patches)
    draw = index // npatch
    orientation, distance = patches[index % npatch]
    return draw, orientation, distance


def run_index(index: int, repo: str = DEFAULT_HF_REPO, resume: bool = False) -> None:
    draw, orientation, distance = tile(index, _patches(repo))
    key = patch_key(distance, orientation)
    out_dir = OUT_ROOT / f"draw_{draw:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{key}.json"
    if resume and out_path.exists():
        print(f"skip (resume): {out_path}")
        return
    print(f"fitting draw={draw} patch={key} (index={index}) ...")
    params = fit_one_patch(orientation, distance, repo=repo, draw=draw)
    out_path.write_text(json.dumps(params, sort_keys=True, indent=2))
    print(f"  -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", type=int, required=True, help="array task id, 0..B*NPATCH-1")
    ap.add_argument("--repo", default=DEFAULT_HF_REPO)
    ap.add_argument("--resume", action="store_true", help="skip if the tile JSON already exists")
    args = ap.parse_args()
    run_index(args.index, repo=args.repo, resume=args.resume)


if __name__ == "__main__":
    main()
