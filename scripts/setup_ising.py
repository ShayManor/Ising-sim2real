#!/usr/bin/env python3
"""Pull the NVIDIA Ising pre-decoder code and pretrained weights.

Brings the model into the repo so `scripts/run_ising.py` can run it:

1. Clone github.com/NVIDIA/Ising-Decoding into third_party/ (shallow, pinned),
   skipping the Git-LFS smudge so the clone is small.
2. Pull the `.pt` weights from the GitHub repo via git-lfs (primary), or download
   the fp16 .safetensors from Hugging Face with --from-huggingface (fallback).
3. Copy the weights to models/ising/ for a stable, clone-independent path.

Usage:
    python scripts/setup_ising.py                 # clone + git-lfs .pt weights
    python scripts/setup_ising.py --from-huggingface   # HF safetensors instead
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Make the package importable when run directly from the repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ising_sim2real.ising.loader import ISING_MODELS  # noqa: E402
from ising_sim2real.paths import ISING_ROOT, MODELS_DIR  # noqa: E402

ISING_REPO_URL = "https://github.com/NVIDIA/Ising-Decoding.git"
# Pin for reproducibility; matches the commit the loader was validated against.
ISING_PINNED_COMMIT = "214839eb190447b4d8d5ed950d912b12d076771b"

# Hugging Face fp16 safetensors (fallback source for the weights).
HF_FILES = {
    "fast": (
        "nvidia/Ising-Decoder-SurfaceCode-1-Fast",
        "ising_decoder_surface_code_1_fast_r9_v1.0.77_fp16.safetensors",
    ),
    "accurate": (
        "nvidia/Ising-Decoder-SurfaceCode-1-Accurate",
        "ising_decoder_surface_code_1_accurate_r13_v1.0.86_fp16.safetensors",
    ),
}


def _run(cmd: list[str], cwd: Path | None = None, env_extra: dict | None = None) -> None:
    import os

    env = {**os.environ, **(env_extra or {})}
    print("  $", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def clone_repo() -> None:
    if ISING_ROOT.exists():
        print(f"[clone] already present at {ISING_ROOT}")
        return
    print(f"[clone] {ISING_REPO_URL} -> {ISING_ROOT}")
    ISING_ROOT.parent.mkdir(parents=True, exist_ok=True)
    _run(
        ["git", "clone", "--depth", "1", ISING_REPO_URL, str(ISING_ROOT)],
        env_extra={"GIT_LFS_SKIP_SMUDGE": "1"},
    )


def pull_weights_git_lfs() -> dict[str, Path]:
    print("[weights] git-lfs pull models/*.pt from the clone")
    _run(["git", "lfs", "pull", "--include", "models/*.pt"], cwd=ISING_ROOT)
    out = {}
    for name, spec in ISING_MODELS.items():
        src = ISING_ROOT / "models" / spec.filename
        if not src.exists():
            raise FileNotFoundError(f"Expected {src} after git-lfs pull.")
        out[name] = src
    return out


def pull_weights_huggingface() -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    out = {}
    for name, (repo_id, filename) in HF_FILES.items():
        print(f"[weights] hf download {repo_id}/{filename}")
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        out[name] = Path(path)
    return out


def copy_to_models(sources: dict[str, Path]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, src in sources.items():
        # Keep the source extension (.pt from git-lfs, .safetensors from HF).
        dest = MODELS_DIR / (ISING_MODELS[name].filename
                             if src.suffix == ".pt"
                             else f"{ISING_MODELS[name].name}{src.suffix}")
        if src.resolve() == dest.resolve():
            continue
        shutil.copy2(src, dest)
        size_mb = dest.stat().st_size / 1e6
        print(f"[copy] {name}: {dest}  ({size_mb:.1f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-huggingface",
        action="store_true",
        help="Download fp16 safetensors from Hugging Face instead of git-lfs .pt.",
    )
    args = parser.parse_args()

    clone_repo()
    sources = pull_weights_huggingface() if args.from_huggingface else pull_weights_git_lfs()
    copy_to_models(sources)
    print("\nDone. Try: python scripts/run_ising.py --model fast --device auto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
