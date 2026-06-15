"""Canonical filesystem locations for the harness.

All paths are derived from the repo root so the package works regardless of the
current working directory. The vendored NVIDIA repo location can be overridden
with the ISING_DECODING_ROOT environment variable.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/ising_sim2real/paths.py -> repo root is two parents up from this file's dir.
REPO_ROOT = Path(__file__).resolve().parents[2]

THIRD_PARTY = REPO_ROOT / "third_party"

# Vendored clone of github.com/NVIDIA/Ising-Decoding (gitignored).
ISING_ROOT = Path(os.environ.get("ISING_DECODING_ROOT", THIRD_PARTY / "Ising-Decoding"))
ISING_CODE = ISING_ROOT / "code"

# Where setup_ising.py copies the pretrained .pt weights so the project has a
# stable path independent of the clone's internals.
MODELS_DIR = REPO_ROOT / "models" / "ising"

# Willow hardware data and experiment outputs (both gitignored).
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"

# The Google Willow below-threshold dataset, unpacked at the repo root
# (Zenodo 10.5281/zenodo.13273331). Overridable for tests / alternate locations.
WILLOW_RAW_DIR = Path(
    os.environ.get("WILLOW_RAW_DIR", REPO_ROOT / "google_105Q_surface_code_d3_d5_d7")
)
