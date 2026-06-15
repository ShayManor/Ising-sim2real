"""Shared fixtures for the test suite.

Tests run against the real Willow dataset unpacked at the repo root. When the
dataset is absent (e.g. a fresh clone before download), data-backed tests skip
rather than fail, so the suite stays runnable everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ising_sim2real.ingest.willow import WillowConfig
from ising_sim2real.paths import WILLOW_RAW_DIR


@pytest.fixture(scope="session")
def willow_dir() -> Path:
    """Root of the unpacked Willow dataset, or skip if it is not present."""
    if not WILLOW_RAW_DIR.exists():
        pytest.skip(f"Willow dataset not found at {WILLOW_RAW_DIR}")
    return WILLOW_RAW_DIR


@pytest.fixture(scope="session")
def d3_config() -> WillowConfig:
    """A small, fast configuration: distance 3, Z basis, 1 round."""
    return WillowConfig(distance=3, basis="Z", rounds=1, orientation="q4_5")
