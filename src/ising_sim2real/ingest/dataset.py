"""Enumerate the Willow configurations to evaluate.

SCAFFOLD ONLY -- interfaces are defined; bodies are not implemented yet.

Provides the cartesian product over (distance, basis, orientation, rounds) that
the harness iterates when it runs every decoder on every configuration.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from ising_sim2real.ingest.willow import WillowConfig


def discover_configs(data_dir: Path) -> list[WillowConfig]:
    """Scan the dataset directory and return every available configuration."""
    raise NotImplementedError("Ingest pipeline not implemented yet (method step 1).")


def iter_configs(
    data_dir: Path,
    distances: tuple[int, ...] = (3, 5, 7),
    bases: tuple[str, ...] = ("X", "Z"),
) -> Iterator[WillowConfig]:
    """Yield the subset of discovered configs matching the given filters."""
    raise NotImplementedError("Ingest pipeline not implemented yet (method step 1).")
