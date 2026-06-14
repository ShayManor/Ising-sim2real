"""Enumerate the Willow configurations to evaluate.

Provides the cartesian product over (distance, basis, orientation, rounds) that
the harness iterates when it runs every decoder on every configuration.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ising_sim2real.ingest.willow import WillowConfig


def discover_configs(data_dir: Path) -> list[WillowConfig]:
    """Scan the dataset directory and return every available configuration.

    Expects the Willow layout:
        data_dir / d{D}_at_{orientation} / {basis} / r{R} / metadata.json
    """
    configs: list[WillowConfig] = []
    for meta_path in sorted(data_dir.glob("d*_at_*/*/r*/metadata.json")):
        meta = json.loads(meta_path.read_text())
        top_dir = meta_path.parents[2].name  # e.g. "d3_at_q4_5"
        _, _, orientation = top_dir.partition("_at_")
        configs.append(
            WillowConfig(
                distance=int(meta["distance"]),
                basis=str(meta["basis"]),
                rounds=int(meta["rounds"]),
                orientation=orientation,
            )
        )
    return configs


def iter_configs(
    data_dir: Path,
    distances: tuple[int, ...] = (3, 5, 7),
    bases: tuple[str, ...] = ("X", "Z"),
) -> Iterator[WillowConfig]:
    """Yield the subset of discovered configs matching the given filters."""
    for cfg in discover_configs(data_dir):
        if cfg.distance in distances and cfg.basis in bases:
            yield cfg
