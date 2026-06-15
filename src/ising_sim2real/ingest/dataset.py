"""Enumerate the Willow configurations to evaluate.

SCAFFOLD ONLY -- interfaces are defined; bodies are not implemented yet.

Provides the cartesian product over (distance, basis, orientation, rounds) that
the harness iterates when it runs every decoder on every configuration.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from ising_sim2real.ingest.willow import WillowConfig

# Patch dirs look like "d3_at_q4_5": distance, then the patch location label.
_PATCH_RE = re.compile(r"^d(\d+)_at_(.+)$")
# Cycle dirs look like "r01", "r13", "r110".
_CYCLES_RE = re.compile(r"^r(\d+)$")


def discover_configs(data_dir: Path) -> list[WillowConfig]:
    """Scan the dataset directory and return every available configuration.

    Walks the ``<patch>/<basis>/<cycles>`` tree documented in the dataset README
    and yields one :class:`WillowConfig` per leaf. The result is sorted so the
    enumeration is deterministic across runs.
    """
    data_dir = Path(data_dir)
    configs: list[WillowConfig] = []

    for patch_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        patch_match = _PATCH_RE.match(patch_dir.name)
        if patch_match is None:
            continue
        distance = int(patch_match.group(1))
        orientation = patch_match.group(2)

        for basis in ("X", "Z"):
            basis_dir = patch_dir / basis
            if not basis_dir.is_dir():
                continue
            for cycles_dir in sorted(c for c in basis_dir.iterdir() if c.is_dir()):
                cycles_match = _CYCLES_RE.match(cycles_dir.name)
                if cycles_match is None:
                    continue
                configs.append(
                    WillowConfig(
                        distance=distance,
                        basis=basis,
                        rounds=int(cycles_match.group(1)),
                        orientation=orientation,
                    )
                )

    return sorted(
        configs, key=lambda c: (c.distance, c.orientation, c.basis, c.rounds)
    )


def iter_configs(
    data_dir: Path,
    distances: tuple[int, ...] = (3, 5, 7),
    bases: tuple[str, ...] = ("X", "Z"),
) -> Iterator[WillowConfig]:
    """Yield the subset of discovered configs matching the given filters."""
    for cfg in discover_configs(data_dir):
        if cfg.distance in distances and cfg.basis in bases:
            yield cfg
