"""Tests for enumerating the Willow configurations on disk."""

from __future__ import annotations

from pathlib import Path

from ising_sim2real.ingest.dataset import discover_configs, iter_configs
from ising_sim2real.ingest.willow import WillowConfig, config_dir


def test_discover_finds_known_config(willow_dir: Path) -> None:
    configs = discover_configs(willow_dir)
    assert WillowConfig(distance=3, basis="Z", rounds=1, orientation="q4_5") in configs


def test_every_discovered_config_resolves_to_a_real_leaf(willow_dir: Path) -> None:
    configs = discover_configs(willow_dir)
    assert len(configs) > 0
    for cfg in configs:
        leaf = config_dir(willow_dir, cfg)
        assert (leaf / "circuit_ideal.stim").exists(), cfg


def test_discovered_distances_and_bases_are_valid(willow_dir: Path) -> None:
    configs = discover_configs(willow_dir)
    assert {c.distance for c in configs} <= {3, 5, 7}
    assert {c.basis for c in configs} == {"X", "Z"}
    # The dataset ships distance 7 only as a single patch; make sure we see it.
    assert 7 in {c.distance for c in configs}


def test_discover_is_deterministic_and_unique(willow_dir: Path) -> None:
    a = discover_configs(willow_dir)
    b = discover_configs(willow_dir)
    assert a == b                      # stable ordering
    assert len(a) == len(set(a))       # no duplicates


def test_iter_configs_filters_distance_and_basis(willow_dir: Path) -> None:
    only = list(iter_configs(willow_dir, distances=(3,), bases=("Z",)))
    assert len(only) > 0
    assert all(c.distance == 3 and c.basis == "Z" for c in only)
    # Filtering is a strict subset of discovery.
    assert set(only) <= set(discover_configs(willow_dir))


def test_iter_configs_default_covers_all_distances(willow_dir: Path) -> None:
    everything = list(iter_configs(willow_dir))
    assert {c.distance for c in everything} == {3, 5, 7}
