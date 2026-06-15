"""Tests for loading a Willow configuration off disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import stim

from ising_sim2real.ingest.detectors import measurements_to_detectors
from ising_sim2real.ingest.willow import (
    config_dir,
    load_run,
    load_shipped_detection_data,
)


def test_config_dir_resolves_patch_basis_rounds(willow_dir: Path, d3_config) -> None:
    path = config_dir(willow_dir, d3_config)
    assert path == willow_dir / "d3_at_q4_5" / "Z" / "r01"
    assert (path / "circuit_ideal.stim").exists()


def test_load_run_shapes_and_metadata(willow_dir: Path, d3_config) -> None:
    run = load_run(willow_dir, d3_config)

    assert run.config == d3_config
    nm = run.circuit.num_measurements
    nsweep = run.circuit.num_sweep_bits
    shots = run.measurements.shape[0]

    assert run.measurements.shape == (shots, nm)
    assert run.sweep_bits.shape == (shots, nsweep)
    assert run.measurements.dtype == np.bool_
    # metadata.json reports 50000 shots for this config.
    assert shots == 50000


def test_load_run_dems_are_detector_error_models(willow_dir: Path, d3_config) -> None:
    run = load_run(willow_dir, d3_config)
    assert isinstance(run.dem_si1000, stim.DetectorErrorModel)
    assert isinstance(run.dem_rl, stim.DetectorErrorModel)
    # The shipped DEMs index the same detectors as the circuit.
    assert run.dem_si1000.num_detectors == run.circuit.num_detectors


def test_loaded_inputs_reproduce_shipped_detection_events(
    willow_dir: Path, d3_config
) -> None:
    """The raw inputs from load_run, run through m2d, equal the shipped events."""
    run = load_run(willow_dir, d3_config)
    derived = measurements_to_detectors(
        run.circuit, run.measurements, sweep_bits=run.sweep_bits
    )
    shipped = load_shipped_detection_data(willow_dir, d3_config)

    assert np.array_equal(derived.detectors, shipped.detectors)
    assert np.array_equal(derived.observables, shipped.observables)
