"""Sample synthetic detection events for one rung of the noise-fidelity ladder.

Each rung is (ideal circuit, a DEM to sample from, the same DEM to decode with) --
matched-prior decoding, the idealized synthetic benchmark (see
``docs/superpowers/eval_synth_spec.md`` S3/S6). Three rungs:

``si1000``    the shipped SI1000 circuit-level DEM, used as-is.
``uniform``   the shipped SI1000 DEM with every ``error(p)`` instruction rewritten
              to a single uniform probability, keeping the detector/logical graph
              identical ("same error graph, flat weights" -- spec S3.2A).
``syndrome``  the shipped SI1000 DEM's graph, reweighted with probabilities
              estimated from real Willow detection-event statistics (no physical
              noise model, no ML -- arXiv:2606.11496; see
              ``docs/superpowers/syndrome_dem_rung_spec.md``).

Reuses the HF dataset fetch (circuit + shipped DEMs) so this needs no local Willow
tree; only the *source* of detectors/observables differs from ``load_config_from_hf``.
"""

from __future__ import annotations

import zlib

import numpy as np
import stim

from ising_sim2real.ingest.hf import (
    DEFAULT_HF_REPO,
    HFConfigData,
    _download,
    _load_dem_gz,
    _stem,
    load_config_from_hf,
)
from ising_sim2real.ingest.willow import WillowConfig

RUNGS = ("uniform", "si1000", "syndrome")


def _flatten_dem(dem: stim.DetectorErrorModel, p: float) -> stim.DetectorErrorModel:
    """Rewrite every error instruction's probability to ``p``, keeping the graph."""
    out = stim.DetectorErrorModel()
    for instr in dem.flattened():
        if instr.type == "error":
            out.append("error", p, instr.targets_copy())
        else:
            out.append(instr)
    return out


def build_rung_dem(
    rung: str,
    dem_si1000: stim.DetectorErrorModel,
    p: float,
    detection_events: np.ndarray | None = None,
) -> stim.DetectorErrorModel:
    """Build the DEM a rung samples from and decodes with (matched-prior)."""
    if rung == "si1000":
        return dem_si1000
    if rung == "uniform":
        return _flatten_dem(dem_si1000, p)
    if rung == "syndrome":
        if detection_events is None:
            raise ValueError("rung='syndrome' requires detection_events (real Willow data)")
        from ising_sim2real.ingest.syndrome_dem import estimate_dem_from_syndromes

        return estimate_dem_from_syndromes(dem_si1000, detection_events)
    raise ValueError(f"unknown rung {rung!r}; choose from {RUNGS}")


def _seed_for(cfg: WillowConfig, base_seed: int) -> int:
    # zlib.crc32, not the builtin hash(), so a re-run reproduces the same events
    # (str hashing is salted per-process unless PYTHONHASHSEED is fixed).
    return (base_seed + zlib.crc32(_stem(cfg).encode())) & 0x7FFFFFFF


def sample_config(
    cfg: WillowConfig,
    rung: str,
    p: float,
    shots: int,
    seed: int,
    repo: str = DEFAULT_HF_REPO,
) -> HFConfigData:
    """Fetch the ideal circuit + shipped SI1000 DEM for ``cfg``, then sample a rung.

    The ``syndrome`` rung additionally needs REAL detection events to estimate
    from -- fetched via ``load_config_from_hf`` (which also gives circuit+DEM in
    one call, avoiding a second round-trip). The returned ``detectors``/
    ``observables`` are always FRESHLY SAMPLED from the rung DEM, never the real
    shots themselves, for every rung including ``syndrome``.
    """
    if rung == "syndrome":
        real = load_config_from_hf(cfg, repo=repo)
        circuit = real.circuit
        rung_dem = build_rung_dem(rung, real.dem_si1000, p, detection_events=real.detectors)
    else:
        stem = _stem(cfg)
        circuit_path = _download(repo, f"circuits/{stem}.stim")
        if circuit_path is None:
            raise FileNotFoundError(f"circuits/{stem}.stim missing on {repo}")
        circuit = stim.Circuit.from_file(circuit_path)

        dem_si1000 = _load_dem_gz(_download(repo, f"dems/{stem}.si1000.dem.gz"))
        if dem_si1000 is None:
            raise FileNotFoundError(f"dems/{stem}.si1000.dem.gz missing on {repo}")

        rung_dem = build_rung_dem(rung, dem_si1000, p)

    sampler = rung_dem.compile_sampler(seed=_seed_for(cfg, seed))
    detectors, observables, _ = sampler.sample(shots=shots)

    return HFConfigData(
        circuit=circuit,
        detectors=detectors,
        observables=observables,
        dem_si1000=rung_dem,
        dem_rl=None,
    )
