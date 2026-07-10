"""Sample synthetic detection events for one rung of the noise-fidelity ladder.

Each rung is (ideal circuit, a DEM to sample from, the same DEM to decode with) --
matched-prior decoding, the idealized synthetic benchmark (see
``docs/superpowers/eval_synth_spec.md`` S3/S6). Four rungs:

``si1000``    the shipped SI1000 circuit-level DEM, used as-is.
``uniform``   the shipped SI1000 DEM with every ``error(p)`` instruction rewritten
              to a single uniform probability, keeping the detector/logical graph
              identical ("same error graph, flat weights" -- spec S3.2A).
``syndrome``  the shipped SI1000 DEM's graph, reweighted with probabilities
              estimated from real Willow detection-event statistics (no physical
              noise model, no ML -- arXiv:2606.11496; see
              ``docs/superpowers/syndrome_dem_rung_spec.md``).
``fit``       a 25-parameter circuit-level noise model, fit per-patch to real
              Willow detection-event statistics (state-prep, measurement,
              idle-in-gate, idle-in-SPAM, two-qubit-gate channels), injected into
              the shipped SI1000-noisy circuit template at the SAME instruction
              positions SI1000 already uses -- see
              ``docs/superpowers/specs/2026-07-09-25-param-fit-rung-design.md``.

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

RUNGS = ("uniform", "si1000", "syndrome", "fit")


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
    patch: str | None = None,
    circuit_noisy_template: stim.Circuit | None = None,
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
    if rung == "fit":
        if patch is None or circuit_noisy_template is None:
            raise ValueError("rung='fit' requires patch and circuit_noisy_template")
        from ising_sim2real.ingest.noise_injector import inject_noise_model

        noise = _load_fitted_noise_model(patch)
        noisy_circuit = inject_noise_model(circuit_noisy_template, noise)
        # approximate_disjoint_errors=True is REQUIRED once PAULI_CHANNEL_2
        # sites exist (confirmed directly against stim during Task 3 -- every
        # fitted model has nonzero 2Q-gate params, so this always applies here).
        return noisy_circuit.detector_error_model(approximate_disjoint_errors=True)
    raise ValueError(f"unknown rung {rung!r}; choose from {RUNGS}")


def _seed_for(cfg: WillowConfig, base_seed: int) -> int:
    # zlib.crc32, not the builtin hash(), so a re-run reproduces the same events
    # (str hashing is salted per-process unless PYTHONHASHSEED is fixed).
    return (base_seed + zlib.crc32(_stem(cfg).encode())) & 0x7FFFFFFF


def _load_fitted_noise_model(patch: str):
    import json
    import sys

    from ising_sim2real.paths import ISING_CODE, REPO_ROOT

    path = REPO_ROOT / "results" / "fitted_noise_models" / f"{patch}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python scripts/fit_noise_models.py --patch "
            f"{patch}` first."
        )
    p = str(ISING_CODE)
    if p not in sys.path:
        sys.path.insert(0, p)
    from qec.noise_model import NoiseModel  # type: ignore

    return NoiseModel(**json.loads(path.read_text()))


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
    from; the ``fit`` rung additionally needs the SI1000-noisy circuit TEMPLATE
    (to inject its patch's fitted model into) -- both fetched via
    ``load_config_from_hf``/``fetch_si1000_noisy_circuit``. The returned
    ``detectors``/``observables`` are always FRESHLY SAMPLED from the rung DEM,
    never the real shots themselves, for every rung.
    """
    if rung == "syndrome":
        real = load_config_from_hf(cfg, repo=repo)
        circuit = real.circuit
        rung_dem = build_rung_dem(rung, real.dem_si1000, p, detection_events=real.detectors)
    elif rung == "fit":
        from ising_sim2real.ingest.hf import fetch_si1000_noisy_circuit
        from ising_sim2real.ingest.willow import patch_key

        real = load_config_from_hf(cfg, repo=repo)
        circuit = real.circuit
        template = fetch_si1000_noisy_circuit(cfg, repo=repo)
        rung_dem = build_rung_dem(
            rung, real.dem_si1000, p,
            patch=patch_key(cfg.distance, cfg.orientation),
            circuit_noisy_template=template,
        )
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
