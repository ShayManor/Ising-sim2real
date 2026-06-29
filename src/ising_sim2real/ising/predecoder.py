"""NVIDIA Ising pre-decoder + PyMatching, behind the panel's decoder contract.

Wires the full RQ4 path for one configuration:

    detection events
      -> reorder into NVIDIA's native order (ising.adapter.LatticeLayout)
      -> PreDecoderMemoryEvalModule  (vendored; model forward -> logits ->
         threshold -> induced syndrome -> residual + logical frame pre_L)
      -> PyMatching on the residual, built from a synthetic circuit's DEM
      -> final flip = pre_L XOR match(residual)

The residual lives in the *synthetic* CSS detector layout, so its matcher is built
from ``stim.Circuit.generated(... rotated_memory ...)`` with circuit-level noise (a
noiseless circuit yields an empty DEM with no boundary edges, which PyMatching
cannot match). The residual-assembly and frame logic are reused verbatim from the
vendored ``evaluation.logical_error_rate`` so they stay bit-identical to the path
the public models were trained/evaluated against.

Caveat (CLAUDE.md guardrail): the public models were trained for a receptive field
R far larger than d=3/5/7, and the Willow XZZX device -> CSS model lattice
correspondence is the open step-4 question. Numbers from this decoder are *not*
validated against the MWPM baseline; that comparison is the experiment.
"""

from __future__ import annotations

import sys
import time
from functools import lru_cache

import numpy as np
import pymatching
import stim
import torch
from omegaconf import OmegaConf

from ising_sim2real.decoders.base import DecodeResult
from ising_sim2real.ising.adapter import LatticeLayout
from ising_sim2real.paths import ISING_CODE

# Default uniform circuit-level depolarizing strength used to give the synthetic
# residual matcher a well-formed matching graph (boundaries + edge structure).
# The weights barely matter for the graph topology; this is the "uniform" rung of
# the fidelity ladder and is overridable.
DEFAULT_SYN_NOISE = 1e-3


def _ensure_vendored_on_path() -> None:
    if not ISING_CODE.exists():
        raise FileNotFoundError(
            f"Vendored Ising-Decoding code not found at {ISING_CODE}; "
            "run scripts/setup_ising.py."
        )
    p = str(ISING_CODE)
    if p not in sys.path:
        sys.path.insert(0, p)


def _eval_module_factory():
    _ensure_vendored_on_path()
    from evaluation.logical_error_rate import (  # type: ignore
        PreDecoderMemoryEvalModule,
        _build_stab_maps,
    )

    return PreDecoderMemoryEvalModule, _build_stab_maps


@lru_cache(maxsize=None)
def memory_circuit(distance: int, rounds: int, basis: str, rotation: str, syn_noise: float):
    """NVIDIA's ``MemoryCircuit`` -- the circuit the model's residual is laid out for.

    The pre-decoder's residual detectors are in this circuit's native emission
    order (``2 * T * half`` columns, ``[X-group, Z-group]`` per round, with
    ``add_boundary_detectors=True``), NOT ``stim.Circuit.generated``'s order. The
    residual matcher and any synthetic eval data MUST come from here or the
    residual is matched on the wrong graph (verified: native decode beats MWPM at
    d3/d5/d7; the generated-circuit graph lands at chance). Mirrors the reference
    ``run_inference_and_decode_pre_decoder_memory`` (vendored).
    """
    _ensure_vendored_on_path()
    from qec.surface_code.memory_circuit import MemoryCircuit  # type: ignore

    mc = MemoryCircuit(
        distance=distance,
        idle_error=syn_noise,
        sqgate_error=syn_noise,
        tqgate_error=syn_noise,
        spam_error=(2.0 / 3.0) * syn_noise,
        n_rounds=rounds,
        basis=basis,
        code_rotation=rotation,
        add_boundary_detectors=True,
    )
    mc.set_error_rates()
    return mc


@lru_cache(maxsize=None)
def _residual_matcher(distance: int, rounds: int, basis: str, rotation: str, syn_noise: float) -> pymatching.Matching:
    """PyMatching for the residual, from NVIDIA's ``MemoryCircuit`` DEM.

    Cached per ``(distance, rounds, basis, rotation, syn_noise)`` because the same
    layout is shared by every patch orientation at that ``(d, basis, T)``.
    """
    mc = memory_circuit(distance, rounds, basis, rotation, syn_noise)
    dem = mc.stim_circuit.detector_error_model(
        decompose_errors=True, approximate_disjoint_errors=True
    )
    return pymatching.Matching.from_detector_error_model(dem)


class IsingPreDecoder:
    """Ising pre-decoder followed by MWPM on the cleaned residual.

    Built once per configuration from its circuit; ``model`` is shared across
    configurations (the net is fully convolutional, so T and D are free).
    """

    name = "ising+mwpm"

    #: Minimum rounds the vendored residual transform supports (needs T >= 2).
    MIN_ROUNDS = 2

    def __init__(
        self,
        model: torch.nn.Module,
        circuit: stim.Circuit | None,
        basis: str,
        distance: int,
        rounds: int,
        device: torch.device,
        rotation: str = "XV",
        syn_noise: float = DEFAULT_SYN_NOISE,
    ) -> None:
        if rounds < self.MIN_ROUNDS:
            raise ValueError(
                f"Ising residual transform needs rounds >= {self.MIN_ROUNDS}, got {rounds}"
            )
        self._device = device
        # ``circuit is None`` => native mode: detectors are already in the model's
        # MemoryCircuit emission order (synthetic data sampled from MemoryCircuit),
        # so no reorder is applied. Otherwise reorder the data circuit's detectors.
        self._layout = (
            None
            if circuit is None
            else LatticeLayout.from_circuit(circuit, basis=basis, distance=distance, rotation=rotation)
        )

        eval_module_cls, build_stab_maps = _eval_module_factory()
        cfg = OmegaConf.create(
            {
                "distance": distance,
                "enable_fp16": False,
                "data": {"code_rotation": rotation},
                "test": {
                    "meas_basis_test": basis,
                    "th_data": 0.0,
                    "th_syn": 0.0,
                    "sampling_mode": "threshold",
                    "temperature": 1.0,
                    "n_rounds": rounds,
                },
            }
        )
        self._module = eval_module_cls(model, cfg, build_stab_maps(distance, rotation), device).to(device).eval()
        self._matcher = _residual_matcher(distance, rounds, basis, rotation, float(syn_noise))
        # Cap shots per forward so the (B, 4, T, D, D) tensor + conv activations stay
        # bounded: a single 50k-shot pass at d7/r250 is a ~10 GB input alone and OOMs
        # the GPU. Keep B*T*D*D under a fixed cell budget; clamp to [256, 16384].
        cells = rounds * distance * distance
        self._chunk = int(np.clip(self.CELL_BUDGET // max(cells, 1), 256, 16384))

    #: Target (shots * rounds * D^2) per forward pass.
    CELL_BUDGET = 30_000_000

    def decode_batch(self, detectors: np.ndarray) -> DecodeResult:
        reordered = (
            np.ascontiguousarray(detectors)
            if self._layout is None
            else self._layout.reorder(detectors)
        ).astype(np.uint8)
        start = time.perf_counter()
        pre_parts, res_parts = [], []
        with torch.no_grad():
            for lo in range(0, reordered.shape[0], self._chunk):
                chunk = reordered[lo:lo + self._chunk]
                out = self._module(torch.as_tensor(chunk, dtype=torch.uint8, device=self._device))
                pre_parts.append(out[:, 0].to(torch.int32).cpu().numpy())
                res_parts.append(out[:, 1:].to(torch.int32).cpu().numpy().astype(np.uint8))
        pre_L = np.concatenate(pre_parts)
        residual = np.ascontiguousarray(np.concatenate(res_parts), dtype=np.uint8)
        matched = np.asarray(self._matcher.decode_batch(residual), dtype=np.uint8).reshape(-1)
        seconds = time.perf_counter() - start
        final = ((pre_L + matched) % 2).astype(bool).reshape(-1, 1)
        return DecodeResult(predictions=final, seconds=seconds)
