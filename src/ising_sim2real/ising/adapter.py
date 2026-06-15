"""Map Willow detection events into the Ising model's lattice layout.

The pre-decoder consumes a ``(B, 4, T, D, D)`` tensor: B shots, 4 channels
(``[x_syndrome, z_syndrome, x_present, z_present]``), T = number of QEC rounds,
on a D x D lattice. The single source of truth for that tensor is NVIDIA's
``dets_to_predecoder_inputs`` (vendored under ``third_party/Ising-Decoding``),
which the public models were trained against.

Why a reorder instead of a re-implementation
---------------------------------------------
``dets_to_predecoder_inputs`` expects detectors in the *synthetic* Stim circuit's
emission order, and it is **lossy** -- for a memory experiment it keeps only the
syndrome slices the model was trained on (the final data-basis layer and the last
bulk round of the opposite type are dropped). Re-implementing that by hand is
error prone, so instead we:

1. generate the reference synthetic circuit for ``(D, T, basis)`` -- this *is*
   NVIDIA's expected detector order;
2. reorder a circuit's detection events into that order via a canonical key
   ``(layer, stabilizer-type, x, y)`` shared by both circuits;
3. delegate to ``dets_to_predecoder_inputs``.

For any circuit the result is therefore **bit-identical** to NVIDIA's transform
(verified on synthetic circuits across distances, rounds, and bases). What the
canonical key does *not* certify is that Willow's XZZX device stabilizers map to
the same physical grid cells as the CSS synthetic code -- that cross-code
correspondence is the empirical step-4 validation (run the model, match the
classical MWPM baseline; CLAUDE.md). The reorder machinery is exact; the physical
Willow<->model alignment is the open research question this harness exists to test.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import stim
import torch

from ising_sim2real.paths import ISING_CODE


def _ensure_vendored_on_path() -> None:
    if not ISING_CODE.exists():
        raise FileNotFoundError(
            f"Vendored Ising-Decoding code not found at {ISING_CODE}; "
            "run scripts/setup_ising.py. The lattice convention lives there."
        )
    p = str(ISING_CODE)
    if p not in sys.path:
        sys.path.insert(0, p)


def _dets_to_predecoder_inputs():
    _ensure_vendored_on_path()
    from data.predecoder_transform import dets_to_predecoder_inputs  # type: ignore

    return dets_to_predecoder_inputs


def _canonical_rank(circuit: stim.Circuit) -> np.ndarray:
    """Stable rank of every detector under the key ``(layer, type, x, y)``.

    ``type`` is 0 for basis-matching stabilizers (those present in the first
    detector layer) and 1 for the opposite type (those that only appear in the
    bulk layers). The final data-basis layer keeps type 0. The key is unique per
    detector, so the rank is a bijection onto ``[0, num_detectors)``.
    """
    coords = circuit.get_detector_coordinates()
    n = circuit.num_detectors
    if n == 0 or not coords:
        raise ValueError("circuit has no annotated detectors")
    xy = np.array([[coords[i][0], coords[i][1]] for i in range(n)], dtype=float)
    layer = np.array([int(round(coords[i][2])) for i in range(n)], dtype=int)

    last = int(layer.max())
    first_positions = {tuple(p) for p in xy[layer == 0]}
    bulk = (layer != 0) & (layer != last)
    other_positions = {tuple(p) for p in xy[bulk]} - first_positions
    stab_type = np.array(
        [1 if tuple(p) in other_positions else 0 for p in xy], dtype=int
    )

    order = np.lexsort((xy[:, 1], xy[:, 0], stab_type, layer))
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(n)
    return rank


@lru_cache(maxsize=None)
def _reference_order(distance: int, rounds: int, basis: str, rotation: str) -> tuple:
    """NVIDIA's native detector order for a synthetic ``(D, T, basis)`` circuit.

    Returns ``ref_at_keypos``: the synthetic native index sitting at each
    canonical-key position. Cached because circuit generation is the slow part.
    """
    code = f"surface_code:rotated_memory_{basis.lower()}"
    circuit = stim.Circuit.generated(code, distance=distance, rounds=rounds)
    rank = _canonical_rank(circuit)
    ref_at_keypos = np.empty(rank.shape[0], dtype=int)
    ref_at_keypos[rank] = np.arange(rank.shape[0])
    return tuple(ref_at_keypos.tolist())


@dataclass(frozen=True)
class LatticeLayout:
    """Reorders a configuration's detection events into NVIDIA's model layout.

    Built once per ``(distance, basis, rounds)`` from the Stim circuit, then
    applied to any batch of detection events for that configuration.
    """

    distance: int
    basis: str
    rounds: int                  # T: number of QEC rounds the model sees
    num_detectors: int
    to_nvidia: np.ndarray        # native detector idx -> synthetic native idx
    rotation: str = "XV"

    @classmethod
    def from_circuit(
        cls,
        circuit: stim.Circuit,
        basis: str,
        distance: int | None = None,
        rotation: str = "XV",
    ) -> "LatticeLayout":
        basis = basis.strip().upper()
        if basis not in ("X", "Z"):
            raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")

        num_detectors = circuit.num_detectors
        coords = circuit.get_detector_coordinates()
        n_layers = int(max(int(round(coords[i][2])) for i in range(num_detectors))) + 1
        rounds = n_layers - 1  # the final data-basis layer is the +1
        D = distance if distance is not None else _infer_distance(num_detectors, rounds)
        if (D * D - 1) // 2 * 2 * rounds != num_detectors:
            raise ValueError(
                f"detector count {num_detectors} != 2*rounds*half for "
                f"distance={D}, rounds={rounds}"
            )

        rank = _canonical_rank(circuit)
        ref_at_keypos = np.array(
            _reference_order(D, rounds, basis, rotation), dtype=int
        )
        if ref_at_keypos.shape[0] != num_detectors:
            raise ValueError(
                "reference synthetic circuit has a different detector count "
                f"({ref_at_keypos.shape[0]}) than the target ({num_detectors}); "
                "the configuration is not a standard rotated memory experiment."
            )
        to_nvidia = ref_at_keypos[rank]

        return cls(
            distance=D,
            basis=basis,
            rounds=rounds,
            num_detectors=num_detectors,
            to_nvidia=to_nvidia,
            rotation=rotation,
        )

    def reorder(self, detectors: np.ndarray) -> np.ndarray:
        """Permute detection-event columns into NVIDIA's native detector order."""
        detectors = np.ascontiguousarray(detectors)
        if detectors.ndim != 2 or detectors.shape[1] != self.num_detectors:
            raise ValueError(
                f"detectors must be (shots, {self.num_detectors}); got {detectors.shape}"
            )
        out = np.zeros_like(detectors)
        out[:, self.to_nvidia] = detectors
        return out


def _infer_distance(num_detectors: int, rounds: int) -> int:
    half = num_detectors // (2 * rounds)
    D = int(round((2 * half + 1) ** 0.5))
    if (D * D - 1) // 2 != half:
        raise ValueError(
            f"cannot infer odd distance from {num_detectors} detectors over "
            f"{rounds} rounds; pass distance= explicitly."
        )
    return D


def detection_events_to_lattice(
    detectors: np.ndarray, layout: LatticeLayout
) -> torch.Tensor:
    """Reshape flat detection events into the ``(B, 4, T, D, D)`` model input.

    Delegates to NVIDIA's ``dets_to_predecoder_inputs`` after reordering the
    detectors into its expected emission order, so the output is bit-identical to
    the transform the public models were trained against.

    Args:
        detectors: shape (shots, layout.num_detectors), bool-like.
        layout: placement precomputed from the configuration's circuit.

    Returns:
        Float32 tensor of shape (shots, 4, layout.rounds, distance, distance).
    """
    reordered = layout.reorder(detectors)
    transform = _dets_to_predecoder_inputs()
    train_x, _x_syn, _z_syn = transform(
        torch.as_tensor(reordered, dtype=torch.int64),
        distance=layout.distance,
        n_rounds=layout.rounds,
        basis=layout.basis,
        code_rotation=layout.rotation,
    )
    return train_x.to(torch.float32)
