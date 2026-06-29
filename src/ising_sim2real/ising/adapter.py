"""Map Willow detection events into the Ising model's lattice layout.

The pre-decoder consumes a ``(B, 4, T, D, D)`` tensor: B shots, 4 channels
(``[x_syndrome, z_syndrome, x_present, z_present]``), T = number of QEC rounds,
on a D x D lattice. The single source of truth for that tensor is NVIDIA's
``dets_to_predecoder_inputs`` (vendored under ``third_party/Ising-Decoding``),
which the public models were trained against. Its input contract: detectors in
``MemoryCircuit`` emission order -- a flat ``(B, 2*T*half)`` array viewed as
``(B, 2*T, half)`` timeline groups, where group ``idx_x[t]`` / ``idx_z[t]`` holds
the X- / Z-stabilizer syndromes of round ``t`` (boundary rounds masked).

The Willow device circuit does **not** emit detectors in that order: its data
qubits live on a 45-degrees-rotated diagonal lattice, its stabilizers are laid
out under a transpose of NVIDIA's CSS grid, and its detector layers index time
differently. This adapter establishes the physical correspondence rigorously and
reorders Willow detection events into the model's contract layout:

1. **Spatial.** Willow data qubits map to a ``D x D`` grid via ``(u, v) =
   (x+y, x-y)``. The unique square symmetry (here a transpose) that carries every
   Willow X/Z stabilizer's data-qubit support onto NVIDIA's ``Hx``/``Hz`` rows is
   found by search -- this both *finds* and *proves* the alignment (a wrong guess
   has no consistent support match). It yields, per Willow ancilla, its NVIDIA
   stabilizer-row index (= the within-group column the transform expects).
2. **Temporal.** Each Willow detector at layer ``L`` and stabilizer-row ``r`` of
   type X/Z lands at flat index ``idx_{x,z}[L] * half + r``; final data-basis and
   boundary-masked detectors map to the transform's sentinel and are dropped.

The result decodes Willow at every distance and basis (Ising LER well below
chance, validated against the MWPM baseline). The residual is then matched in the
same ``MemoryCircuit`` DEM layout (see ``predecoder._residual_matcher``).
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


@lru_cache(maxsize=None)
def _nvidia_supports(distance: int, rotation: str) -> tuple:
    """NVIDIA CSS stabilizer supports: ``(Xrows, Zrows)``.

    Each is a tuple of ``frozenset`` data-grid cells ``(row, col)`` for stabilizer
    row ``r`` (``r`` is the within-group column the predecoder transform expects).
    """
    _ensure_vendored_on_path()
    from qec.surface_code.memory_circuit import SurfaceCode  # type: ignore

    sc = SurfaceCode(distance, first_bulk_syndrome_type=rotation[0], rotated_type=rotation[1])
    D = distance

    def rows(H):
        H = np.asarray(H).astype(int)
        return tuple(
            frozenset((c // D, c % D) for c in np.nonzero(H[r])[0]) for r in range(H.shape[0])
        )

    return rows(sc.hx), rows(sc.hz)


# Square symmetries (D x D grid) -- one of these carries Willow's stabilizer
# layout onto NVIDIA's; the matching one is found by support comparison.
_DIHEDRAL = (
    ("id", lambda r, c, D: (r, c)),
    ("rot90", lambda r, c, D: (c, D - 1 - r)),
    ("rot180", lambda r, c, D: (D - 1 - r, D - 1 - c)),
    ("rot270", lambda r, c, D: (D - 1 - c, r)),
    ("flipH", lambda r, c, D: (r, D - 1 - c)),
    ("flipV", lambda r, c, D: (D - 1 - r, c)),
    ("transpose", lambda r, c, D: (c, r)),
    ("antitranspose", lambda r, c, D: (D - 1 - c, D - 1 - r)),
)


def _willow_geometry(circuit: stim.Circuit, distance: int):
    """Data-qubit grid, ancilla supports/types from a Willow device circuit.

    Returns ``(coord2anc, asupp, basis_anc)``:
      * ``coord2anc``: ``(x, y) -> ancilla qubit`` for stabilizer placement.
      * ``asupp``: ``ancilla -> frozenset`` of data-grid cells it checks.
      * ``basis_anc``: set of ancillas whose type matches the memory basis (they
        carry the round-0 detectors).
    """
    D = distance
    qc = circuit.get_final_qubit_coordinates()
    meas = [
        t.qubit_value
        for inst in circuit.flattened()
        if inst.name in ("M", "MZ", "MX", "MR", "MRZ", "MRX")
        for t in inst.targets_copy()
        if t.is_qubit_target
    ]
    data = meas[-D * D :]  # data qubits are the final D^2 measured (end readout)
    dcoord = {q: tuple(qc[q]) for q in data}

    # D x D grid via the 45-degree rotation u = x+y, v = x-y.
    umin = min(x + y for x, y in dcoord.values())
    vmin = min(x - y for x, y in dcoord.values())
    dgrid = {q: (int((x + y - umin) // 2), int((x - y - vmin) // 2)) for q, (x, y) in dcoord.items()}
    if len(set(dgrid.values())) != D * D:
        raise ValueError(f"Willow data qubits do not form a {D}x{D} grid: {sorted(dgrid.values())}")

    anc = sorted(set(meas) - set(data))
    acoord = {q: tuple(qc[q]) for q in anc}
    coord2data = {v: k for k, v in dcoord.items()}

    def support(ax, ay):
        s = set()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # axial data neighbours
            nb = coord2data.get((ax + dx, ay + dy))
            if nb is not None:
                s.add(dgrid[nb])
        return frozenset(s)

    asupp = {q: support(*acoord[q]) for q in anc}
    detc = circuit.get_detector_coordinates()
    r0 = {tuple(detc[d][:2]) for d in range(circuit.num_detectors) if int(round(detc[d][2])) == 0}
    basis_anc = {q for q in anc if acoord[q] in r0}
    coord2anc = {acoord[q]: q for q in anc}
    return coord2anc, asupp, basis_anc


def _stab_row_maps(circuit: stim.Circuit, distance: int, basis: str, rotation: str):
    """Per-ancilla NVIDIA stabilizer-row index, via the unique square symmetry.

    Returns ``(xmap, zmap)``: ``ancilla -> row`` for X- and Z-type stabilizers.
    Raises if no symmetry reproduces NVIDIA's support sets (would mean the device
    layout is not the expected rotated surface code).
    """
    D = distance
    coord2anc, asupp, basis_anc = _willow_geometry(circuit, distance)
    Xrows, Zrows = _nvidia_supports(distance, rotation)
    nvX = {s: r for r, s in enumerate(Xrows)}
    nvZ = {s: r for r, s in enumerate(Zrows)}
    anc_all = set(asupp)
    if basis.upper() == "Z":
        zanc, xanc = basis_anc, anc_all - basis_anc
    else:
        xanc, zanc = basis_anc, anc_all - basis_anc

    for _name, g in _DIHEDRAL:
        def gs(s):
            return frozenset(g(r, c, D) for r, c in s)

        try:
            xmap = {q: nvX[gs(asupp[q])] for q in xanc}
            zmap = {q: nvZ[gs(asupp[q])] for q in zanc}
        except KeyError:
            continue
        if len(set(xmap.values())) == len(Xrows) and len(set(zmap.values())) == len(Zrows):
            return xmap, zmap
    raise ValueError(
        f"no square symmetry aligns Willow stabilizers to NVIDIA's CSS layout "
        f"(distance={D}, basis={basis}, rotation={rotation}); not a standard rotated code"
    )


def _timeline_idx(T: int, basis: str):
    """Transform timeline-group index per round, ``(idx_x, idx_z, sentinel)``.

    Mirrors ``predecoder_transform._predecoder_transform_core`` exactly: the
    X-/Z-syndrome of round ``t`` is read from flat timeline group ``idx_x[t]`` /
    ``idx_z[t]``; ``sentinel == 2*T`` marks a boundary round the transform zeroes.
    """
    sent = 2 * T
    xb = list(range(1, 2 * T - 1, 2))
    zb = list(range(2, 2 * T, 2))
    if basis.upper() == "X":
        idx_x = [0] + xb
        idx_z = [sent] + zb[:-1] + [sent]
    else:
        idx_x = [sent] + xb[:-1] + [sent]
        idx_z = [0] + zb
    return idx_x, idx_z, sent


@dataclass(frozen=True)
class LatticeLayout:
    """Reorders a Willow configuration's detection events into the model layout.

    Built once per ``(distance, basis, rounds)`` from the device Stim circuit,
    then applied to any batch of detection events for that configuration. The
    reorder is a *scatter*: some device detectors (final data-basis layer,
    boundary-masked rounds) are dropped, and the contract array's masked slots
    stay zero -- the transform ignores them.
    """

    distance: int
    basis: str
    rounds: int                  # T: number of QEC rounds the model sees
    num_detectors: int           # device detector count (reorder input width)
    src: np.ndarray              # device detector indices that are kept
    dst: np.ndarray              # contract flat index each kept detector maps to
    rotation: str = "XV"

    @property
    def contract_width(self) -> int:
        return 2 * self.rounds * ((self.distance * self.distance - 1) // 2)

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
        half = (D * D - 1) // 2
        if half * 2 * rounds != num_detectors:
            raise ValueError(
                f"detector count {num_detectors} != 2*rounds*half for distance={D}, rounds={rounds}"
            )

        xmap, zmap = _stab_row_maps(circuit, D, basis, rotation)
        idx_x, idx_z, sentinel = _timeline_idx(rounds, basis)
        coord2anc, _asupp, _basis_anc = _willow_geometry(circuit, D)

        src, dst = [], []
        for i in range(num_detectors):
            x, y, layer = coords[i][0], coords[i][1], int(round(coords[i][2]))
            anc = coord2anc.get((x, y))
            if anc is None or layer >= rounds:
                continue  # final data-basis layer / out of contract range -> dropped
            is_x = anc in xmap
            group = (idx_x if is_x else idx_z)[layer]
            if group == sentinel:
                continue  # boundary-masked round -> dropped
            row = xmap[anc] if is_x else zmap[anc]
            src.append(i)
            dst.append(group * half + row)

        return cls(
            distance=D,
            basis=basis,
            rounds=rounds,
            num_detectors=num_detectors,
            src=np.asarray(src, dtype=np.intp),
            dst=np.asarray(dst, dtype=np.intp),
            rotation=rotation,
        )

    def reorder(self, detectors: np.ndarray) -> np.ndarray:
        """Scatter device detection events into the model's contract layout."""
        detectors = np.ascontiguousarray(detectors)
        if detectors.ndim != 2 or detectors.shape[1] != self.num_detectors:
            raise ValueError(
                f"detectors must be (shots, {self.num_detectors}); got {detectors.shape}"
            )
        out = np.zeros((detectors.shape[0], self.contract_width), dtype=detectors.dtype)
        out[:, self.dst] = detectors[:, self.src]
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

    Reorders the device detectors into NVIDIA's contract layout, then delegates to
    ``dets_to_predecoder_inputs``.

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
