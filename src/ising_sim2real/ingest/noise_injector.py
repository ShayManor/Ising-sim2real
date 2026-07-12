"""Inject a 25-parameter circuit-level noise model into Willow's shipped
SI1000-noisy circuit template -- the ``fit`` rung of the noise-fidelity ladder.

Reuses the SAME noise-instruction *positions* Google's own SI1000 template
already picked (a stateful, tick-by-tick classification of each site's role:
state-prep / measurement / idle-in-gate / idle-in-SPAM / two-qubit-gate), only
replacing each site's probability/channel with the corresponding value from a
candidate ``NoiseModel`` (vendored from ``third_party/Ising-Decoding``). This
mirrors how the ``syndrome`` rung reuses the shipped SI1000 DEM's graph structure
unchanged, only reweighting probabilities.

The exact classification rule below was reverse-engineered by reading real
shipped ``circuit_noisy_si1000.stim`` files (both X- and Z-basis memory
experiments) -- see the implementation plan's Global Constraints for the two
load-bearing findings this rule depends on: Willow's circuits never use
``RX``/``MX``/``Z_ERROR`` (so ``p_prep_X``/``p_meas_X`` are fixed equal to their
Z counterparts), and idle-in-SPAM is applied to every qubit idling in a
SPAM-window tick, not only "data qubits" as ``NoiseModel``'s own docstring
describes.
"""

from __future__ import annotations

import stim


def _qubits_of(instr: stim.CircuitInstruction) -> list[int]:
    """Target qubit indices of a plain-qubit-target instruction (no sweep bits,
    no measurement-record targets -- every instruction this module touches is
    plain-qubit-only, per the shipped template's own structure)."""
    return [t.value for t in instr.targets_copy()]


def _split_into_ticks(circuit: stim.Circuit) -> list[list[stim.CircuitInstruction]]:
    """Split a REPEAT-free circuit's top-level instructions into TICK-delimited
    groups. Instructions before the first TICK form group 0; each TICK starts a
    new group. TICK instructions themselves are dropped (the caller re-inserts
    exactly one TICK between groups when reassembling in Task 3).

    Raises ``NotImplementedError`` on a ``stim.CircuitRepeatBlock`` -- Willow's
    shipped templates are fully unrolled (verified by enumerating every
    instruction name in two real files); a REPEAT block here means an
    unexpected circuit source this module has not been validated against.
    """
    ticks: list[list[stim.CircuitInstruction]] = [[]]
    for instr in circuit:
        if isinstance(instr, stim.CircuitRepeatBlock):
            raise NotImplementedError(
                "REPEAT blocks are not supported -- Willow's shipped "
                "circuit_noisy_si1000.stim files are fully unrolled; a REPEAT "
                "block here means an unexpected circuit source."
            )
        if instr.name == "TICK":
            ticks.append([])
            continue
        ticks[-1].append(instr)
    return ticks


def _is_spam_tick(tick: list[stim.CircuitInstruction]) -> bool:
    """A tick is a SPAM (state-prep-and-measurement) tick if it contains any
    ``R`` (reset) or ``M`` (measurement) instruction -- the boundary the
    shipped template itself already encodes via its elevated idle probabilities
    around those ticks (see Global Constraints)."""
    return any(instr.name in ("R", "M") for instr in tick)


def _rewrite_tick(
    tick: list[stim.CircuitInstruction], noise
) -> list[stim.CircuitInstruction]:
    """Rewrite one tick's noise-instruction sites per the module docstring's rule.

    ``noise`` is a ``qec.noise_model.NoiseModel`` instance (vendored, imported by
    the caller -- this function only calls its public accessor methods, never
    constructs one, so it has no import-path dependency of its own).
    """
    out: list[stim.CircuitInstruction] = []
    spam_tick = _is_spam_tick(tick)
    idle_qubits: set[int] = set()

    i = 0
    n = len(tick)
    while i < n:
        instr = tick[i]

        if instr.name == "R":
            out.append(instr)
            nxt = tick[i + 1] if i + 1 < n else None
            if nxt is not None and nxt.name == "X_ERROR" and set(_qubits_of(nxt)) == set(_qubits_of(instr)):
                out.append(stim.CircuitInstruction("X_ERROR", nxt.targets_copy(), [noise.p_prep_Z]))
                i += 2
                continue
            i += 1
            continue

        if instr.name == "RX":
            raise NotImplementedError(
                "RX (X-basis reset) found -- Willow's shipped circuits never use "
                "this gate (verified: only R + H-conjugation). p_prep_X is not "
                "wired to any injection site; refusing to silently drop this noise."
            )

        if instr.name == "CZ":
            out.append(instr)
            nxt = tick[i + 1] if i + 1 < n else None
            if nxt is not None and nxt.name == "DEPOLARIZE2" and set(_qubits_of(nxt)) == set(_qubits_of(instr)):
                out.append(stim.CircuitInstruction(
                    "PAULI_CHANNEL_2", nxt.targets_copy(), list(noise.to_stim_pauli_channel_2_args())
                ))
                i += 2
                continue
            i += 1
            continue

        if instr.name == "M":
            out.append(stim.CircuitInstruction("M", instr.targets_copy(), [noise.p_meas_Z]))
            i += 1
            continue

        if instr.name == "MX":
            raise NotImplementedError(
                "MX (X-basis measurement) found -- Willow's shipped circuits "
                "never use this gate (verified: only M + H-conjugation). "
                "p_meas_X is not wired to any injection site; refusing to "
                "silently drop this noise."
            )

        if instr.name == "DEPOLARIZE1":
            idle_qubits.update(_qubits_of(instr))
            i += 1
            continue

        # Everything else (H, X, Y, CX-with-sweep-bits, DETECTOR,
        # OBSERVABLE_INCLUDE, ...) passes through unchanged.
        out.append(instr)
        i += 1

    if idle_qubits:
        probs = (
            noise.to_stim_pauli_channel_1_args_spam()
            if spam_tick
            else noise.to_stim_pauli_channel_1_args_cnot()
        )
        targets = [stim.GateTarget(q) for q in sorted(idle_qubits)]
        out.append(stim.CircuitInstruction("PAULI_CHANNEL_1", targets, list(probs)))

    return out


def inject_noise_model(circuit_noisy_template: stim.Circuit, noise) -> stim.Circuit:
    """Rewrite ``circuit_noisy_template`` (a shipped ``circuit_noisy_si1000.stim``,
    parsed) with ``noise``'s 25 parameters, keeping SI1000's own noise-instruction
    positions unchanged -- see module docstring for the exact per-site rule.
    """
    ticks = _split_into_ticks(circuit_noisy_template)
    out = stim.Circuit()
    for idx, tick in enumerate(ticks):
        for instr in _rewrite_tick(tick, noise):
            out.append(instr)
        if idx < len(ticks) - 1:
            out.append(stim.CircuitInstruction("TICK", [], []))
    return out
