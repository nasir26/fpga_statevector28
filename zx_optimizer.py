"""
zx_optimizer.py
=================
ZX-calculus circuit optimization stage (https://zxcalculus.com), using the
`pyzx` library, inserted as an optional pre-processing pass before a
circuit is handed to any of the three backends.

Pipeline: Qiskit QuantumCircuit
            -> OpenQASM 2.0 text
            -> pyzx.Circuit  (parses gates into a ZX-diagram-ready form)
            -> pyzx Graph
            -> zx.full_reduce()   (the actual ZX-calculus rewriting: spider
               fusion, bialgebra/Hopf rules, local complementation and
               pivoting — the graph-rewrite rules at zxcalculus.com applied
               to reduce the diagram)
            -> zx.extract_circuit()  (turn the reduced diagram back into a
               sequence of gates — nontrivial in general; this is the
               well-known "circuit extraction" problem ZX-calculus tooling
               solves via the pivoting/gaussian-elimination procedure
               `extract_circuit` implements)
            -> .to_basic_gates()   (expand any composite gates extraction
               produced into circuit primitives)
            -> OpenQASM 2.0 text -> Qiskit QuantumCircuit

then finally lowered to the FPGA kernel's native gate set the same way
every other circuit in this project is (`circuits.lower_to_native`), so
the optimizer's output is compared apples-to-apples with the unoptimized
circuit on identical downstream hardware.

Honesty note (see circuits.py docstring): ZX-calculus rewriting is
strongest on Clifford+T circuits. It is not guaranteed to reduce — and can
increase — gate count on circuits dominated by arbitrary-angle rotations
(e.g. QFT's controlled-phase ladder), because such gates don't participate
in the same fusion/elimination rules as Clifford spiders. This module
reports the true before/after counts either way; `benchmark_28q.py` prints
and plots both cases rather than only the favorable one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import env  # noqa: F401

try:
    import pyzx as zx
    PYZX_AVAILABLE = True
except ImportError:
    PYZX_AVAILABLE = False

from qiskit import QuantumCircuit, qasm2


@dataclass
class ZXReductionStats:
    gate_count_before: int
    gate_count_after: int
    two_qubit_before: int
    two_qubit_after: int
    t_count_before: int
    t_count_after: int
    reduced: bool  # True iff gate_count_after < gate_count_before

    @property
    def gate_reduction_pct(self) -> float:
        if self.gate_count_before == 0:
            return 0.0
        return 100.0 * (self.gate_count_before - self.gate_count_after) / self.gate_count_before


_TWO_QUBIT_NAMES = {'cx', 'cnot', 'cz', 'cp', 'swap'}


def _count(qc: QuantumCircuit) -> Dict[str, int]:
    counts = {'total': 0, 'two_qubit': 0, 't': 0}
    for instr in qc.data:
        name = instr.operation.name.lower()
        counts['total'] += 1
        if name in _TWO_QUBIT_NAMES:
            counts['two_qubit'] += 1
        if name in ('t', 'tdg'):
            counts['t'] += 1
    return counts


def zx_optimize(qc: QuantumCircuit) -> "tuple[QuantumCircuit, ZXReductionStats]":
    """Run the circuit through ZX-calculus reduction. Returns the
    (possibly unchanged) optimized circuit plus before/after stats.
    Falls back to returning the input circuit unchanged if pyzx is not
    installed or extraction fails on this circuit (extraction is not
    guaranteed to succeed on every diagram; pyzx raises in that case)."""
    before = _count(qc)
    if not PYZX_AVAILABLE:
        return qc, ZXReductionStats(before['total'], before['total'],
                                     before['two_qubit'], before['two_qubit'],
                                     before['t'], before['t'], False)
    try:
        qasm_in = qasm2.dumps(qc)
        zxc = zx.Circuit.from_qasm(qasm_in)
        g = zxc.to_graph()
        zx.full_reduce(g)
        g.normalize()
        zxc_out = zx.extract_circuit(g.copy()).to_basic_gates()
        qasm_out = zxc_out.to_qasm()
        qc_out = qasm2.loads(qasm_out)
    except Exception:
        # Extraction/reduction failed on this diagram — degrade to the
        # original circuit rather than hard-failing the whole benchmark.
        return qc, ZXReductionStats(before['total'], before['total'],
                                     before['two_qubit'], before['two_qubit'],
                                     before['t'], before['t'], False)

    after = _count(qc_out)
    stats = ZXReductionStats(
        gate_count_before=before['total'], gate_count_after=after['total'],
        two_qubit_before=before['two_qubit'], two_qubit_after=after['two_qubit'],
        t_count_before=before['t'], t_count_after=after['t'],
        reduced=after['total'] < before['total'],
    )
    return qc_out, stats


if __name__ == '__main__':
    from circuits import clifford_t_circuit, qft_circuit
    for label, (gates, qc) in (
        ('Clifford+T(6,depth=20)', clifford_t_circuit(6, depth=20, seed=0)),
        ('QFT(6)', qft_circuit(6)),
    ):
        _, stats = zx_optimize(qc)
        print(f"{label:24s} gates {stats.gate_count_before:4d} -> {stats.gate_count_after:4d} "
              f"({stats.gate_reduction_pct:+.1f}%)  T-count {stats.t_count_before} -> {stats.t_count_after}")
