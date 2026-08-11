"""
circuits.py
============
Circuit generators used by the benchmark, each returned in two equivalent
forms:

  * a `gates` list — (name, qubits, params) tuples in the vocabulary
    `fpga_simulator_v07` already knows how to lower to the FPGA kernel's
    native gate set (see `_lower_to_kernel_basis` there). This is what gets
    handed to the FPGA backend, and to `statevector_model.State` for the
    reference/validation run.

  * a Qiskit `QuantumCircuit` — for the ZX-calculus optimizer (which needs
    a Qiskit circuit to hand to `pyzx`) and for the Aer CPU/GPU backends.

Three circuit families are provided, chosen so the ZX-calculus comparison
in the benchmark isn't cherry-picked to make ZX look good everywhere:

  * `qft_circuit`        — QFT. Dominated by arbitrary-angle controlled
                            phase rotations. ZX-calculus's rewrite rules
                            are strongest on Clifford+T structure, so QFT
                            is the honest "ZX may not help much" case.
  * `grover_circuit`      — Grover's algorithm (H / X / multi-controlled-Z
                            oracle / diffusion). Mixed Clifford + multi-
                            controlled structure.
  * `clifford_t_circuit`  — a random circuit built only from {H, S, CX, T},
                            the regime ZX-calculus (T-count / gate-count
                            reduction via spider fusion, pivoting, etc.) is
                            actually designed for. This is the case
                            expected to show a real reduction.
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Tuple

import env  # noqa: F401  (sets up sys.path for the sibling repo + XRT)
from fpga_simulator_v07 import _lower_to_kernel_basis, _validate_and_split_gates

from qiskit import QuantumCircuit

Gate = Tuple[str, List[int], Optional[List[float]]]


# =============================================================================
#  QFT
# =============================================================================
def qft_circuit(n: int) -> Tuple[List[Gate], QuantumCircuit]:
    gates: List[Gate] = []
    qc = QuantumCircuit(n)
    for i in range(n):
        gates.append(('h', [i], None))
        qc.h(i)
        for j in range(i + 1, n):
            theta = math.pi / (2 ** (j - i))
            gates.append(('cp', [j, i], [theta]))
            qc.cp(theta, j, i)
    for i in range(n // 2):
        gates.append(('swap', [i, n - 1 - i], None))
        qc.swap(i, n - 1 - i)
    return gates, qc


# =============================================================================
#  Grover — single-iteration search for the marked index `marked`
#  (matches the pattern already used elsewhere in the parent repo: X-sandwich
#  multi-controlled-Z oracle + standard diffusion operator).
# =============================================================================
def grover_circuit(n: int, marked: int = 0, iterations: Optional[int] = None) -> Tuple[List[Gate], QuantumCircuit]:
    if iterations is None:
        iterations = max(1, round(math.pi / 4 * math.sqrt(2 ** n)))
    gates: List[Gate] = []
    qc = QuantumCircuit(n)

    def h_all():
        for q in range(n):
            gates.append(('h', [q], None))
            qc.h(q)

    def x_pattern(idx: int):
        bits = format(idx, f'0{n}b')
        for q in range(n):
            if bits[n - 1 - q] == '0':
                gates.append(('x', [q], None))
                qc.x(q)

    def mcz_all_qubits():
        # H on last qubit + MCX(all others -> last) + H = multi-controlled Z
        gates.append(('h', [n - 1], None)); qc.h(n - 1)
        ctrls = list(range(n - 1))
        gates.append(('mcx', ctrls + [n - 1], None))
        qc.mcx(ctrls, n - 1)
        gates.append(('h', [n - 1], None)); qc.h(n - 1)

    h_all()
    for _ in range(iterations):
        # Oracle: phase-flip |marked>
        x_pattern(marked)
        mcz_all_qubits()
        x_pattern(marked)
        # Diffusion
        h_all()
        x_pattern(0)  # flip about |11...1> -> use all-zero pattern trick
        mcz_all_qubits()
        x_pattern(0)
        h_all()
    return gates, qc


# =============================================================================
#  Clifford+T random circuit — the regime ZX-calculus optimization targets.
# =============================================================================
def clifford_t_circuit(n: int, depth: int, seed: int = 0) -> Tuple[List[Gate], QuantumCircuit]:
    rng = random.Random(seed)
    gates: List[Gate] = []
    qc = QuantumCircuit(n)
    single = ['h', 's', 't']
    for _ in range(depth):
        for q in range(n):
            g = rng.choice(single)
            gates.append((g, [q], None))
            getattr(qc, g)(q)
        if n >= 2:
            a, b = rng.sample(range(n), 2)
            gates.append(('cx', [a, b], None))
            qc.cx(a, b)
    return gates, qc


# =============================================================================
#  Lower an (already-decomposed if needed) gate list to the FPGA kernel's
#  strictly-native gate set, exactly the way the FPGA driver itself would,
#  so the reference `State` model runs the byte-for-byte identical
#  instruction stream the FPGA executes.
# =============================================================================
def lower_to_native(gates: List[Gate]) -> List[Gate]:
    segments, resets, meas = _validate_and_split_gates(gates)
    assert not resets and not meas, "circuits.py generators never emit reset/measure"
    out: List[Gate] = []
    for seg in segments:
        out.extend(_lower_to_kernel_basis(seg))
    return out
