#!/usr/bin/env python3
"""
test_correctness.py
=====================
Cross-checks, for every circuit family in circuits.py:

  * this project's own mathematical-model reference (statevector_model.State)
    against Qiskit Aer's statevector simulator, by fidelity;
  * the ZX-calculus-optimized version of each circuit against the
    unoptimized original, by fidelity (proves zx_optimize() never changes
    the function the circuit computes, only its gate count);
  * real FPGA hardware (Alveo U55C) against the reference model, by
    fidelity and by the FPGA driver's own built-in validator, if the FPGA
    is available in this environment (skipped, not failed, otherwise).

Run directly: python tests/test_correctness.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import env  # noqa: E402  (sys.path + CUDA LD_LIBRARY_PATH fix — import before qiskit_aer/pyxrt)

import numpy as np  # noqa: E402
from qiskit import transpile  # noqa: E402
from qiskit_aer import AerSimulator  # noqa: E402

from circuits import qft_circuit, grover_circuit, clifford_t_circuit, lower_to_native  # noqa: E402
from statevector_model import State  # noqa: E402
from zx_optimizer import zx_optimize  # noqa: E402
import backends as bk  # noqa: E402

FID_TOL = 1e-6
_backend = AerSimulator(method='statevector')


def _aer_statevector(qc) -> np.ndarray:
    qc2 = qc.copy()
    qc2.save_statevector()
    tqc = transpile(qc2, _backend, optimization_level=0)
    res = _backend.run(tqc, shots=1).result()
    return np.asarray(res.get_statevector(tqc))


def test_reference_matches_aer():
    print("== reference model vs Qiskit Aer ==")
    cases = (
        [('QFT', n, *qft_circuit(n)) for n in (2, 3, 4, 5)] +
        [('Grover', n, *grover_circuit(n)) for n in (2, 3, 4)] +
        [('Clifford+T', n, *clifford_t_circuit(n, depth=8, seed=1)) for n in (2, 3, 4, 5)]
    )
    ok = True
    for name, n, gates, qc in cases:
        lowered = lower_to_native(gates)
        s = State(n).run(lowered)
        ref = _aer_statevector(qc)
        fid = abs(np.vdot(ref, s.state)) ** 2
        passed = fid > 1 - FID_TOL and abs(s.norm() - 1) < FID_TOL
        ok &= passed
        print(f"  {name:12s} n={n} fidelity={fid:.8f}  {'OK' if passed else 'FAIL'}")
    assert ok, "reference model diverged from Qiskit Aer"


def test_zx_optimization_preserves_function():
    print("== ZX-optimized circuit vs original ==")
    cases = [
        ('Clifford+T', *clifford_t_circuit(5, depth=15, seed=2)),
        ('QFT', *qft_circuit(5)),
    ]
    ok = True
    for name, gates, qc in cases:
        ref = _aer_statevector(qc)
        qc_opt, stats = zx_optimize(qc)
        got = _aer_statevector(qc_opt)
        fid = abs(np.vdot(ref, got)) ** 2
        passed = fid > 1 - FID_TOL
        ok &= passed
        print(f"  {name:12s} gates {stats.gate_count_before}->{stats.gate_count_after}  "
              f"fidelity={fid:.8f}  {'OK' if passed else 'FAIL'}")
    assert ok, "ZX-optimized circuit is not functionally equivalent to the original"


def test_fpga_matches_reference():
    print("== real FPGA hardware vs reference model ==")
    try:
        fpga = bk.init_fpga()
    except Exception as e:
        print(f"  [skip] FPGA unavailable: {e}")
        return
    ok = True
    for name, n, (gates, qc) in (
        ('QFT', 4, qft_circuit(4)),
        ('Grover', 3, grover_circuit(3)),
        ('Clifford+T', 4, clifford_t_circuit(4, depth=10, seed=3)),
    ):
        r = bk.run_fpga(fpga, gates, n, circuit_name=name)
        ref = bk.run_reference(gates, n)
        fid = bk.fidelity(r.statevector, ref.statevector)
        passed = bool(r.passed) and fid > 1 - FID_TOL
        ok &= passed
        print(f"  {name:12s} n={n} fidelity={fid:.8f} validator_passed={r.passed}  "
              f"{'OK' if passed else 'FAIL'}")
    assert ok, "FPGA output diverged from the reference model"


if __name__ == '__main__':
    test_reference_matches_aer()
    test_zx_optimization_preserves_function()
    test_fpga_matches_reference()
    print("\nAll correctness checks passed.")
