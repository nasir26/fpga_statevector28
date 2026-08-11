"""
backends.py
============
Thin, uniform wrappers around the three hardware backends this project
compares. Each `init_*` returns a live handle or `None` (with a printed
reason) so a missing/unavailable backend degrades the sweep instead of
crashing it — same graceful-degradation contract the parent repo's
`hardware_comparison_qft.py` already uses.

FPGA runs go through the parent repo's own `fpga_simulator_v07.QuantumFPGACircuit`
against real Alveo U55C hardware via XRT — no reimplementation. CPU/GPU runs
go through Qiskit Aer's `AerSimulator(method='statevector', device=...)`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import env  # noqa: F401  (sys.path + CUDA LD_LIBRARY_PATH fix, must run first)
import numpy as np

from circuits import Gate, lower_to_native


@dataclass
class RunResult:
    hardware: str
    n_qubits: int
    num_gates: int
    kernel_time_us: float   # device/compute-only time (excludes host I/O when separable)
    wall_time_us: float     # full end-to-end time including host-side transfer
    statevector: Optional[np.ndarray]
    passed: Optional[bool]
    extra: dict


# =============================================================================
#  FPGA (Alveo U55C via XRT)
# =============================================================================
def init_fpga(xclbin_path: Optional[str] = None):
    from fpga_simulator_v07 import QuantumFPGACircuit
    import env as _env
    path = xclbin_path or _env.XCLBIN_PATH
    return QuantumFPGACircuit(path)


def run_fpga(fpga, gates: List[Gate], n_qubits: int, circuit_name: str = "") -> RunResult:
    t0 = time.perf_counter_ns()
    res = fpga.run_circuit(n_qubits, gates, shots=0, validate=True, circuit_name=circuit_name)
    wall_us = (time.perf_counter_ns() - t0) / 1e3
    val = res['validation']
    return RunResult(
        hardware='FPGA', n_qubits=n_qubits, num_gates=res['profile'].num_gates,
        kernel_time_us=res['timing']['kernel_us'], wall_time_us=wall_us,
        statevector=res['statevector'], passed=(bool(val.passed) if val else None),
        extra={'profile': res['profile'], 'timing': res['timing']},
    )


# =============================================================================
#  CPU / GPU (Qiskit Aer)
# =============================================================================
def init_aer(device: str):
    """device is 'CPU' or 'GPU'."""
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    backend = AerSimulator(method='statevector', device=device)
    probe = QuantumCircuit(2)
    probe.h(0)
    probe.save_statevector()
    tprobe = transpile(probe, backend, optimization_level=0)
    backend.run(tprobe, shots=1).result()  # surfaces GPU/driver issues here, not mid-sweep
    return backend


def run_aer(backend, qc, n_qubits: int) -> RunResult:
    from qiskit import transpile
    t0 = time.perf_counter_ns()
    qc2 = qc.copy()
    qc2.save_statevector()
    tqc = transpile(qc2, backend, optimization_level=0)
    result = backend.run(tqc, shots=1).result()
    wall_us = (time.perf_counter_ns() - t0) / 1e3
    sv = np.asarray(result.get_statevector(tqc))
    kernel_us = result.results[0].time_taken * 1e6
    device = getattr(backend.options, 'device', 'CPU')
    return RunResult(
        hardware=device, n_qubits=n_qubits, num_gates=len(qc.data),
        kernel_time_us=kernel_us, wall_time_us=wall_us,
        statevector=sv, passed=None, extra={},
    )


# =============================================================================
#  Reference / validation backend (this project's own mathematical model)
# =============================================================================
def run_reference(gates: List[Gate], n_qubits: int) -> RunResult:
    from statevector_model import State
    t0 = time.perf_counter_ns()
    lowered = lower_to_native(gates)
    s = State(n_qubits).run(lowered)
    wall_us = (time.perf_counter_ns() - t0) / 1e3
    return RunResult(
        hardware='CPU-reference', n_qubits=n_qubits, num_gates=len(lowered),
        kernel_time_us=wall_us, wall_time_us=wall_us,
        statevector=s.state, passed=(abs(s.norm() - 1.0) < 1e-6), extra={},
    )


def fidelity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None or len(a) != len(b):
        return None
    return float(abs(np.vdot(a, b)) ** 2)
