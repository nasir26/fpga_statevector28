"""
statevector_model.py
=====================
The mathematical model behind the whole project, and a small reference
implementation of it, independent of Qiskit and of any hardware backend.

1. The model
------------
An n-qubit pure state is a unit vector

    |psi> = sum_{i=0}^{2^n - 1} c_i |i>,   c_i in C,   sum_i |c_i|^2 = 1

represented here as a flat NumPy array `state` of length N = 2**n, where
index i's binary expansion (i_{n-1} ... i_1 i_0) gives the computational
basis state of qubits (n-1 ... 1 0). This is the *mathematical* model:
psi is a single complex vector in C^N.

A gate acting on qubit q is a linear map that only mixes the two basis
states differing in bit q: for a 2x2 unitary U = [[a,b],[c,d]],

    c_i'   = a*c_i + b*c_j
    c_j'   = c*c_i + d*c_j        where j = i with bit q flipped

Implemented below via `np.moveaxis` on the state reshaped to a rank-n
tensor of shape (2,)*n — qubit q becomes tensor axis q, and applying the
gate is a small matmul contracted along that axis. This is the textbook
"tensor network of one node" view of a statevector simulator and is
algorithmically identical to what every other statevector backend in this
project does (Aer's C++ kernels, the FPGA's HLS kernel) — only the loop
nesting / hardware mapping differs.

Two-qubit gates (cx, cz — the only two-qubit gates the FPGA kernel has
natively) act analogously on the pair of tensor axes (control, target).

2. The hardware realization
----------------------------
The FPGA kernel (`opt_kernel.cpp` in the sibling repo) does not store psi
as a single complex array. AXI/HBM ports move real scalars, so the kernel
splits

    psi = state_real + i * state_imag,   state_real, state_imag in R^N

into two separate flat float64 HBM buffers (see fpga_simulator_v07.py:
`_distribute_statevector` / `_collect_statevector`). The bijection
psi <-> (state_real, state_imag) is exact and lossless; `to_split_arrays`/
`from_split_arrays` below implement it, so this model's output can be
compared element-for-element against what comes back from real hardware.

3. The cost model
-------------------
The FPGA driver's own profiler (`fpga_simulator_v07._profile_gates`)
estimates, per gate, a cycle count and HBM byte count from simple closed
forms:

    single-qubit gate (touches N/2 amplitude PAIRS):
        cycles   = (N/2) * PIPELINE_CYCLES_PER_PAIR
        hbm_bytes = (N/2) * 2 * BYTES_PER_AMP      (read + write, per pair)

    cx / cz (touches all N amplitudes, one HBM port each):
        cycles   = N * PIPELINE_CYCLES_PER_PAIR
        hbm_bytes = N * BYTES_PER_AMP

with PIPELINE_CYCLES_PER_PAIR = 16 (the HLS pipeline's initiation-interval-
scaled latency per complex multiply-accumulate pair, from opt_kernel.cpp)
and BYTES_PER_AMP = 16 (complex128 = 2 x float64). `predict_profile` below
re-derives that SAME closed form independently (not by importing the
driver's private function) so it stands on its own as a model, and
`benchmark_28q.py` cross-checks it against the driver's own
`est_kernel_time_us`/`total_hbm_read_mb`/`total_hbm_write_mb` AND against
the wall-clock time actually measured on the real Alveo U55C — i.e. the
model is falsifiable against silicon, not just asserted.

The same closed form gives an a-priori reason ZX-calculus optimization
should help: total predicted HBM traffic is linear in gate count at fixed
n, so any circuit rewrite that removes gates removes HBM traffic (and
hence FPGA/CPU/GPU time) roughly proportionally, *provided* the removed
gates are the expensive kind (two-qubit gates cost 2x what single-qubit
gates cost per the model above, since N vs N/2 amplitudes touched) rather
than merely fewer-but-cheaper gates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── Hardware-realization constants (matches opt_kernel.cpp / fpga_simulator_v07.py) ──
BYTES_PER_AMP           = 16       # complex128 = 2 x float64
BYTES_PER_F64           = 8
PIPELINE_CYCLES_PER_PAIR = 16      # HLS pipeline latency per amplitude-pair op
KERNEL_CLOCK_MHZ        = 225.0
KERNEL_CLOCK_PERIOD_NS  = 1e3 / KERNEL_CLOCK_MHZ

TWO_QUBIT_NATIVE = frozenset({'cx', 'cnot', 'cz'})

Gate = Tuple[str, Sequence[int], Optional[Sequence[float]]]


# =============================================================================
#  Native single-qubit gate matrices (must match KERNEL_GATE_OPS in
#  fpga_simulator_v07.py — same names, same semantics)
# =============================================================================
def _rot_matrix(name: str, params: Optional[Sequence[float]]) -> np.ndarray:
    p = list(params) if params else []
    if name in ('id',):
        return np.eye(2, dtype=np.complex128)
    if name == 'x':
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    if name == 'y':
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    if name == 'z':
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    if name == 'h':
        return (1 / math.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    if name == 's':
        return np.array([[1, 0], [0, 1j]], dtype=np.complex128)
    if name == 'sdg':
        return np.array([[1, 0], [0, -1j]], dtype=np.complex128)
    if name == 't':
        return np.array([[1, 0], [0, np.exp(1j * math.pi / 4)]], dtype=np.complex128)
    if name == 'tdg':
        return np.array([[1, 0], [0, np.exp(-1j * math.pi / 4)]], dtype=np.complex128)
    if name == 'sx':
        return 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128)
    if name == 'sxdg':
        return 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]], dtype=np.complex128)
    if name == 'rx':
        th = p[0]
        c, s = math.cos(th / 2), math.sin(th / 2)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
    if name == 'ry':
        th = p[0]
        c, s = math.cos(th / 2), math.sin(th / 2)
        return np.array([[c, -s], [s, c]], dtype=np.complex128)
    if name == 'rz':
        th = p[0]
        return np.array([[np.exp(-1j * th / 2), 0], [0, np.exp(1j * th / 2)]], dtype=np.complex128)
    if name in ('p', 'phase', 'u1'):
        lam = p[0]
        return np.array([[1, 0], [0, np.exp(1j * lam)]], dtype=np.complex128)
    if name == 'u2':
        phi, lam = p[0], p[1]
        f = 1 / math.sqrt(2)
        return f * np.array([[1, -np.exp(1j * lam)],
                              [np.exp(1j * phi), np.exp(1j * (phi + lam))]], dtype=np.complex128)
    if name in ('u3', 'u'):
        th, phi, lam = p[0], p[1], p[2]
        c, s = math.cos(th / 2), math.sin(th / 2)
        return np.array([[c, -np.exp(1j * lam) * s],
                          [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]], dtype=np.complex128)
    raise ValueError(f"Not a native single-qubit gate: {name!r}")


# =============================================================================
#  State: a psi in C^N, plus in-place gate application
# =============================================================================
class State:
    def __init__(self, n_qubits: int, vec: Optional[np.ndarray] = None):
        self.n = n_qubits
        self.N = 1 << n_qubits
        if vec is None:
            self.state = np.zeros(self.N, dtype=np.complex128)
            self.state[0] = 1.0
        else:
            assert vec.shape == (self.N,)
            self.state = vec.astype(np.complex128, copy=True)

    # -- hardware bijection: psi <-> (Re(psi), Im(psi)) --------------------
    def to_split_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.ascontiguousarray(self.state.real), np.ascontiguousarray(self.state.imag)

    @classmethod
    def from_split_arrays(cls, real: np.ndarray, imag: np.ndarray) -> "State":
        n = int(math.log2(len(real)))
        return cls(n, real.astype(np.complex128) + 1j * imag.astype(np.complex128))

    # -- gate application ---------------------------------------------------
    def _axis(self, qubit: int) -> int:
        # Qiskit's convention is little-endian: qubit 0 is the LEAST
        # significant bit of the basis-state index. A flat array reshaped
        # to (2,)*n in C order has axis 0 as the MOST significant bit, so
        # qubit q lives on tensor axis (n-1-q), not axis q.
        return self.n - 1 - qubit

    def apply_1q(self, name: str, qubit: int, params: Optional[Sequence[float]] = None) -> "State":
        u = _rot_matrix(name, params)
        ax = self._axis(qubit)
        t = self.state.reshape((2,) * self.n)
        t = np.moveaxis(t, ax, 0)
        shape = t.shape
        t = (u @ t.reshape(2, -1)).reshape(shape)
        t = np.moveaxis(t, 0, ax)
        self.state = t.reshape(-1)
        return self

    def apply_cx(self, control: int, target: int) -> "State":
        axc, axt = self._axis(control), self._axis(target)
        t = self.state.reshape((2,) * self.n)
        t = np.moveaxis(t, [axc, axt], [0, 1])
        t[1, 0, ...], t[1, 1, ...] = t[1, 1, ...].copy(), t[1, 0, ...].copy()
        t = np.moveaxis(t, [0, 1], [axc, axt])
        self.state = t.reshape(-1)
        return self

    def apply_cz(self, control: int, target: int) -> "State":
        axc, axt = self._axis(control), self._axis(target)
        t = self.state.reshape((2,) * self.n)
        t = np.moveaxis(t, [axc, axt], [0, 1])
        t[1, 1, ...] *= -1
        t = np.moveaxis(t, [0, 1], [axc, axt])
        self.state = t.reshape(-1)
        return self

    def apply(self, gate: Gate) -> "State":
        name, qubits, params = gate[0], gate[1], (gate[2] if len(gate) > 2 else None)
        name = name.lower()
        if name in ('cx', 'cnot'):
            return self.apply_cx(int(qubits[0]), int(qubits[1]))
        if name == 'cz':
            return self.apply_cz(int(qubits[0]), int(qubits[1]))
        return self.apply_1q(name, int(qubits[0]), params)

    def run(self, gates: Sequence[Gate]) -> "State":
        for g in gates:
            self.apply(g)
        return self

    def norm(self) -> float:
        return float(np.linalg.norm(self.state))

    def probabilities(self) -> np.ndarray:
        return (self.state.real ** 2 + self.state.imag ** 2)


# =============================================================================
#  Cost model: predicted cycles / HBM bytes / time, per the closed form in
#  the module docstring. Independent re-derivation of the FPGA driver's own
#  `_profile_gates` formula, meant to be cross-checked against it and
#  against real measured `xrt_kernel_us`.
# =============================================================================
@dataclass
class PredictedProfile:
    n_qubits: int
    num_gates: int
    num_single_qubit: int
    num_two_qubit: int
    total_cycles: int
    total_hbm_bytes: int
    predicted_kernel_time_us: float
    predicted_hbm_read_mb: float
    predicted_hbm_write_mb: float


def predict_profile(gates: Sequence[Gate], n_qubits: int,
                     clock_mhz: float = KERNEL_CLOCK_MHZ) -> PredictedProfile:
    n = 1 << n_qubits
    pairs = n >> 1
    period_ns = 1e3 / clock_mhz
    total_cycles = 0
    total_bytes = 0
    n1q = n2q = 0
    for g in gates:
        name = g[0].lower() if isinstance(g[0], str) else g[0]
        if name in TWO_QUBIT_NATIVE:
            total_cycles += n * PIPELINE_CYCLES_PER_PAIR
            total_bytes += n * BYTES_PER_AMP
            n2q += 1
        else:
            total_cycles += pairs * PIPELINE_CYCLES_PER_PAIR
            total_bytes += pairs * 2 * BYTES_PER_AMP
            n1q += 1
    return PredictedProfile(
        n_qubits=n_qubits, num_gates=len(gates),
        num_single_qubit=n1q, num_two_qubit=n2q,
        total_cycles=total_cycles, total_hbm_bytes=total_bytes,
        predicted_kernel_time_us=total_cycles * period_ns / 1e3,
        predicted_hbm_read_mb=total_bytes / 2 / (1024 ** 2),
        predicted_hbm_write_mb=total_bytes / 2 / (1024 ** 2),
    )


def state_memory_bytes(n_qubits: int, complex128: bool = True) -> int:
    """Bytes to hold one full statevector on a given backend."""
    return (1 << n_qubits) * (16 if complex128 else 8)


if __name__ == '__main__':
    # Tiny self-check: H|0> should give the |+> state.
    s = State(1).apply_1q('h', 0)
    print("H|0> =", s.state, " norm =", s.norm())
    prof = predict_profile([('h', [0], None)], 1)
    print(prof)
