# FPGA 28-Qubit Statevector Simulation — Mathematical Model, ZX-Calculus Optimization, and a CPU/GPU/FPGA Speed Comparison

This project studies one specific, concrete use case: simulating a
**28-qubit quantum statevector** — the largest the on-board FPGA kernel
supports — and measuring how fast that simulation runs on three real pieces
of hardware in this machine, from first principles.

It is a separate, self-contained directory from the sibling
[`fpga_mps_simulator`](../fpga_mps_simulator) repo. It does **not**
reimplement that repo's FPGA host driver or HLS kernel — those already
exist and are proven against real hardware — it imports and reuses them
(`env.py`), and adds:

1. a from-scratch **mathematical model** of statevector simulation and of
   the FPGA's own execution-time cost function (`statevector_model.py`),
2. a **ZX-calculus** ([zxcalculus.com](https://zxcalculus.com))
   circuit-optimization stage inserted before simulation (`zx_optimizer.py`),
3. a **benchmark** that runs identical circuits on FPGA, CPU, and GPU and
   reports real, measured numbers (`benchmark_28q.py`).

## Why 28 qubits specifically

It's not an arbitrary round number — it's the ceiling the FPGA build in
the sibling repo was actually configured for. `build_statevector/alveo_u55c.cfg`
gives the compiled kernel's `state_real` and `state_imag` HBM ports 8 HBM
banks each (256 MB/bank ⇒ 2 GiB/array); a full statevector at n qubits needs
`2^n × 8 bytes` per array, and `2^28 × 8 B = 2 GiB` exactly — so 28 is the
largest qubit count that HBM layout can hold without widening it. That
constant is recorded as `MAX_QUBITS_FPGA = 28` in
`fpga_simulator_v07.py`. This project takes that ceiling as the headline
benchmark point.

## Hardware in this machine

| Role | Device | Notes |
|---|---|---|
| FPGA | AMD/Xilinx Alveo U55C (`xilinx_u55c_gen3x16_xdma_base_3`) | via XRT 2.15.225, real hardware at `/dev/xclmgmt35840` |
| CPU  | AMD EPYC 7742, 64 cores | Qiskit Aer `AerSimulator(device='CPU')` |
| GPU  | NVIDIA RTX A4000, 16 GiB | Qiskit Aer `AerSimulator(device='GPU')`, `qiskit-aer-gpu` (CUDA 12.x) |

All three are driven for real by this project — nothing here is simulated-
in-software as a stand-in for hardware that isn't present.

## 1. The mathematical model (`statevector_model.py`)

An n-qubit pure state is a unit vector

```
|ψ⟩ = Σ_{i=0}^{2ⁿ-1} c_i |i⟩,     c_i ∈ ℂ,     Σ|c_i|² = 1
```

held as a flat array of length `N = 2ⁿ`. A gate on qubit `q` is a linear
map mixing only the two basis states whose index differs in bit `q`:

```
c_i' = a·c_i + b·c_j
c_j' = c·c_i + d·c_j        (j = i with bit q flipped, U = [[a,b],[c,d]])
```

`statevector_model.State` implements this by reshaping the flat state into
a rank-`n` tensor of shape `(2,)*n` and matrix-multiplying along the axis
for qubit `q` (`np.moveaxis` + a 2×2 or two-axis contraction) — the same
"one-node tensor network" view every statevector simulator here uses
(Aer's C++ kernels and the FPGA's HLS kernel included); only the loop
nesting / hardware mapping differs. It's independently validated against
Qiskit's Aer statevector simulator for QFT, Grover, and Clifford+T circuits
(fidelity 1.0 in every case — see `tests/test_correctness.py`), and against
real FPGA hardware output too.

**Hardware realization.** The FPGA kernel doesn't store `ψ` as one complex
array — AXI/HBM ports move real scalars, so `opt_kernel.cpp` splits
`ψ = Re(ψ) + i·Im(ψ)` into two separate flat `float64` HBM buffers. That
bijection is exact; `State.to_split_arrays()` / `.from_split_arrays()`
implement it so this model's output can be compared element-for-element
against what actually comes back off the card.

**The cost model.** The FPGA driver's own profiler estimates, per gate, a
cycle count and HBM byte count from a closed form derived from the HLS
pipeline's structure:

```
single-qubit gate (touches N/2 amplitude pairs):
    cycles     = (N/2) × 16
    hbm_bytes  = (N/2) × 2 × 16        (read + write, complex128 per pair)

cx / cz (touches all N amplitudes):
    cycles     = N × 16
    hbm_bytes  = N × 16
```

`predict_profile()` re-derives this independently and `benchmark_28q.py`
plots it against the driver's own estimate *and* the real measured kernel
time — i.e. the model is checked against silicon, not just asserted. One
honest finding from doing this (see `results/sweep_QFT.png`): the closed
form only captures the memory-bandwidth-bound term. At small qubit counts,
fixed per-gate kernel-launch/XRT-call overhead dominates and the FPGA is
*slower* than CPU/GPU; the predicted line only starts tracking the measured
line once the workload is big enough for HBM traffic to dominate over
launch overhead. That crossover is itself one of the things the benchmark
measures.

## 2. ZX-calculus optimization (`zx_optimizer.py`)

[ZX-calculus](https://zxcalculus.com) is a graphical/algebraic language for
quantum circuits: gates become "spiders" in a graph, and a set of proven
rewrite rules (spider fusion, bialgebra/Hopf, local complementation,
pivoting) lets you simplify the graph while preserving the exact linear
map it represents. This project uses [`pyzx`](https://github.com/Quantomatic/pyzx)
as a pre-processing pass:

```
Qiskit circuit → OpenQASM → pyzx graph → zx.full_reduce() → zx.extract_circuit()
               → OpenQASM → Qiskit circuit → lowered to FPGA-native gates
```

**This is deliberately not oversold.** ZX-calculus rewriting is strongest
on Clifford+T structure. `circuits.py` provides three circuit families
specifically so the comparison isn't cherry-picked:

- **Clifford+T** (`clifford_t_circuit`) — the regime ZX-calculus targets.
  Measured: **up to ~49% gate-count reduction** at depth 20-30 (see
  `results/zx_comparison.csv`), and full-precision fidelity 1.0 against the
  unoptimized circuit (`tests/test_correctness.py` checks this on every run).
- **QFT** (`qft_circuit`) — dominated by arbitrary-angle controlled-phase
  rotations, which don't participate in the same fusion rules. Measured:
  ZX reduction *increases* gate count here (circuit extraction re-expresses
  the phase ladder as more, smaller pieces). Reported as-is.
- **Grover** (`grover_circuit`) — mixed Clifford + multi-controlled
  structure, included in the small-n sweep for a middle case.

Every `zx_optimize()` output is checked for functional equivalence
(fidelity 1.0, up to global phase) against the original circuit before any
timing number is recorded.

## 3. The benchmark (`benchmark_28q.py`)

Three stages, run from `main()`:

1. **Sweep** (`--sweep-min`/`--sweep-max`, default 2..18) — QFT and
   Clifford+T on FPGA/CPU/GPU, kernel time vs qubit count, plus the FPGA
   cost-model prediction overlaid. `results/sweep_*.png`, `sweep.csv`.
2. **ZX comparison** (on by default, `--no-zx` to skip) — raw vs
   ZX-optimized, same circuit, all three backends. `results/zx_comparison.png/.csv`.
3. **Headline** (`--headline`, off by default — it's the expensive one) —
   the actual **28-qubit** run. A full-precision statevector at n=28 is
   `2^28 × 16 B ≈ 4 GiB`; each backend materializes at least one copy.
   Grover is excluded by default (`--include-grover-headline` to force it)
   because its oracle needs a 27-control multi-controlled-X, which — with
   no ancilla qubits available in this gate model — decomposes into
   **13,200 native gates for one such gate alone**
   (`_selftest_mcx_scaling.py` measured this scaling directly); a single
   Grover iteration at n=28 is a valid, if extreme, benchmark workload but
   not a real search (a real search needs ~2^13 iterations at this size).
   QFT-28 and Clifford+T-28 don't have this problem and run by default.

```bash
# from this directory, using the sibling repo's existing venv:
VENV=/home/cdac/Documents/fpga_mps_simulator/.venv/bin/python

# quick validation sweep + ZX comparison (few minutes)
$VENV benchmark_28q.py

# the real 28-qubit headline run (QFT-28, Clifford+T-28; slow)
$VENV benchmark_28q.py --no-sweep --no-zx --headline --headline-n 28
```

Results (CSV + PNG) land in `results/`.

## Correctness

`tests/test_correctness.py` cross-checks, for every circuit family:
`statevector_model.State` (this project's own reference) vs Qiskit Aer vs
real FPGA hardware output, by fidelity — not just norm. Run it before
trusting any timing number from a change to `circuits.py` or the lowering
path.

## Repo layout

```
env.py                    sys.path + env-var setup (sibling repo, XRT, CUDA fix)
statevector_model.py       the mathematical model + cost model (no qiskit dependency)
circuits.py                 QFT / Grover / Clifford+T circuit generators
zx_optimizer.py             pyzx-based ZX-calculus reduction pass
backends.py                  uniform FPGA/CPU/GPU/reference run wrappers
benchmark_28q.py             sweep + ZX comparison + 28-qubit headline, CSV + plots
tests/test_correctness.py    cross-backend fidelity checks
results/                     benchmark output (CSV + PNG), generated by running the above
```

## A note on this machine's environment

Two environment quirks this project works around (see comments in `env.py`):

- **`pyxrt`** (the FPGA driver's Python bindings) isn't a pip package — it
  ships inside the XRT install at `/opt/xilinx/xrt/python`, and needs
  `XILINX_XRT` set. `env.py` adds both.
- This shell's `LD_LIBRARY_PATH` points at an older system CUDA install
  that shadows the venv's own pip-installed `nvidia-*-cu12` wheels with a
  `libnvJitLink` too old for `qiskit-aer-gpu`'s bundled `libcusparse`,
  causing an `undefined symbol` `ImportError` on GPU backend import.
  `env.py` clears `LD_LIBRARY_PATH` and re-execs the interpreter once
  before anything that needs it is imported.

Every script in this project does `import env` first, for exactly this reason.

## Authorship

Written for a specific author's own GitHub repository — the commit history
for this directory is intended to be authored solely by that person, kept
separate from the sibling `fpga_mps_simulator` repo's own history.
