#!/usr/bin/env python3
"""
benchmark_28q.py
==================
Main benchmark for the FPGA-based 28-statevector use case.

Three things happen here, in order:

  1. SWEEP  — QFT and Clifford+T circuits over a modest qubit range
     (default 2..18) on every available backend (FPGA / CPU / GPU),
     producing scaling curves and cross-checking every real hardware
     result against this project's own mathematical-model reference
     (`statevector_model.State`) for correctness, and against the
     closed-form cost model (`statevector_model.predict_profile`) for
     the FPGA timing prediction.

  2. ZX COMPARISON — the same circuit families, run once raw and once
     through ZX-calculus reduction (`zx_optimizer.zx_optimize`), so the
     benchmark reports honestly whether the optimizer helped (expected:
     yes for Clifford+T, not necessarily for QFT — see circuits.py).

  3. HEADLINE — the actual 28-qubit runs (`--headline`), the qubit count
     the FPGA kernel (Alveo U55C, MAX_QUBITS_FPGA=28) was built for. Off
     by default since a real 28-qubit statevector is ~4 GiB per backend
     and this can take a while; pass --headline to run it.

Outputs land in results/: a CSV per stage and PNG plots.
"""
from __future__ import annotations

import argparse
import csv
import os
import time
import traceback
from dataclasses import asdict
from typing import Dict, List, Optional

import env  # noqa: F401  (sys.path + CUDA LD_LIBRARY_PATH fix — import first)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from circuits import qft_circuit, grover_circuit, clifford_t_circuit, lower_to_native
from zx_optimizer import zx_optimize
from statevector_model import predict_profile, state_memory_bytes
import backends as bk

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Palette (dataviz skill, categorical slots 1/2/3 — validated CVD-safe order)
COLOR = {'FPGA': '#2a78d6', 'CPU': '#eb6834', 'GPU': '#1baf7a'}
MARKER = {'FPGA': 'o', 'CPU': 's', 'GPU': '^'}
GRID_KW = dict(color='#d9d9d6', linewidth=0.6, alpha=0.6)

REFERENCE_MAX_QUBITS = 16  # above this, skip the O(2^n)-memory reference fidelity check


# =============================================================================
#  Backend bring-up
# =============================================================================
def bring_up_backends() -> Dict[str, object]:
    handles = {}
    print("Initialising backends ...")
    try:
        handles['FPGA'] = bk.init_fpga()
        print("  [ok]   FPGA (Alveo U55C)")
    except Exception as e:
        print(f"  [skip] FPGA: {e}")
    for dev in ('CPU', 'GPU'):
        try:
            handles[dev] = bk.init_aer(dev)
            print(f"  [ok]   {dev} (Qiskit Aer)")
        except Exception as e:
            print(f"  [skip] {dev}: {e}")
    if not handles:
        raise RuntimeError("No backend available — nothing to benchmark.")
    print(f"Active hardware: {', '.join(handles)}\n")
    return handles


def run_one(hw: str, handle, gates, qc, n: int, name: str) -> Optional[bk.RunResult]:
    try:
        if hw == 'FPGA':
            return bk.run_fpga(handle, gates, n, circuit_name=name)
        return bk.run_aer(handle, qc, n)
    except Exception as e:
        print(f"    [{hw} @ {name}] {type(e).__name__}: {e}")
        return None


# =============================================================================
#  1. Sweep
# =============================================================================
def run_sweep(handles: Dict[str, object], min_n: int, max_n: int) -> List[dict]:
    rows = []
    families = {
        'QFT': lambda n: qft_circuit(n),
        'Clifford+T': lambda n: clifford_t_circuit(n, depth=max(4, n * 2), seed=42),
    }
    print(f"Sweep: n={min_n}..{max_n}, circuits={list(families)}")
    for fam_name, build in families.items():
        for n in range(min_n, max_n + 1):
            gates, qc = build(n)
            lowered = lower_to_native(gates)
            pred = predict_profile(lowered, n)
            ref_sv = None
            if n <= REFERENCE_MAX_QUBITS:
                ref_sv = bk.run_reference(gates, n).statevector
            line = f"  {fam_name:10s} n={n:2d} gates={len(lowered):5d}"
            for hw, handle in handles.items():
                r = run_one(hw, handle, gates, qc, n, f"{fam_name}{n}")
                if r is None:
                    continue
                fid = bk.fidelity(r.statevector, ref_sv) if ref_sv is not None else None
                rows.append({
                    'stage': 'sweep', 'circuit': fam_name, 'n_qubits': n, 'hardware': hw,
                    'num_gates': r.num_gates, 'kernel_time_us': r.kernel_time_us,
                    'wall_time_us': r.wall_time_us, 'fidelity_vs_reference': fid,
                    'predicted_fpga_kernel_time_us': pred.predicted_kernel_time_us,
                })
                line += f"  {hw}={r.kernel_time_us:9.2f}us"
                if fid is not None:
                    line += f"({fid:.4f})"
            print(line)
    return rows


# =============================================================================
#  2. ZX-calculus comparison
# =============================================================================
def run_zx_comparison(handles: Dict[str, object], n_qft: int, n_ct: int, ct_depth: int) -> List[dict]:
    rows = []
    cases = [
        ('QFT', *qft_circuit(n_qft), n_qft),
        ('Clifford+T', *clifford_t_circuit(n_ct, depth=ct_depth, seed=7), n_ct),
    ]
    print(f"\nZX-calculus comparison (QFT n={n_qft}, Clifford+T n={n_ct} depth={ct_depth})")
    for fam_name, gates, qc, n in cases:
        qc_opt, stats = zx_optimize(qc)
        gates_opt = _qiskit_to_gates(qc_opt)
        print(f"  {fam_name}: gate count {stats.gate_count_before} -> {stats.gate_count_after} "
              f"({stats.gate_reduction_pct:+.1f}%), T-count {stats.t_count_before} -> {stats.t_count_after}")
        for label, g, c in (('raw', gates, qc), ('zx_optimized', gates_opt, qc_opt)):
            lowered = lower_to_native(g)
            for hw, handle in handles.items():
                r = run_one(hw, handle, g, c, n, f"{fam_name}_{label}")
                if r is None:
                    continue
                rows.append({
                    'stage': 'zx_comparison', 'circuit': fam_name, 'variant': label,
                    'n_qubits': n, 'hardware': hw, 'num_gates': len(lowered),
                    'kernel_time_us': r.kernel_time_us, 'wall_time_us': r.wall_time_us,
                    'gate_reduction_pct': stats.gate_reduction_pct,
                })
                print(f"    {label:14s} {hw:5s} gates={len(lowered):5d} kernel={r.kernel_time_us:9.2f}us")
    return rows


def _qiskit_to_gates(qc) -> list:
    """Turn a Qiskit circuit back into (name, qubits, params) tuples in the
    vocabulary circuits.py / fpga_simulator_v07 expect."""
    out = []
    for instr in qc.data:
        name = instr.operation.name.lower()
        qubits = [qc.find_bit(q).index for q in instr.qubits]
        params = [float(p) for p in instr.operation.params] if instr.operation.params else None
        out.append((name, qubits, params))
    return out


# =============================================================================
#  3. Headline: the actual 28-qubit runs
# =============================================================================
def run_headline(handles: Dict[str, object], n: int, grover_iters: int, ct_depth: int,
                  include_grover: bool = False) -> List[dict]:
    rows = []
    print(f"\n{'=' * 72}\nHEADLINE: n={n} qubits "
          f"({state_memory_bytes(n) / 1e9:.2f} GiB per full-precision statevector copy)\n{'=' * 72}")
    cases = [
        ('QFT', *qft_circuit(n)),
        ('Clifford+T', *clifford_t_circuit(n, depth=ct_depth, seed=99)),
    ]
    if include_grover:
        # An (n-1)-control MCX (the oracle/diffusion primitive) decomposes,
        # ancilla-free, into O(n^2)-ish native gates with a real constant —
        # 13.2k gates for 27 controls alone (see _selftest_mcx_scaling.py) —
        # and mcz_all_qubits() calls it twice per Grover iteration. That's
        # tractable for one iteration but explicitly NOT a real search (a
        # real search at n=28 needs ~2^13 iterations); off by default so a
        # plain --headline run stays fast, opt in deliberately.
        cases.append((f'Grover(iters={grover_iters}, not a full search)',
                       *grover_circuit(n, marked=0, iterations=grover_iters)))
    else:
        print("  [skip] Grover at headline n — pass --include-grover-headline to run it "
              "(single-iteration only, not a real search; see comment in run_headline).")
    for fam_name, gates, qc in cases:
        qc_opt, stats = zx_optimize(qc)
        gates_opt = _qiskit_to_gates(qc_opt)
        print(f"\n{fam_name}: raw {stats.gate_count_before} gates, "
              f"zx-optimized {stats.gate_count_after} gates ({stats.gate_reduction_pct:+.1f}%)")
        for label, g, c in (('raw', gates, qc), ('zx_optimized', gates_opt, qc_opt)):
            lowered = lower_to_native(g)
            pred = predict_profile(lowered, n)
            for hw, handle in handles.items():
                t0 = time.perf_counter()
                r = run_one(hw, handle, g, c, n, f"{fam_name}_{label}_{n}q")
                dt = time.perf_counter() - t0
                if r is None:
                    continue
                rows.append({
                    'stage': 'headline', 'circuit': fam_name, 'variant': label, 'n_qubits': n,
                    'hardware': hw, 'num_gates': len(lowered),
                    'kernel_time_us': r.kernel_time_us, 'wall_time_us': r.wall_time_us,
                    'passed': r.passed, 'predicted_fpga_kernel_time_us': pred.predicted_kernel_time_us,
                })
                print(f"  {label:14s} {hw:5s} gates={len(lowered):5d}  "
                      f"kernel={r.kernel_time_us/1e3:10.2f} ms  wall={r.wall_time_us/1e3:10.2f} ms  "
                      f"(script wall {dt:.1f}s)  passed={r.passed}")
    return rows


# =============================================================================
#  CSV + plots
# =============================================================================
def write_csv(rows: List[dict], path: str):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path}")


def plot_sweep(rows: List[dict]):
    families = sorted({r['circuit'] for r in rows if r['stage'] == 'sweep'})
    for fam in families:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        for hw in ('FPGA', 'CPU', 'GPU'):
            pts = sorted([(r['n_qubits'], r['kernel_time_us']) for r in rows
                          if r['stage'] == 'sweep' and r['circuit'] == fam and r['hardware'] == hw])
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker=MARKER[hw], color=COLOR[hw], linewidth=2, markersize=6, label=hw)
        pred_pts = sorted({(r['n_qubits'], r['predicted_fpga_kernel_time_us']) for r in rows
                            if r['stage'] == 'sweep' and r['circuit'] == fam})
        if pred_pts:
            xs, ys = zip(*pred_pts)
            ax.plot(xs, ys, '--', color='#52514e', linewidth=1.4, label='FPGA cost model (predicted)')
        ax.set_yscale('log')
        ax.set_xlabel('Qubits (n)')
        ax.set_ylabel('Kernel time (µs, log scale)')
        ax.set_title(f'{fam} — statevector simulation speed vs qubit count')
        ax.grid(True, which='both', **GRID_KW)
        ax.legend(frameon=False)
        fig.tight_layout()
        out = os.path.join(RESULTS_DIR, f"sweep_{fam.replace('+', 'T').replace(' ', '_')}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  wrote {out}")


def plot_zx_comparison(rows: List[dict]):
    zc = [r for r in rows if r['stage'] == 'zx_comparison']
    if not zc:
        return
    families = sorted({r['circuit'] for r in zc})
    hws = sorted({r['hardware'] for r in zc})
    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 5), squeeze=False)
    for ax, fam in zip(axes[0], families):
        width = 0.35
        x = np.arange(len(hws))
        for i, variant in enumerate(('raw', 'zx_optimized')):
            ys = [next((r['kernel_time_us'] for r in zc
                        if r['circuit'] == fam and r['hardware'] == hw and r['variant'] == variant), 0)
                  for hw in hws]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, ys, width, label=variant,
                           color=[COLOR[hw] for hw in hws], alpha=(1.0 if variant == 'zx_optimized' else 0.45),
                           edgecolor='#0b0b0b', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(hws)
        ax.set_ylabel('Kernel time (µs)')
        ax.set_title(fam)
        ax.grid(True, axis='y', **GRID_KW)
        ax.legend(frameon=False)
    fig.suptitle('ZX-calculus circuit optimization: raw vs reduced, per hardware')
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "zx_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_headline(rows: List[dict], n: int):
    hl = [r for r in rows if r['stage'] == 'headline']
    if not hl:
        return
    families = sorted({r['circuit'] for r in hl})
    hws = sorted({r['hardware'] for r in hl})
    fig, ax = plt.subplots(figsize=(max(8, 3 * len(families)), 6))
    width = 0.8 / max(1, len(hws))
    x = np.arange(len(families))
    for i, hw in enumerate(hws):
        ys = [next((r['kernel_time_us'] / 1e3 for r in hl
                    if r['circuit'] == fam and r['hardware'] == hw and r['variant'] == 'zx_optimized'), 0)
              for fam in families]
        ax.bar(x + (i - (len(hws) - 1) / 2) * width, ys, width, label=hw, color=COLOR[hw],
               edgecolor='#0b0b0b', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(families, rotation=15, ha='right')
    ax.set_ylabel('Kernel time (ms)')
    ax.set_yscale('log')
    ax.set_title(f'{n}-qubit statevector — FPGA vs CPU vs GPU (ZX-optimized circuits)')
    ax.grid(True, axis='y', **GRID_KW)
    ax.legend(frameon=False)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, f"headline_{n}q.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


# =============================================================================
#  Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sweep-min', type=int, default=2)
    ap.add_argument('--sweep-max', type=int, default=18)
    ap.add_argument('--no-sweep', action='store_true')
    ap.add_argument('--no-zx', action='store_true')
    ap.add_argument('--zx-qft-n', type=int, default=8)
    ap.add_argument('--zx-ct-n', type=int, default=8)
    ap.add_argument('--zx-ct-depth', type=int, default=30)
    ap.add_argument('--headline', action='store_true', help='run the real 28-qubit case (slow, ~4GiB/backend)')
    ap.add_argument('--headline-n', type=int, default=28)
    ap.add_argument('--headline-ct-depth', type=int, default=40)
    ap.add_argument('--grover-iters', type=int, default=1,
                     help='Grover iterations at headline n (a real search needs ~2^(n/2); '
                          'default 1 is a fixed workload size for benchmarking, not a real search)')
    ap.add_argument('--include-grover-headline', action='store_true',
                     help='include Grover in the headline run (off by default — see run_headline)')
    args = ap.parse_args()

    handles = bring_up_backends()
    all_rows: List[dict] = []

    if not args.no_sweep:
        rows = run_sweep(handles, args.sweep_min, args.sweep_max)
        all_rows += rows
        write_csv(rows, os.path.join(RESULTS_DIR, 'sweep.csv'))
        plot_sweep(rows)

    if not args.no_zx:
        rows = run_zx_comparison(handles, args.zx_qft_n, args.zx_ct_n, args.zx_ct_depth)
        all_rows += rows
        write_csv(rows, os.path.join(RESULTS_DIR, 'zx_comparison.csv'))
        plot_zx_comparison(rows)

    if args.headline:
        rows = run_headline(handles, args.headline_n, args.grover_iters, args.headline_ct_depth,
                             include_grover=args.include_grover_headline)
        all_rows += rows
        write_csv(rows, os.path.join(RESULTS_DIR, f'headline_{args.headline_n}q.csv'))
        plot_headline(rows, args.headline_n)

    write_csv(all_rows, os.path.join(RESULTS_DIR, 'all_results.csv'))
    print("\nDone.")


if __name__ == '__main__':
    main()
