"""
Environment bootstrap for this project.

This project deliberately does NOT reimplement the FPGA host driver — that
already exists, is proven against real Alveo U55C hardware, and lives in the
sibling `fpga_mps_simulator` repo as `fpga_simulator_v07.py`. This module
just puts that repo, and the system XRT Python bindings (`pyxrt`, which are
installed system-wide under XRT, not pip-installable), on `sys.path` so the
rest of this project can `import fpga_simulator_v07` / `import pyxrt`
directly.

Import this before importing either of those, anywhere in this project.
"""
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Sibling repo holding the proven fpga_simulator_v07.py host driver + the
# built quantum_fpga_kernel_complex128.xclbin. Override with an env var if
# this project is ever relocated relative to it.
FPGA_MPS_SIM_DIR = os.environ.get(
    "FPGA_MPS_SIM_DIR",
    os.path.normpath(os.path.join(THIS_DIR, "..", "fpga_mps_simulator")),
)

# System XRT install's Python bindings (pyxrt .so), matching the venv's
# CPython ABI (built for 3.8 here). Not on PyPI — must come from the XRT
# install itself.
XRT_INSTALL_DIR = os.environ.get("XILINX_XRT", "/opt/xilinx/xrt")
XRT_PYTHON_DIR = os.environ.get("XRT_PYTHON_DIR", os.path.join(XRT_INSTALL_DIR, "python"))

for _p in (FPGA_MPS_SIM_DIR, XRT_PYTHON_DIR):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# pyxrt itself only needs XILINX_XRT set (it resolves libxrt_coreutil.so.2
# etc. via its own $ORIGIN-relative RUNPATH, not LD_LIBRARY_PATH — so this
# does NOT need to touch LD_LIBRARY_PATH and doesn't reintroduce the CUDA
# conflict _fix_cuda_ld_path() below removes).
if os.path.isdir(XRT_INSTALL_DIR):
    os.environ.setdefault("XILINX_XRT", XRT_INSTALL_DIR)

XCLBIN_PATH = os.environ.get(
    "XCLBIN_PATH",
    os.path.join(FPGA_MPS_SIM_DIR, "quantum_fpga_kernel_complex128.xclbin"),
)


def _fix_cuda_ld_path():
    """
    This machine's shell profile exports LD_LIBRARY_PATH=/usr/local/cuda/lib64,
    an older system CUDA install. That shadows the venv's own pip-installed
    nvidia-*-cu12 wheels (which qiskit-aer-gpu depends on and normally finds
    via each .so's own RPATH, no LD_LIBRARY_PATH needed) with a libnvJitLink
    too old to provide the versioned symbols qiskit-aer-gpu's bundled
    libcusparse needs, e.g.:

        ImportError: .../nvidia/cusparse/lib/libcusparse.so.12: undefined
        symbol: __nvJitLinkGetErrorLogSize_12_9, version libnvJitLink.so.12

    glibc's dynamic linker reads LD_LIBRARY_PATH once at process start, so
    clearing os.environ after the fact doesn't help already-loaded objects.
    The fix is to drop the variable and re-exec this same interpreter before
    qiskit_aer (or anything importing it) is loaded — every script in this
    project imports `env` first, so this runs before that happens.
    """
    sentinel = "_FPGA28_CUDA_LDFIX_DONE"
    if os.environ.get(sentinel) or "LD_LIBRARY_PATH" not in os.environ:
        return
    os.environ.pop("LD_LIBRARY_PATH", None)
    os.environ[sentinel] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)


_fix_cuda_ld_path()
