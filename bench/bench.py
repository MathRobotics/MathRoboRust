#!/usr/bin/env python3
import argparse
import array
import ctypes
import math
import sys
import time
from pathlib import Path
import platform


def perf(label, iters, warmup, fn):
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - start
    us_per = (elapsed / iters) * 1e6
    print(f"{label}: {elapsed:.6f}s total, {us_per:.2f} us/op")


def default_lib_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    target = root / "ffi" / "target"
    system = platform.system().lower()
    if system.startswith("darwin"):
        ext = "dylib"
    elif system.startswith("windows"):
        ext = "dll"
    else:
        ext = "so"

    candidates = [
        target / "release" / f"libmathroborust_cabi.{ext}",
        target / "debug" / f"libmathroborust_cabi.{ext}",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_cabi(lib_path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(lib_path))

    lib.mr_so3_new.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_double]
    lib.mr_so3_new.restype = ctypes.c_void_p
    lib.mr_so3_free.argtypes = [ctypes.c_void_p]
    lib.mr_so3_free.restype = None
    lib.mr_so3_apply.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.mr_so3_apply.restype = None

    lib.mr_se3_new.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.mr_se3_new.restype = ctypes.c_void_p
    lib.mr_se3_free.argtypes = [ctypes.c_void_p]
    lib.mr_se3_free.restype = None
    lib.mr_se3_apply.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.mr_se3_apply.restype = None
    lib.mr_se3_exp.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.mr_se3_exp.restype = None
    lib.mr_se3_adjoint.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double)]
    lib.mr_se3_adjoint.restype = None

    return lib


def bench_pyo3(iters: int, warmup: int) -> None:
    try:
        import mathrobors as mr
    except Exception as exc:
        print("Failed to import mathrobors (PyO3 module).")
        print("Build it first from python/: uv run maturin develop --release")
        raise SystemExit(1) from exc

    axis = [0.0, 0.0, 1.0]
    angle = math.pi / 2.0

    so3 = mr.SO3.from_axis_angle(axis, angle)
    v = [1.0, 0.0, 0.0]
    out3 = array.array("d", [0.0, 0.0, 0.0])

    def so3_apply():
        so3.apply(v)

    perf("PyO3 SO3.apply", iters, warmup, so3_apply)

    def so3_apply_into():
        so3.apply_into(v, out3)

    perf("PyO3 SO3.apply_into", iters, warmup, so3_apply_into)

    se3 = mr.SE3.from_axis_angle_translation(axis, angle, [0.25, -0.5, 1.0])
    p = [1.0, 0.0, 0.0]
    out3_se3 = array.array("d", [0.0, 0.0, 0.0])

    def se3_apply():
        se3.apply(p)

    perf("PyO3 SE3.apply", iters, warmup, se3_apply)

    def se3_apply_into():
        se3.apply_into(p, out3_se3)

    perf("PyO3 SE3.apply_into", iters, warmup, se3_apply_into)

    twist = [0.1, -0.2, 0.3, 1.0, -2.0, 3.0]
    out16 = array.array("d", [0.0] * 16)

    def se3_exp():
        mr.SE3.exp(twist)

    perf("PyO3 SE3.exp", iters, warmup, se3_exp)

    def se3_exp_into():
        mr.SE3.exp_into(twist, out16)

    perf("PyO3 SE3.exp_into", iters, warmup, se3_exp_into)

    out36 = array.array("d", [0.0] * 36)

    def se3_adj():
        se3.mat_adj()

    perf("PyO3 SE3.mat_adj", iters, warmup, se3_adj)

    def se3_adj_into():
        se3.mat_adj_into(out36)

    perf("PyO3 SE3.mat_adj_into", iters, warmup, se3_adj_into)


def bench_cabi(iters: int, warmup: int, lib_path: Path) -> None:
    if not lib_path.exists():
        print(f"C-ABI library not found at {lib_path}")
        print("Build it first: (cd ffi && cargo build --release)")
        raise SystemExit(1)

    lib = load_cabi(lib_path)

    axis = (ctypes.c_double * 3)(0.0, 0.0, 1.0)
    angle = ctypes.c_double(math.pi / 2.0)
    vec = (ctypes.c_double * 3)(1.0, 0.0, 0.0)
    out = (ctypes.c_double * 3)()

    so3 = lib.mr_so3_new(axis, angle)
    if not so3:
        raise SystemExit("mr_so3_new returned null")

    def so3_apply():
        lib.mr_so3_apply(so3, vec, out)

    perf("C-ABI SO3.apply", iters, warmup, so3_apply)
    lib.mr_so3_free(so3)

    translation = (ctypes.c_double * 3)(0.25, -0.5, 1.0)
    se3 = lib.mr_se3_new(axis, angle, translation)
    if not se3:
        raise SystemExit("mr_se3_new returned null")

    def se3_apply():
        lib.mr_se3_apply(se3, vec, out)

    perf("C-ABI SE3.apply", iters, warmup, se3_apply)

    twist = (ctypes.c_double * 6)(0.1, -0.2, 0.3, 1.0, -2.0, 3.0)
    mat4 = (ctypes.c_double * 16)()
    mat6 = (ctypes.c_double * 36)()

    def se3_exp():
        lib.mr_se3_exp(twist, 1.0, mat4)

    perf("C-ABI SE3.exp", iters, warmup, se3_exp)

    def se3_adj():
        lib.mr_se3_adjoint(se3, mat6)

    perf("C-ABI SE3.adjoint", iters, warmup, se3_adj)
    lib.mr_se3_free(se3)


def bench_cffi(iters: int, warmup: int, lib_path: Path) -> None:
    try:
        import cffi  # type: ignore
    except Exception as exc:
        print("cffi is not installed in this Python environment.")
        print("Install it with: uv add --project python --dev cffi")
        raise SystemExit(1) from exc

    if not lib_path.exists():
        print(f"C-ABI library not found at {lib_path}")
        print("Build it first: (cd ffi && cargo build --release)")
        raise SystemExit(1)

    ffi = cffi.FFI()
    ffi.cdef(
        """
        void* mr_so3_new(const double* axis, double angle);
        void mr_so3_free(void* ptr);
        void mr_so3_apply(const void* ptr, const double* vector, double* out);
        void* mr_se3_new(const double* axis, double angle, const double* translation);
        void mr_se3_free(void* ptr);
        void mr_se3_apply(const void* ptr, const double* point, double* out);
        void mr_se3_exp(const double* twist, double scale, double* out);
        void mr_se3_adjoint(const void* ptr, double* out);
        """
    )
    lib = ffi.dlopen(str(lib_path))

    axis = ffi.new("double[3]", [0.0, 0.0, 1.0])
    angle = math.pi / 2.0
    vec = ffi.new("double[3]", [1.0, 0.0, 0.0])
    out = ffi.new("double[3]")

    so3 = lib.mr_so3_new(axis, angle)
    if so3 == ffi.NULL:
        raise SystemExit("mr_so3_new returned null")

    def so3_apply():
        lib.mr_so3_apply(so3, vec, out)

    perf("CFFI SO3.apply", iters, warmup, so3_apply)
    lib.mr_so3_free(so3)

    translation = ffi.new("double[3]", [0.25, -0.5, 1.0])
    se3 = lib.mr_se3_new(axis, angle, translation)
    if se3 == ffi.NULL:
        raise SystemExit("mr_se3_new returned null")

    def se3_apply():
        lib.mr_se3_apply(se3, vec, out)

    perf("CFFI SE3.apply", iters, warmup, se3_apply)

    twist = ffi.new("double[6]", [0.1, -0.2, 0.3, 1.0, -2.0, 3.0])
    mat4 = ffi.new("double[16]")
    mat6 = ffi.new("double[36]")

    def se3_exp():
        lib.mr_se3_exp(twist, 1.0, mat4)

    perf("CFFI SE3.exp", iters, warmup, se3_exp)

    def se3_adj():
        lib.mr_se3_adjoint(se3, mat6)

    perf("CFFI SE3.adjoint", iters, warmup, se3_adj)
    lib.mr_se3_free(se3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini bench: PyO3 vs C-ABI")
    parser.add_argument(
        "--mode",
        choices=["pyo3", "cabi", "cffi", "both", "all"],
        default="both",
    )
    parser.add_argument("--iters", type=int, default=200_000)
    parser.add_argument("--warmup", type=int, default=10_000)
    parser.add_argument("--lib", type=Path, default=default_lib_path())
    args = parser.parse_args()

    if args.mode in ("pyo3", "both", "all"):
        bench_pyo3(args.iters, args.warmup)

    if args.mode in ("cabi", "both", "all"):
        bench_cabi(args.iters, args.warmup, args.lib)

    if args.mode in ("cffi", "all"):
        bench_cffi(args.iters, args.warmup, args.lib)


if __name__ == "__main__":
    main()
