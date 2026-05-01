# Benchmarks

This directory contains two benchmark entry points:

- `bench.py`: per-call overhead for `PyO3` vs `C-ABI` / `CFFI`
- `compare_mathrobo.py`: Python-side comparison between `mathrobors` and the vendored `MathRobo` reference implementation

## Build

PyO3 module (from the existing Python crate):
```bash
cd python
uv sync --dev
uv run maturin develop --release
```

C-ABI shared library:
```bash
cd ffi
cargo build --release
```

Optional: install `cffi` in the Python env (needed for `bench.py --mode cffi` or `--mode all`):
```bash
uv add --project python --dev cffi
```

## Run

Compare `mathrobors` against the vendored `MathRobo` Python implementation:
```bash
cd python
uv run python ../bench/compare_mathrobo.py \
  --json-out ../bench/reports/mathrobo_benchmark.json \
  --report-out ../bench/reports/mathrobo_benchmark_report.md
```

The generated artifacts are:
- `bench/reports/mathrobo_benchmark.json`
- `bench/reports/mathrobo_benchmark_report.md`

`SO3wrench`, `SO3inertia`, and `SE3wrench` are included in the direct comparison matrix. `SE3inertia.hat` and `SE3inertia.hat_commute` are reported separately as Rust-only cases because the vendored `MathRobo` numpy implementation currently fails on those entry points.

Optional tuning:
```bash
cd python
uv run python ../bench/compare_mathrobo.py --repeat 5 --warmup 1 --loop-scale 1.0
```

Compare PyO3 against the C-ABI/CFFI bridge:
```bash
cd python
uv run python ../bench/bench.py --mode both
```

Options:
```bash
cd python
uv run python ../bench/bench.py --mode pyo3
uv run python ../bench/bench.py --mode cabi
uv run python ../bench/bench.py --mode cffi
uv run python ../bench/bench.py --mode all
uv run python ../bench/bench.py --iters 500000 --warmup 20000
uv run python ../bench/bench.py --lib /path/to/libmathroborust_cabi.so
```
