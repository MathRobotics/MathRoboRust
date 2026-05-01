# MathRoboRust

A Rust library that implements SO(3), SE(3), and CMTM operations. Python bindings live in a separate crate so the core does not depend on `pyo3`.

`vendor/MathRobo` is the pinned Python reference implementation used for parity checks and Python-side performance comparisons.

## Layout
- `src/so3.rs`: SO(3) rotation implementation
- `src/se3.rs`: SE(3) rotation and translation transforms
- `src/cmtm.rs`: 6×6 coupled motion transform matrices derived from SE(3)
- `src/lib.rs`: Rust API surface
- `python/`: PyO3 bindings crate + `pyproject.toml` for `uv`
- `python/test_support/`: shared helpers for vendored MathRobo comparisons
- `python/benchmarks/`: `pytest-benchmark` suites for Python-side speed comparisons
- `tests/repro.rs`: Rust-only reproducibility tests
- `python/tests/test_python_repro.py`: parity checks for Python bindings
- `bench/compare_mathrobo.py`: table benchmark for `mathrobors` vs `vendor/MathRobo`
- `bench/reports/`: checked-in benchmark JSON and markdown reports from the latest comparison run
- `examples/speed.rs`: simple throughput benchmark for repeated transforms

## Build
Build the core Rust crate:
```bash
cargo build --release
```

Build the Python extension module with `uv` + `maturin`:
```bash
cd python
uv sync --dev
uv run maturin develop --release
```

## Testing
Verify the Rust-only tests:
```bash
cargo test
```

Verify the Python bindings (build the extension first):
```bash
cd python
uv sync --dev
uv run maturin develop --release
uv run pytest
```

```bash
uv run pytest --benchmark-only -m dev --benchmark-sort=mean
```

## Benchmark
Roughly inspect performance for many transform calls:
```bash
cargo run --release --example speed
```

Compare the Python extension against the vendored `MathRobo` reference:
```bash
cd python
uv sync --dev
uv run maturin develop --release
uv run python ../bench/compare_mathrobo.py \
  --json-out ../bench/reports/mathrobo_benchmark.json \
  --report-out ../bench/reports/mathrobo_benchmark_report.md
```

`SE3inertia.hat` and `SE3inertia.hat_commute` are included as Rust-only benchmark cases because the vendored `MathRobo` numpy path currently errors for those APIs.
