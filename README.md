# MathRoboRust

A Rust library that implements SO(3), SE(3), and CMTM operations. Python bindings live in a separate crate so the core does not depend on `pyo3`.

## Layout
- `src/so3.rs`: SO(3) rotation implementation
- `src/se3.rs`: SE(3) rotation and translation transforms
- `src/cmtm.rs`: 6×6 coupled motion transform matrices derived from SE(3)
- `src/lib.rs`: Rust API surface
- `python/`: PyO3 bindings crate + `pyproject.toml` for `uv`
- `tests/repro.rs`: Rust-only reproducibility tests
- `python/tests/test_python_repro.py`: parity checks for Python bindings
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

## Benchmark
Roughly inspect performance for many transform calls:
```bash
cargo run --release --example speed
```
