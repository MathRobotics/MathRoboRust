# MathRoboRust

A Rust library that implements SO(3), SE(3), and CMTM operations and exposes a unified Python interface through `pyo3`. The Rust and Python sides share the same API surface so results can be reproduced across languages.

## Layout
- `src/so3.rs`: SO(3) rotation implementation
- `src/se3.rs`: SE(3) rotation and translation transforms
- `src/cmtm.rs`: 6×6 coupled motion transform matrices derived from SE(3)
- `src/lib.rs`: Rust API and Python class exports
- `tests/repro.rs`: reproducibility tests between Rust implementations and Python bindings
- `examples/speed.rs`: simple throughput benchmark for repeated transforms

## Build
Build as a regular Rust crate:
```bash
cargo build --release
```

Build the Python extension module (example using `maturin`):
```bash
maturin develop --release
```

## Testing
Verify that the Rust and Python bindings produce identical results:
```bash
cargo test
```

## Benchmark
Roughly inspect performance for many transform calls:
```bash
cargo run --release --example speed
```
