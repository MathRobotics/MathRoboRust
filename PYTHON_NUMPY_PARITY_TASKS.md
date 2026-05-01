# Python Numpy Parity Tasks

Scope:
- Extend the numpy-only Python surface to cover `wrench` / `inertia` variants used by vendored `MathRobo`.
- Add Python-side comparison and timing code against vendored `MathRobo`.
- Write a benchmark report with measured speedups.

Status:
- Complete as of `2026-04-30`.

Completed work:
- Added numpy-only Python wrappers for `SO3wrench`, `SO3inertia`, `SE3wrench`, and the usable `SE3inertia` subset.
- Added parity coverage against vendored `MathRobo`, including wrench/inertia APIs and the vendor-failing `SE3inertia` paths.
- Extended the Python benchmark/comparison cases to cover the new variants and to record skipped vendor cases explicitly.
- Wrote the benchmark artifacts to `bench/reports/mathrobo_benchmark.json` and `bench/reports/mathrobo_benchmark_report.md`.

Notes:
- `SE3inertia.hat` and `SE3inertia.hat_commute` remain Rust-only benchmark cases because the vendored `MathRobo` numpy path currently raises `RecursionError` / `ValueError`.
