# MathRoboRust Benchmark Report

Generated: 2026-05-01 09:44:44
Command settings: `repeat=5`, `warmup=1`, `loop_scale=1.0`

## Summary

- Compared `22` Rust-vs-vendor cases.
- Median speedup over vendor `mathrobo`: `2.38x`.
- Largest observed speedup: `SE3.exp` at `21.35x`.
- Slower-than-vendor cases: `SO3inertia.hat` (0.69x), `SO3inertia.hat_commute` (0.72x).

## Comparison

| case | mathrobors us | mathrobo us | speedup x | loops |
| --- | ---: | ---: | ---: | ---: |
| SO3.hat | 0.366 | 1.196 | 3.27 | 200000 |
| SO3.vee | 0.605 | 0.778 | 1.29 | 200000 |
| SO3.exp | 0.384 | 3.770 | 9.83 | 100000 |
| SO3.apply | 0.287 | 1.199 | 4.18 | 200000 |
| SO3wrench.hat | 1.146 | 1.513 | 1.32 | 200000 |
| SO3wrench.exp | 0.966 | 3.931 | 4.07 | 100000 |
| SO3wrench.exp_integ | 0.972 | 4.059 | 4.18 | 100000 |
| SO3inertia.hat | 1.063 | 0.735 | 0.69 | 200000 |
| SO3inertia.hat_commute | 1.026 | 0.734 | 0.72 | 200000 |
| SE3.hat | 0.404 | 1.085 | 2.69 | 200000 |
| SE3.vee | 0.769 | 1.596 | 2.07 | 200000 |
| SE3.exp | 0.438 | 9.343 | 21.35 | 80000 |
| SE3.apply | 0.263 | 1.178 | 4.48 | 200000 |
| SE3.mat_adj | 0.172 | 2.864 | 16.65 | 120000 |
| SE3wrench.mat_adj | 2.241 | 2.840 | 1.27 | 120000 |
| SE3wrench.mat_inv_adj | 2.546 | 3.310 | 1.30 | 120000 |
| SE3wrench.hat_adj | 2.447 | 3.468 | 1.42 | 150000 |
| SE3wrench.hat_commute | 1.739 | 2.019 | 1.16 | 150000 |
| SE3wrench.hat_commute_adj | 3.462 | 5.126 | 1.48 | 150000 |
| SE3wrench.exp | 1.118 | 12.943 | 11.57 | 80000 |
| SE3wrench.exp_integ | 1.537 | 15.033 | 9.78 | 80000 |
| SE3wrench.mat_var_x_arb_vec | 7.326 | 9.314 | 1.27 | 80000 |

## Rust-only

| case | mathrobors us | std us | loops |
| --- | ---: | ---: | ---: |
| SO3.apply_into | 0.234 | 0.000 | 200000 |
| SE3.apply_into | 0.229 | 0.000 | 200000 |
| SE3.exp_into | 0.323 | 0.001 | 120000 |
| SE3.mat_adj_into | 0.095 | 0.000 | 120000 |
| SE3inertia.hat | 5.988 | 0.015 | 150000 |
| SE3inertia.hat_commute | 3.892 | 0.003 | 150000 |

## Skipped Vendor Cases

- `SE3inertia.hat`: vendored MathRobo currently raises RecursionError in the numpy path.
- `SE3inertia.hat_commute`: vendored MathRobo currently raises ValueError in the numpy path.
