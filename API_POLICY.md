# Python API Policy Draft

## Purpose

This document defines the working policy for the Python-facing API of `mathrobors`.
The goal is to avoid ad-hoc compatibility work and to keep future implementation work aligned with a single public contract.

## Primary Goal

Build a clean, fast, numpy-first Python API on top of the Rust core.

In practice, this means:

- Users should be able to perform the same meaningful computations as `mathrobo`.
- The public API should be easier to debug and maintain than the current mixed PyO3/Python surface.
- Performance-sensitive computation should remain in Rust.

## Non-Goals

- Full reproduction of `mathrobo` internal class inheritance.
- Full reproduction of `mathrobo`'s `LIB='jax'` behavior.
- Reproduction of vendor bugs or undefined behavior.
- Forcing every low-level Rust optimization detail into the public Python API.

## Compatibility Boundary

Compatibility is defined primarily at the level of:

- mathematical meaning
- argument meaning
- frame convention
- returned shape
- returned dtype family
- major public method/operator names

Compatibility is not defined primarily at the level of:

- exact internal inheritance structure
- exact private implementation layout
- exact error text
- vendor-specific broken behavior

The target is practical numpy compatibility, not byte-for-byte source compatibility with every `mathrobo` script.

## Layering Policy

The Python side should be explicitly layered.

### Public Layer

`mathrobors`

- stable user-facing API
- numpy-first behavior
- normalized inputs and outputs
- compatibility-oriented naming
- documented operator behavior

### Native Layer

Private low-level bindings, conceptually:

- `mathrobors._native`
- direct Rust/PyO3 exposure
- buffer-oriented methods
- implementation details
- allowed to be less ergonomic

The public layer may call the native layer, but users should not need to depend on native-layer details.

## Python Shim Policy

The public API should be unified by a Python shim layer.

This does not mean reimplementing heavy math in Python.
It means the shim is responsible for:

- input normalization from `list` / `tuple` / `numpy.ndarray`
- output normalization to `numpy.ndarray` where appropriate
- stable public naming
- compatibility adaptation against `mathrobo`
- error normalization
- absorbing differences between Rust internals and public Python expectations

Heavy numerical work should remain in Rust whenever practical.

## Class Policy

Classes should exist only where they carry clear domain meaning.

Keep classes for:

- `SO3`
- `SE3`
- `CMTM[SO3]`
- `CMTM[SE3]`
- `SO3wrench`
- `SE3wrench`
- `SO3inertia`
- `SE3inertia`
- `CMVector`

Class existence is justified by one or more of:

- a persistent transform/operator state
- natural method grouping
- natural operator overloading
- readability of chained operations

Internal inheritance does not need to match `mathrobo`.
Public behavior matters more than class ancestry.

## Function vs Method Policy

Use methods when:

- the operation naturally depends on object state
- the operation reads clearly as an action of the object
- the operation participates in a stable operator contract

Use module/class-level functions or static methods when:

- the operation is a pure constructor or pure algebraic helper
- the operation does not require stored object state
- the operation is easier to test and document without object coupling

## Operator Policy

Operator overloading is allowed only when the meaning is single-valued and easy to explain.

### Keep `@` for

- `SO3`
  - composition with `SO3`
  - action on 3D vectors
  - action on 3x3 matrices
- `SE3`
  - composition with `SE3`
  - action on 3D points
  - action on twists when documented
  - action on 4x4 and 6x6 matrices when documented
- `CMTM`
  - composition with same-order `CMTM`
  - action on `CMVector`
  - action on compatible matrices
- `SE3wrench`
  - future composition with `SE3wrench`
  - future action on 6D wrench vectors
  - future action on compatible 6x6 adjoint-side matrices

### Avoid `@` for

- ambiguous helper computations
- operations whose meaning depends on hidden conventions
- APIs that are easier to understand through explicit method names

If an overloaded operator cannot be described in one sentence, it should probably be a named method instead.

## Return-Type Policy

Public Python API outputs should be normalized as follows:

- scalars: Python `float` / `int`
- vectors and matrices: `numpy.ndarray`
- transform/operator objects: domain classes
- batch-like variation structures: `numpy.ndarray` or `CMVector`, depending on public contract

The public layer should avoid exposing raw Python `list` results for numerical arrays.

## Input Policy

Public APIs should accept:

- `list`
- `tuple`
- `numpy.ndarray`

The shim should normalize inputs early and validate shape explicitly.

## Public API Surface

The stable Python surface is defined by the following public contracts.

| class | constructor / factory surface | standard ndarray outputs | stable `@` operands |
| --- | --- | --- | --- |
| `SO3` | `SO3()`, `set_mat`, `set_mat_adj`, `eye`, `rand`, `from_axis_angle`, `from_quaternion`, `set_quaternion`, `from_euler_angles`, `set_euler`, `from_rotation_vector` | `mat`, `matrix`, `mat_inv`, `mat_adj`, `mat_inv_adj`, `quaternion`, `rotation_vector`, `apply`, `hat`, `vee`, `exp*` | `SO3`, 3D vector, 3x3 matrix |
| `SE3` | `SE3()`, `set_mat`, `set_mat_adj`, `eye`, `rand`, `from_axis_angle_translation`, `from_parts`, `from_matrix`, `set_pos_quaternion` | `mat`, `matrix`, `mat_inv`, `mat_adj`, `mat_inv_adj`, `translation`, `position`, `pos`, `rot`, `apply`, `hat`, `vee`, `exp*` | `SE3`, 3D point, 6D twist/wrench, 4x4 matrix, 6x6 matrix |
| `SO3wrench` | `SO3wrench()`, `set_mat` | `mat`, `hat`, `hat_commute`, `exp`, `exp_integ` | none |
| `SE3wrench` | `SE3wrench()`, `set_mat`, `change_class` | `mat`, `mat_inv`, `mat_adj`, `mat_inv_adj`, `hat_adj`, `hat_commute`, `hat_commute_adj`, `exp`, `exp_integ`, `mat_var_x_arb_vec*` | `SE3wrench`, 6D wrench vector, 4x4 matrix, 6x6 matrix |
| `SO3inertia` | `SO3inertia` helper/statics | `hat`, `hat_commute` | none |
| `SE3inertia` | `SE3inertia` helper/statics | `hat`, `hat_commute` | none |
| `CMTM[SO3]`, `CMTM[SE3]`, `SO3CMTM`, `SE3CMTM` | `eye`, `rand`, `set_mat`, `set_mat_adj`, constructor from element + derivative rows | `mat`, `mat_adj`, `tangent_mat`, `tangent_mat_adj`, `hat*`, `vee*`, `mat_var_x_arb_vec*` | same-type `CMTM`, `CMVector`, compatible block matrices |
| `CMVector` | `CMVector`, `set_cmvecs` | `vecs`, `cm_vecs`, `vec`, `cm_vec` | none |

## Performance Policy

The system should remain Rust-first for hot numerical paths.

Completion checks and benchmark reports must use a release build installed with:

- `cd python`
- `uv run maturin develop --release`

### Good Python-shim work

- type normalization
- shape validation
- small compatibility adapters
- public API rearrangement

### Bad Python-shim work

- large hot loops
- repeated matrix construction in critical paths without reason
- reimplementation of core Lie algebra logic already present in Rust

Optimization should happen below the public API whenever possible.

## Accepted Slow Paths

The following paths are currently accepted as slower than vendored `mathrobo`:

- `SO3inertia.hat`
- `SO3inertia.hat_commute`

These are documented exceptions, not hidden regressions. All other benchmarked public paths should remain meaningfully faster than the vendored numpy implementation in release builds.

## Deliberate Behavior Differences

The following public behaviors intentionally differ from vendored `mathrobo`:

- `SE3wrench @ <6D vector>` uses the wrench-side adjoint action (`self.mat_adj() @ vector`).
  - Vendored `mathrobo` inherits `SE3.__matmul__` here and therefore applies the twist-style action instead.
  - `mathrobors` treats the wrench class as its own public contract and keeps the operator aligned with `SE3wrench.mat_adj()`.

## Testing Policy

Testing should be split into separate contracts.

### Public API Tests

- return types
- accepted inputs
- operator semantics
- documented compatibility behavior

### Numerical Parity Tests

- comparison with vendored `MathRobo` where the vendor path is valid
- documented exceptions where the vendor implementation is broken

### Performance Tests

- Python-side benchmark against vendored `MathRobo`
- regression-oriented benchmark against previous `mathrobors` behavior

## Documentation Policy

Every stable public behavior should be documented in one place.

At minimum, documentation should define:

- constructor forms
- return types
- supported operands for `@`
- frame conventions
- known deliberate differences from `mathrobo`

## Working Default Decisions

Until explicitly revised, the project should follow these defaults:

- numpy-only public API
- Python shim as the stable public layer
- Rust bindings as the private native layer
- compatibility at the behavior level, not inheritance level
- `numpy.ndarray` as the standard numeric return type
- no compatibility commitment to vendor bugs

## Completion Condition

This work is complete only when the Python API is both clean and fast.

For this project, "clean and fast" means:

- the public API boundary is consistent
- operator behavior is documented and unsurprising
- return types are normalized
- parity expectations are explicit
- tests are stable
- benchmark results remain meaningfully better than the reference where expected
- known slower paths are identified and intentionally accepted or fixed

If this condition is not met, the work is not complete.

## Iteration Notes

- Iteration 1:
  - Added a private native boundary in the Python package through `mathrobors._native`.
  - Moved `SO3` and `SE3` public behavior behind the Python shim layer.
  - Normalized the public `SO3` / `SE3` array-like returns to `numpy.ndarray`.
  - Verified the refactor with Python tests and vendor-parity tests.
  - Re-ran the benchmark and found that the shim-layer normalization introduces too much overhead in hot paths such as `SO3.hat`, `SE3.hat`, and `SO3.apply`.
  - Conclusion: public-shim unification is still the right direction, but ndarray materialization for hot paths must move closer to the native layer.
- Iteration 2:
  - Prototyped `PyBuffer`-based `*_into` returns for more hot paths.
  - Verified that this keeps behavior correct, but it regresses the public benchmark because `np.empty(...)` allocation and buffer acquisition dominate small fixed-size APIs.
  - Conclusion: keep `*_into` for explicit fast-buffer APIs, but do not use it as the default public ndarray materialization strategy.
- Iteration 3:
  - Added direct native ndarray-return helpers for hot public paths in the private PyO3 layer.
  - Switched release benchmarking to be the canonical completion check.
  - Removed dead duplicate `SO3` / `SE3` shim definitions so only one public boundary remains in `python/mathrobors/__init__.py`.
  - Implemented and tested the stable `SE3wrench` operator contract.
  - Updated the saved benchmark report. In release mode the public API median speedup is `2.38x`, with `SE3.exp` reaching `21.35x`, and only the accepted `SO3inertia` paths remaining slower than vendor.

## Task List

- [x] Freeze the public compatibility boundary for `SO3`, `SE3`, `CMTM`, `SO3wrench`, `SE3wrench`, `SO3inertia`, and `SE3inertia`.
- [x] Define a concrete public API table: constructor forms, public methods, return types, and supported `@` operands for each public class.
- [x] Introduce or formalize a private native layer boundary so the Rust/PyO3 surface stops acting as the accidental public API.
- [x] Normalize public numeric return types to `numpy.ndarray` across the API.
- [x] Move hot-path ndarray creation closer to the native layer so public-shim normalization does not erase performance gains.
- [x] Normalize public input handling so `list`, `tuple`, and `numpy.ndarray` are accepted consistently.
- [x] Finalize the operator policy for `SO3`, `SE3`, `CMTM`, and `SE3wrench`, including unsupported operands.
- [x] Implement `SE3wrench` public semantics, including its future `@` contract.
- [x] Decide which helper routines remain methods and which should become pure helper functions or static methods.
- [x] Separate public API tests from low-level native binding tests.
- [x] Expand parity tests so every supported public behavior is either checked against `MathRobo` or explicitly marked as a deliberate difference.
- [x] Expand benchmark coverage so the stable public API, not just internal helpers, is measured against the vendored reference.
- [x] Document all deliberate divergences from `MathRobo`, especially vendor-broken paths.
- [x] Refactor examples and benchmark scripts so they exercise the intended public API rather than accidental internals.
- [x] Review the full task list against the current codebase state.
- [x] Re-list and re-prioritize remaining tasks after the review.
- [x] Check whether a clean and fast program has been achieved.
- [x] If the answer is no, return to the task-list step and produce a revised task list before continuing implementation.
