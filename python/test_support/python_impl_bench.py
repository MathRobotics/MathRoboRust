from __future__ import annotations

import array
import math
from dataclasses import dataclass
from typing import Callable

from .mathrobo_vendor import import_mathrobo


@dataclass(frozen=True)
class ComparisonCase:
    name: str
    loops: int
    rust_fn: Callable[[], object]
    vendor_fn: Callable[[], object]


@dataclass(frozen=True)
class RustOnlyCase:
    name: str
    loops: int
    rust_fn: Callable[[], object]


COMPARISON_CASE_NAMES = (
    "SO3.hat",
    "SO3.vee",
    "SO3.exp",
    "SO3.apply",
    "SO3wrench.hat",
    "SO3wrench.exp",
    "SO3wrench.exp_integ",
    "SO3inertia.hat",
    "SO3inertia.hat_commute",
    "SE3.hat",
    "SE3.vee",
    "SE3.exp",
    "SE3.apply",
    "SE3.mat_adj",
    "SE3wrench.mat_adj",
    "SE3wrench.mat_inv_adj",
    "SE3wrench.hat_adj",
    "SE3wrench.hat_commute",
    "SE3wrench.hat_commute_adj",
    "SE3wrench.exp",
    "SE3wrench.exp_integ",
    "SE3wrench.mat_var_x_arb_vec",
)

RUST_ONLY_CASE_NAMES = (
    "SO3.apply_into",
    "SE3.apply_into",
    "SE3.exp_into",
    "SE3.mat_adj_into",
    "SE3inertia.hat",
    "SE3inertia.hat_commute",
)

SKIPPED_COMPARISON_CASES = (
    (
        "SE3inertia.hat",
        "vendored MathRobo currently raises RecursionError in the numpy path",
    ),
    (
        "SE3inertia.hat_commute",
        "vendored MathRobo currently raises ValueError in the numpy path",
    ),
)


def comparison_cases() -> list[ComparisonCase]:
    import mathrobors
    import numpy as np

    mathrobo = import_mathrobo()

    axis_tuple = (0.0, 0.0, 1.0)
    axis_np = np.array(axis_tuple, dtype=float)
    angle = math.pi / 2.0
    translation_tuple = (0.25, -0.5, 1.0)
    translation_np = np.array(translation_tuple, dtype=float)

    w_tuple = (0.2, 0.3, 0.4)
    w_np = np.array(w_tuple, dtype=float)
    rotvec_tuple = (0.1, -0.2, 0.3)
    rotvec_np = np.array(rotvec_tuple, dtype=float)
    twist_tuple = (0.1, -0.2, 0.3, 1.0, -2.0, 3.0)
    twist_np = np.array(twist_tuple, dtype=float)
    inertia_tuple = (1.0, 2.0, 3.0, 0.4, 0.5, 0.6)
    inertia_np = np.array(inertia_tuple, dtype=float)
    point_tuple = (1.0, 0.0, 0.0)
    point_np = np.array(point_tuple, dtype=float)
    se3_rotvec_tuple = (0.1, -0.2, 0.15)
    se3_rotvec_np = np.array(se3_rotvec_tuple, dtype=float)
    wrench_arb_tuple = (0.4, -0.3, 0.2, 0.1, -0.5, 0.6)
    wrench_arb_np = np.array(wrench_arb_tuple, dtype=float)
    wrench_tan_tuple = (0.05, -0.02, 0.03, 0.2, -0.1, 0.04)
    wrench_tan_np = np.array(wrench_tan_tuple, dtype=float)

    rust_so3 = mathrobors.SO3.from_axis_angle(axis_tuple, angle)
    vendor_so3 = mathrobo.SO3.set_mat(mathrobo.SO3.exp(axis_np, angle))
    rust_se3 = mathrobors.SE3.from_axis_angle_translation(axis_tuple, angle, translation_tuple)
    vendor_se3 = mathrobo.SE3(mathrobo.SO3.exp(axis_np, angle), translation_np)
    rust_se3wrench = mathrobors.SE3wrench(mathrobors.SO3.exp(se3_rotvec_tuple, None), translation_tuple)
    vendor_se3wrench = mathrobo.SE3wrench(mathrobo.SO3.exp(se3_rotvec_np), translation_np)

    rust_so3_hat = mathrobors.SO3.hat(w_tuple)
    vendor_so3_hat = mathrobo.SO3.hat(w_np)
    rust_se3_hat = mathrobors.SE3.hat(twist_tuple)
    vendor_se3_hat = mathrobo.SE3.hat(twist_np)

    return [
        ComparisonCase(
            name="SO3.hat",
            loops=200_000,
            rust_fn=lambda: mathrobors.SO3.hat(w_tuple),
            vendor_fn=lambda: mathrobo.SO3.hat(w_np),
        ),
        ComparisonCase(
            name="SO3.vee",
            loops=200_000,
            rust_fn=lambda: mathrobors.SO3.vee(rust_so3_hat),
            vendor_fn=lambda: mathrobo.SO3.vee(vendor_so3_hat),
        ),
        ComparisonCase(
            name="SO3.exp",
            loops=100_000,
            rust_fn=lambda: mathrobors.SO3.exp(rotvec_tuple, None),
            vendor_fn=lambda: mathrobo.SO3.exp(rotvec_np),
        ),
        ComparisonCase(
            name="SO3.apply",
            loops=200_000,
            rust_fn=lambda: rust_so3.apply(point_tuple),
            vendor_fn=lambda: vendor_so3 @ point_np,
        ),
        ComparisonCase(
            name="SO3wrench.hat",
            loops=200_000,
            rust_fn=lambda: mathrobors.SO3wrench.hat(w_tuple),
            vendor_fn=lambda: mathrobo.SO3wrench.hat(w_np),
        ),
        ComparisonCase(
            name="SO3wrench.exp",
            loops=100_000,
            rust_fn=lambda: mathrobors.SO3wrench.exp(rotvec_tuple, 0.7),
            vendor_fn=lambda: mathrobo.SO3wrench.exp(rotvec_np, 0.7),
        ),
        ComparisonCase(
            name="SO3wrench.exp_integ",
            loops=100_000,
            rust_fn=lambda: mathrobors.SO3wrench.exp_integ(rotvec_tuple, 0.7),
            vendor_fn=lambda: mathrobo.SO3wrench.exp_integ(rotvec_np, 0.7),
        ),
        ComparisonCase(
            name="SO3inertia.hat",
            loops=200_000,
            rust_fn=lambda: mathrobors.SO3inertia.hat(inertia_tuple),
            vendor_fn=lambda: mathrobo.SO3inertia.hat(inertia_np),
        ),
        ComparisonCase(
            name="SO3inertia.hat_commute",
            loops=200_000,
            rust_fn=lambda: mathrobors.SO3inertia.hat_commute(w_tuple),
            vendor_fn=lambda: mathrobo.SO3inertia.hat_commute(w_np),
        ),
        ComparisonCase(
            name="SE3.hat",
            loops=200_000,
            rust_fn=lambda: mathrobors.SE3.hat(twist_tuple),
            vendor_fn=lambda: mathrobo.SE3.hat(twist_np),
        ),
        ComparisonCase(
            name="SE3.vee",
            loops=200_000,
            rust_fn=lambda: mathrobors.SE3.vee(rust_se3_hat),
            vendor_fn=lambda: mathrobo.SE3.vee(vendor_se3_hat),
        ),
        ComparisonCase(
            name="SE3.exp",
            loops=80_000,
            rust_fn=lambda: mathrobors.SE3.exp(twist_tuple, None),
            vendor_fn=lambda: mathrobo.SE3.exp(twist_np),
        ),
        ComparisonCase(
            name="SE3.apply",
            loops=200_000,
            rust_fn=lambda: rust_se3.apply(point_tuple),
            vendor_fn=lambda: vendor_se3 @ point_np,
        ),
        ComparisonCase(
            name="SE3.mat_adj",
            loops=120_000,
            rust_fn=rust_se3.mat_adj,
            vendor_fn=vendor_se3.mat_adj,
        ),
        ComparisonCase(
            name="SE3wrench.mat_adj",
            loops=120_000,
            rust_fn=rust_se3wrench.mat_adj,
            vendor_fn=vendor_se3wrench.mat_adj,
        ),
        ComparisonCase(
            name="SE3wrench.mat_inv_adj",
            loops=120_000,
            rust_fn=rust_se3wrench.mat_inv_adj,
            vendor_fn=vendor_se3wrench.mat_inv_adj,
        ),
        ComparisonCase(
            name="SE3wrench.hat_adj",
            loops=150_000,
            rust_fn=lambda: mathrobors.SE3wrench.hat_adj(twist_tuple),
            vendor_fn=lambda: mathrobo.SE3wrench.hat_adj(twist_np),
        ),
        ComparisonCase(
            name="SE3wrench.hat_commute",
            loops=150_000,
            rust_fn=lambda: mathrobors.SE3wrench.hat_commute(twist_tuple),
            vendor_fn=lambda: mathrobo.SE3wrench.hat_commute(twist_np),
        ),
        ComparisonCase(
            name="SE3wrench.hat_commute_adj",
            loops=150_000,
            rust_fn=lambda: mathrobors.SE3wrench.hat_commute_adj(twist_tuple),
            vendor_fn=lambda: mathrobo.SE3wrench.hat_commute_adj(twist_np),
        ),
        ComparisonCase(
            name="SE3wrench.exp",
            loops=80_000,
            rust_fn=lambda: mathrobors.SE3wrench.exp(twist_tuple, 0.5),
            vendor_fn=lambda: mathrobo.SE3wrench.exp(twist_np, 0.5),
        ),
        ComparisonCase(
            name="SE3wrench.exp_integ",
            loops=80_000,
            rust_fn=lambda: mathrobors.SE3wrench.exp_integ(twist_tuple, 0.5),
            vendor_fn=lambda: mathrobo.SE3wrench.exp_integ(twist_np, 0.5),
        ),
        ComparisonCase(
            name="SE3wrench.mat_var_x_arb_vec",
            loops=80_000,
            rust_fn=lambda: rust_se3wrench.mat_var_x_arb_vec(wrench_arb_tuple, wrench_tan_tuple),
            vendor_fn=lambda: vendor_se3wrench.mat_var_x_arb_vec(wrench_arb_np, wrench_tan_np),
        ),
    ]


def rust_only_cases() -> list[RustOnlyCase]:
    import mathrobors

    axis = (0.0, 0.0, 1.0)
    angle = math.pi / 2.0
    point = (1.0, 0.0, 0.0)
    twist = (0.1, -0.2, 0.3, 1.0, -2.0, 3.0)
    inertia = (1.0, 2.0, 3.0, 0.4, 0.5, 0.6)
    wrench_twist = (0.1, -0.2, 0.3, 0.4, -0.5, 0.6)

    so3 = mathrobors.SO3.from_axis_angle(axis, angle)
    se3 = mathrobors.SE3.from_axis_angle_translation(axis, angle, (0.25, -0.5, 1.0))

    so3_out = array.array("d", [0.0, 0.0, 0.0])
    se3_out = array.array("d", [0.0, 0.0, 0.0])
    exp_out = array.array("d", [0.0] * 16)
    adj_out = array.array("d", [0.0] * 36)

    return [
        RustOnlyCase(
            name="SO3.apply_into",
            loops=200_000,
            rust_fn=lambda: so3.apply_into(point, so3_out),
        ),
        RustOnlyCase(
            name="SE3.apply_into",
            loops=200_000,
            rust_fn=lambda: se3.apply_into(point, se3_out),
        ),
        RustOnlyCase(
            name="SE3.exp_into",
            loops=120_000,
            rust_fn=lambda: mathrobors.SE3.exp_into(twist, exp_out, None),
        ),
        RustOnlyCase(
            name="SE3.mat_adj_into",
            loops=120_000,
            rust_fn=lambda: se3.mat_adj_into(adj_out),
        ),
        RustOnlyCase(
            name="SE3inertia.hat",
            loops=150_000,
            rust_fn=lambda: mathrobors.SE3inertia.hat((2.0, 0.25, -0.5, 0.75, *inertia)),
        ),
        RustOnlyCase(
            name="SE3inertia.hat_commute",
            loops=150_000,
            rust_fn=lambda: mathrobors.SE3inertia.hat_commute(wrench_twist),
        ),
    ]


def comparison_case(name: str) -> ComparisonCase:
    return {case.name: case for case in comparison_cases()}[name]


def rust_only_case(name: str) -> RustOnlyCase:
    return {case.name: case for case in rust_only_cases()}[name]
