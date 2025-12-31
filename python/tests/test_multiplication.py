import math
import mathrobors


def approx_eq(iter_a, iter_b, tol=1e-12):
    for x, y in zip(iter_a, iter_b):
        assert abs(x - y) < tol, f"expected {y}, got {x}"


def approx_eq_matrix3(a, b, tol=1e-12):
    for r in range(3):
        approx_eq(a[r], b[r], tol)


def approx_eq_matrix4(a, b, tol=1e-12):
    for r in range(4):
        approx_eq(a[r], b[r], tol)


def approx_eq_matrix6(a, b, tol=1e-12):
    for r in range(6):
        approx_eq(a[r], b[r], tol)


def test_so3_mul_matches_compose_and_apply():
    left = mathrobors.SO3.from_axis_angle((0.0, 0.0, 1.0), math.pi / 3.0)
    right = mathrobors.SO3.from_axis_angle((0.0, 1.0, 0.0), -math.pi / 6.0)

    via_mul = left * right
    via_compose = left.compose(right)

    approx_eq_matrix3(via_mul.matrix(), via_compose.matrix(), 1e-12)
    approx_eq(
        via_mul.apply((0.5, -0.25, 1.0)),
        via_compose.apply((0.5, -0.25, 1.0)),
        1e-12,
    )


def test_se3_mul_matches_compose_and_apply():
    rotation_a = mathrobors.SO3.from_axis_angle((0.0, 0.0, 1.0), math.pi / 4.0)
    rotation_b = mathrobors.SO3.from_axis_angle((0.0, 1.0, 0.0), math.pi / 3.0)

    left = mathrobors.SE3.from_parts(rotation_a, (0.25, -0.5, 0.75))
    right = mathrobors.SE3.from_parts(rotation_b, (1.0, 0.5, -0.25))

    via_mul = left * right
    via_compose = left.compose(right)

    approx_eq_matrix4(via_mul.matrix(), via_compose.matrix(), 1e-12)
    approx_eq(
        via_mul.apply((0.25, 0.5, -1.0)),
        via_compose.apply((0.25, 0.5, -1.0)),
        1e-12,
    )


def test_cmtm_mul_matches_compose_matrix():
    base_a = mathrobors.SE3.from_axis_angle_translation(
        (0.0, 0.0, 1.0), math.pi / 6.0, (0.0, 0.5, -0.25)
    )
    base_b = mathrobors.SE3.from_axis_angle_translation(
        (1.0, 0.0, 0.0), -math.pi / 4.0, (1.0, -0.25, 0.75)
    )

    left = mathrobors.CMTM.from_se3(base_a)
    right = mathrobors.CMTM.from_se3(base_b)

    via_mul = left * right
    via_compose = left.compose(right)

    approx_eq_matrix6(via_mul.matrix(), via_compose.matrix(), 1e-12)
