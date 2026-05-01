import math

import mathrobors
import numpy as np


def approx_eq(a, b, tol=1e-12):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(x - y) < tol, f"expected {y}, got {x}"


def approx_eq_matrix(a, b, tol=1e-12):
    for r in range(3):
        for c in range(3):
            x = a[r][c]
            y = b[r][c]
            assert abs(x - y) < tol, f"expected {y}, got {x} at ({r},{c})"


def approx_eq_matrix4(a, b, tol=1e-12):
    for r in range(4):
        for c in range(4):
            x = a[r][c]
            y = b[r][c]
            assert abs(x - y) < tol, f"expected {y}, got {x} at ({r},{c})"


def test_python_interface_produces_same_values():
    transform = mathrobors.SE3.from_axis_angle_translation(
        (0.0, 0.0, 1.0), math.pi / 2.0, (0.5, -0.25, 1.25)
    )
    result = transform.apply((1.0, 0.0, 0.0))
    approx_eq(result, (0.5, 0.75, 1.25), 1e-12)


def test_se3_translation_and_rotation_combined():
    rotation = mathrobors.SO3.from_axis_angle((0.0, 1.0, 0.0), math.pi / 2.0)
    transform = mathrobors.SE3.from_parts(rotation, (1.0, 2.0, 3.0))

    applied = transform.apply((1.0, 0.0, 0.0))
    approx_eq(applied, (1.0, 2.0, 2.0), 1e-12)


def test_quaternion_roundtrip_preserves_matrix():
    rotation = mathrobors.SO3.from_axis_angle((0.0, 0.0, 1.0), math.pi / 2.0)
    quat = rotation.quaternion()
    rebuilt = mathrobors.SO3.set_quaternion(quat)

    approx_eq_matrix(rotation.matrix(), rebuilt.matrix(), 1e-12)


def test_hat_and_vee_functions_roundtrip_vector():
    vector = (0.1, -0.25, 0.75)
    hat = mathrobors.SO3.hat(vector)
    recovered = mathrobors.SO3.vee(hat)
    approx_eq(recovered, vector, 1e-12)


def test_se3_hat_and_vee_are_inverses():
    twist = (0.1, -0.2, 0.3, 1.0, -2.0, 3.0)
    hat = mathrobors.SE3.hat(twist)
    recovered = mathrobors.SE3.vee(hat)

    approx_eq(recovered, twist, 1e-12)


def test_se3_exp_with_pure_translation_matches_expected():
    twist = (0.0, 0.0, 0.0, 1.0, 2.0, 3.0)
    exp = mathrobors.SE3.exp(twist, None)

    expected = (
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0, 2.0),
        (0.0, 0.0, 1.0, 3.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    approx_eq_matrix4(exp, expected, 1e-12)


def test_se3_from_matrix_round_trip():
    rotation = mathrobors.SO3.from_axis_angle((0.0, 0.0, 1.0), math.pi / 2.0)
    transform = mathrobors.SE3.from_parts(rotation, (0.25, -0.5, 0.75))
    matrix = transform.matrix()
    rebuilt = mathrobors.SE3.from_matrix(matrix)

    approx_eq_matrix4(matrix, rebuilt.matrix(), 1e-12)
    approx_eq_matrix(rotation.matrix(), rebuilt.rotation().matrix(), 1e-12)
    approx_eq(transform.translation(), rebuilt.translation(), 1e-12)


def test_python_functions_match_original_names():
    rotation = mathrobors.SO3.set_euler((0.1, -0.2, 0.3))
    approx_eq_matrix(rotation.mat(), rotation.matrix(), 1e-12)
    approx_eq_matrix(rotation.mat_inv(), rotation.inverse().matrix(), 1e-12)
    approx_eq_matrix(rotation.mat_adj(), rotation.matrix(), 1e-12)
    approx_eq_matrix(rotation.mat_inv_adj(), rotation.inverse().matrix(), 1e-12)

    quat = rotation.quaternion()
    approx_eq_matrix(mathrobors.SO3.quaternion_to_mat(quat), rotation.matrix(), 1e-12)
    approx_eq(mathrobors.SO3.mat_to_quaternion(rotation.matrix()), quat, 1e-12)

    identity = mathrobors.SO3.eye()
    approx_eq_matrix(identity.mat(), mathrobors.SO3.set_mat(identity.matrix()).matrix(), 1e-12)

    hat = mathrobors.SO3.hat((0.2, 0.3, 0.4))
    commute = mathrobors.SO3.hat_commute((0.2, 0.3, 0.4))
    for r in range(3):
        for c in range(3):
            assert hat[r][c] == -commute[r][c]

    vee_adj = mathrobors.SO3.vee_adj(hat)
    approx_eq(vee_adj, (0.2, 0.3, 0.4), 1e-12)

    exp_matrix = mathrobors.SO3.exp((0.1, -0.2, 0.3), None)
    approx_eq_matrix(exp_matrix, mathrobors.SO3.from_rotation_vector((0.1, -0.2, 0.3)).matrix(), 1e-12)


def test_public_api_returns_numpy_arrays():
    rotation = mathrobors.SO3.from_axis_angle((0.0, 0.0, 1.0), math.pi / 4.0)
    transform = mathrobors.SE3.from_parts(rotation, (0.25, -0.5, 0.75))

    assert isinstance(rotation.mat(), np.ndarray)
    assert isinstance(rotation.quaternion(), np.ndarray)
    assert isinstance(rotation.rotation_vector(), np.ndarray)
    assert isinstance(rotation @ np.array([1.0, 0.0, 0.0]), np.ndarray)

    assert isinstance(transform.mat(), np.ndarray)
    assert isinstance(transform.pos(), np.ndarray)
    assert isinstance(transform.rot(), np.ndarray)
    assert isinstance(transform @ np.array([1.0, 0.0, 0.0]), np.ndarray)
    assert isinstance(transform @ np.array([0.1, -0.2, 0.3, 1.0, -2.0, 3.0]), np.ndarray)


def test_wrench_variants_follow_expected_adjoint_relations():
    rotation = mathrobors.SO3.from_axis_angle((0.0, 0.0, 1.0), math.pi / 3.0)
    transform = mathrobors.SE3.from_parts(rotation, (0.25, -0.5, 0.75))
    wrench = mathrobors.SE3wrench(transform.rot(), transform.pos())
    expected_hat = [[-value for value in row] for row in mathrobors.SO3.hat((0.2, 0.3, 0.4))]

    approx_eq_matrix(mathrobors.SO3wrench.hat((0.2, 0.3, 0.4)), expected_hat, 1e-12)
    approx_eq_matrix4(wrench.mat(), transform.mat(), 1e-12)
    approx_eq_matrix4(wrench.mat_inv(), transform.mat_inv(), 1e-12)
    expected_adj = list(map(list, zip(*transform.mat_inv_adj())))
    for row in range(6):
        for col in range(6):
            assert abs(wrench.mat_adj()[row][col] - expected_adj[row][col]) < 1e-12


def test_se3wrench_operator_contract():
    left = mathrobors.SE3wrench(
        mathrobors.SO3.exp((0.1, -0.2, 0.15), None), (0.25, -0.4, 0.6)
    )
    right = mathrobors.SE3wrench(
        mathrobors.SO3.exp((-0.05, 0.1, 0.2), None), (-0.2, 0.3, 0.1)
    )
    wrench_vec = np.array([0.2, -0.1, 0.3, 1.0, -0.5, 0.25], dtype=float)

    composed = left @ right
    assert isinstance(composed, mathrobors.SE3wrench)
    assert isinstance(left @ wrench_vec, np.ndarray)
    assert isinstance(left @ np.eye(6, dtype=float), np.ndarray)
    approx_eq_matrix4(composed.mat(), left.mat() @ right.mat(), 1e-12)
    approx_eq(left @ wrench_vec, left.mat_adj() @ wrench_vec, 1e-12)


def test_inertia_variants_produce_expected_shapes():
    so3_hat = mathrobors.SO3inertia.hat((1.0, 2.0, 3.0, 0.4, 0.5, 0.6))
    se3_hat = mathrobors.SE3inertia.hat((2.0, 0.25, -0.5, 0.75, 1.0, 2.0, 3.0, 0.4, 0.5, 0.6))
    se3_commute = mathrobors.SE3inertia.hat_commute((0.1, -0.2, 0.3, 1.0, -2.0, 3.0))

    assert len(so3_hat) == 3 and len(so3_hat[0]) == 3
    assert len(se3_hat) == 6 and len(se3_hat[0]) == 6
    assert len(se3_commute) == 6 and len(se3_commute[0]) == 10


def test_se3inertia_block_formulas_are_consistent():
    vec10 = (2.0, 0.25, -0.5, 0.75, 1.0, 2.0, 3.0, 0.4, 0.5, 0.6)
    twist = (0.1, -0.2, 0.3, 0.4, -0.5, 0.6)

    hat = mathrobors.SE3inertia.hat(vec10)
    expected_hat = [[0.0] * 6 for _ in range(6)]
    so3_inertia = mathrobors.SO3inertia.hat(vec10[4:10])
    so3_wrench = mathrobors.SO3wrench.hat(vec10[1:4])
    so3_hat = mathrobors.SO3.hat(vec10[1:4])
    for r in range(3):
        for c in range(3):
            expected_hat[r][c] = so3_inertia[r][c]
            expected_hat[r][c + 3] = so3_wrench[r][c]
            expected_hat[r + 3][c] = so3_hat[r][c]
            expected_hat[r + 3][c + 3] = vec10[0] if r == c else 0.0
    approx_eq_matrix4([row[:4] for row in hat[:4]], [row[:4] for row in expected_hat[:4]], 1e-12)
    for r in range(6):
        approx_eq(hat[r], expected_hat[r], 1e-12)

    commute = mathrobors.SE3inertia.hat_commute(twist)
    expected_commute = [[0.0] * 10 for _ in range(6)]
    so3_wrench_commute = mathrobors.SO3wrench.hat_commute(twist[3:6])
    so3_commute = mathrobors.SO3.hat_commute(twist[0:3])
    so3_inertia_commute = mathrobors.SO3inertia.hat_commute(twist[0:3])
    for r in range(3):
        for c in range(3):
            expected_commute[r][c + 1] = so3_wrench_commute[r][c]
            expected_commute[r + 3][c + 1] = so3_commute[r][c]
        for c in range(6):
            expected_commute[r][c + 4] = so3_inertia_commute[r][c]
    for r in range(3):
        expected_commute[r + 3][0] = twist[r + 3]
    for r in range(6):
        approx_eq(commute[r], expected_commute[r], 1e-12)
