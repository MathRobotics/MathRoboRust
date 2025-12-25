import math
import pytest
import mathrobors


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


def test_python_interface_produces_same_values():
    transform = mathrobors.SE3.from_axis_angle_translation(
        (0.0, 0.0, 1.0), math.pi / 2.0, (0.5, -0.25, 1.25)
    )
    result = transform.apply((1.0, 0.0, 0.0))
    approx_eq(result, (0.5, 0.75, 1.25), 1e-12)


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

@pytest.mark.dev
def test_compare_mathrobo():
    pass
