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
    import mathrobo as mr
    import numpy as np

    rotation = mathrobors.SO3.from_axis_angle((0.0, 0.0, 1.0), math.pi / 2.0)
    rotation_mr = mr.SO3.set_mat(mr.SO3.exp((0.0, 0.0, 1.0), math.pi / 2.0))

    assert rotation.matrix() == rotation_mr.mat().tolist()
    assert rotation.quaternion() == rotation_mr.quaternion().tolist()

    eye = mathrobors.SO3.eye()
    eye_mr = mr.SO3.eye()
    assert eye.mat() == eye_mr.mat().tolist()

    hat = mathrobors.SO3.hat((0.2, 0.3, 0.4))
    hat_mr = mr.SO3.hat(np.array([0.2, 0.3, 0.4]))
    assert hat == hat_mr.tolist()

    vee = mathrobors.SO3.vee(hat)
    vee_mr = mr.SO3.vee(hat_mr)
    assert vee == vee_mr.tolist()

    commute = mathrobors.SO3.hat_commute((0.2, 0.3, 0.4))
    commute_mr = mr.SO3.hat_commute(np.array([0.2, 0.3, 0.4]))
    assert commute == commute_mr.tolist()

    exp = mathrobors.SO3.exp((0.1, -0.2, 0.3), None)
    exp_mr = mr.SO3.exp(np.array([0.1, -0.2, 0.3]))
    assert exp == exp_mr.tolist()

@pytest.mark.dev
@pytest.mark.parametrize("impl", ["mathrobors", "mathrobo"])
def test_so3_exp_benchmark(benchmark, impl):
    import numpy as np
    import mathrobo as mr

    v = (0.1, -0.2, 0.3)
    v_np = np.array(v)

    if impl == "mathrobors":
        fn = lambda: mathrobors.SO3.exp(v, None)
    else:
        fn = lambda: mr.SO3.exp(v_np)

    benchmark(fn)

@pytest.mark.dev
@pytest.mark.parametrize("impl", ["mathrobors", "mathrobo"])
def test_so3_hat_benchmark(benchmark, impl):
    import numpy as np
    import mathrobo as mr

    w = (0.2, 0.3, 0.4)
    w_np = np.array(w)

    if impl == "mathrobors":
        fn = lambda: mathrobors.SO3.hat(w)
    else:
        fn = lambda: mr.SO3.hat(w_np)

    benchmark(fn)

@pytest.mark.dev
@pytest.mark.parametrize("impl", ["mathrobors", "mathrobo"])
def test_so3_vee_benchmark(benchmark, impl):
    import numpy as np
    import mathrobo as mr

    w = (0.2, 0.3, 0.4)
    w_np = np.array(w)

    if impl == "mathrobors":
        hat = mathrobors.SO3.hat(w)
        fn = lambda: mathrobors.SO3.vee(hat)
    else:
        hat = mr.SO3.hat(w_np)
        fn = lambda: mr.SO3.vee(hat)

    benchmark(fn)

@pytest.mark.dev
@pytest.mark.parametrize("impl", ["mathrobors", "mathrobo"])
def test_so3_hat_commute_benchmark(benchmark, impl):
    import numpy as np
    import mathrobo as mr

    w = (0.2, 0.3, 0.4)
    w_np = np.array(w)

    if impl == "mathrobors":
        fn = lambda: mathrobors.SO3.hat_commute(w)
    else:
        fn = lambda: mr.SO3.hat_commute(w_np)

    benchmark(fn)

@pytest.mark.dev
@pytest.mark.parametrize("impl", ["mathrobors", "mathrobo"])
def test_so3_eye_benchmark(benchmark, impl):
    import mathrobo as mr

    if impl == "mathrobors":
        fn = lambda: mathrobors.SO3.eye()
    else:
        fn = lambda: mr.SO3.eye()

    benchmark(fn)

