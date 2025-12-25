import math
import pytest
import mathrobors


def approx_eq(a, b, tol=1e-12):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(x - y) < tol, f"expected {y}, got {x}"


def test_python_interface_produces_same_values():
    transform = mathrobors.SE3.from_axis_angle_translation(
        (0.0, 0.0, 1.0), math.pi / 2.0, (0.5, -0.25, 1.25)
    )
    result = transform.apply((1.0, 0.0, 0.0))
    approx_eq(result, (0.5, 0.75, 1.25), 1e-12)

@pytest.mark.dev
def test_compare_mathrobo():
    pass
