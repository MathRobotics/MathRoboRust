import math

import pytest

import mathrobors
from test_support.mathrobo_vendor import import_mathrobo


def approx_eq(a, b, tol=1e-12):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(x - y) < tol, f"expected {y}, got {x}"


def approx_eq_matrix(actual, expected, tol=1e-12):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        approx_eq(actual_row, expected_row, tol)


def approx_eq_matrix_dynamic(actual, expected, tol=1e-12):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        approx_eq(actual_row, expected_row, tol)


@pytest.mark.dev
def test_so3_matches_vendor_mathrobo():
    import numpy as np

    mathrobo = import_mathrobo()

    rotation = mathrobors.SO3.from_axis_angle((0.0, 0.0, 1.0), math.pi / 2.0)
    rotation_mr = mathrobo.SO3.set_mat(
        mathrobo.SO3.exp(np.array([0.0, 0.0, 1.0], dtype=float), math.pi / 2.0)
    )

    approx_eq_matrix(rotation.matrix(), rotation_mr.mat().tolist())
    approx_eq(rotation.quaternion(), rotation_mr.quaternion().tolist())

    hat = mathrobors.SO3.hat((0.2, 0.3, 0.4))
    hat_mr = mathrobo.SO3.hat(np.array([0.2, 0.3, 0.4], dtype=float))
    approx_eq_matrix(hat, hat_mr.tolist())
    approx_eq(mathrobors.SO3.vee(hat), mathrobo.SO3.vee(hat_mr).tolist())
    approx_eq_matrix(
        mathrobors.SO3.exp((0.1, -0.2, 0.3), None),
        mathrobo.SO3.exp(np.array([0.1, -0.2, 0.3], dtype=float)).tolist(),
    )


@pytest.mark.dev
def test_se3_matches_vendor_mathrobo():
    import numpy as np

    mathrobo = import_mathrobo()

    axis = np.array([0.0, 0.0, 1.0], dtype=float)
    angle = math.pi / 3.0
    translation = np.array([0.25, -0.5, 1.0], dtype=float)
    point = np.array([1.0, 0.0, 0.0], dtype=float)
    twist = np.array([0.1, -0.2, 0.3, 1.0, -2.0, 3.0], dtype=float)

    transform = mathrobors.SE3.from_axis_angle_translation(axis.tolist(), angle, translation.tolist())
    transform_mr = mathrobo.SE3(mathrobo.SO3.exp(axis, angle), translation)

    approx_eq_matrix(transform.matrix(), transform_mr.mat().tolist())
    approx_eq_matrix(transform.mat_adj(), transform_mr.mat_adj().tolist())
    approx_eq(transform.apply(point.tolist()), (transform_mr @ point).tolist())
    approx_eq_matrix(mathrobors.SE3.exp(twist.tolist(), None), mathrobo.SE3.exp(twist).tolist())


@pytest.mark.dev
def test_cmtm_matches_vendor_mathrobo():
    import numpy as np

    mathrobo = import_mathrobo()

    so3_rotation = mathrobors.SO3.from_rotation_vector((0.25, -0.35, 0.1))
    so3_derivatives = [[0.1, -0.2, 0.3], [0.05, 0.01, -0.04]]
    so3_cmtm = mathrobors.SO3CMTM.from_so3_with_derivatives(so3_rotation, so3_derivatives)
    so3_cmtm_mr = mathrobo.CMTM[mathrobo.SO3](
        mathrobo.SO3.set_mat(mathrobo.SO3.exp(np.array([0.25, -0.35, 0.1], dtype=float))),
        np.array(so3_derivatives, dtype=float),
    )

    approx_eq_matrix_dynamic(so3_cmtm.mat(None), so3_cmtm_mr.mat().tolist())
    approx_eq_matrix_dynamic(so3_cmtm.mat_adj(None), so3_cmtm_mr.mat_adj().tolist())
    approx_eq_matrix_dynamic(so3_cmtm.mat_inv(None), so3_cmtm_mr.mat_inv().tolist())
    approx_eq_matrix_dynamic(so3_cmtm.mat_inv_adj(None), so3_cmtm_mr.mat_inv_adj().tolist())
    approx_eq_matrix_dynamic(so3_cmtm.tangent_mat(None), so3_cmtm_mr.tangent_mat().tolist())
    approx_eq_matrix_dynamic(
        so3_cmtm.tangent_mat_cm(None), so3_cmtm_mr.tangent_mat_cm().tolist()
    )
    approx_eq_matrix_dynamic(
        mathrobors.SO3CMTM.set_mat(so3_cmtm.mat(None)).mat(None),
        mathrobo.CMTM.set_mat(mathrobo.SO3, so3_cmtm_mr.mat()).mat().tolist(),
    )
    approx_eq_matrix_dynamic(
        mathrobors.SO3CMTM.set_mat_adj(so3_cmtm.mat_adj(None)).mat_adj(None),
        so3_cmtm.mat_adj(None),
    )
    approx_eq_matrix_dynamic(
        so3_cmtm.compose(so3_cmtm).mat(None),
        (so3_cmtm_mr @ so3_cmtm_mr).mat().tolist(),
    )
    approx_eq_matrix_dynamic(
        mathrobors.SO3CMTM.hat(so3_derivatives),
        mathrobo.CMTM.hat(mathrobo.SO3, np.array(so3_derivatives, dtype=float)).tolist(),
    )
    approx_eq_matrix_dynamic(
        mathrobors.SO3CMTM.hat_adj(so3_derivatives),
        mathrobo.CMTM.hat_adj(mathrobo.SO3, np.array(so3_derivatives, dtype=float)).tolist(),
    )
    approx_eq_matrix_dynamic(
        mathrobors.SO3CMTM.hat_commute_adj(so3_derivatives),
        mathrobo.CMTM.hat_commute_adj(mathrobo.SO3, np.array(so3_derivatives, dtype=float)).tolist(),
    )
    approx_eq_matrix_dynamic(
        mathrobors.SO3CMTM.vee(so3_cmtm.mat(None)),
        mathrobo.CMTM.vee(mathrobo.SO3, so3_cmtm_mr.mat()).tolist(),
    )
    approx_eq_matrix_dynamic(
        mathrobors.SO3CMTM.vee_adj(so3_cmtm.mat_adj(None)),
        mathrobo.CMTM.vee_adj(mathrobo.SO3, so3_cmtm_mr.mat_adj()).tolist(),
    )

    se3_rotation = mathrobors.SE3.from_parts(
        mathrobors.SO3.from_rotation_vector((0.15, -0.2, 0.05)), (0.4, -0.6, 0.8)
    )
    se3_derivatives = [
        [0.1, -0.05, 0.03, 1.0, -0.4, 0.2],
        [0.02, 0.01, -0.04, -0.3, 0.6, 0.5],
    ]
    se3_cmtm = mathrobors.SE3CMTM.from_se3_with_derivatives(se3_rotation, se3_derivatives)
    se3_cmtm_mr = mathrobo.CMTM[mathrobo.SE3](
        mathrobo.SE3(mathrobo.SO3.exp(np.array([0.15, -0.2, 0.05], dtype=float)), np.array([0.4, -0.6, 0.8], dtype=float)),
        np.array(se3_derivatives, dtype=float),
    )

    approx_eq_matrix_dynamic(se3_cmtm.mat(None), se3_cmtm_mr.mat().tolist())
    approx_eq_matrix_dynamic(se3_cmtm.mat_adj(None), se3_cmtm_mr.mat_adj().tolist())
    approx_eq_matrix_dynamic(se3_cmtm.mat_inv(None), se3_cmtm_mr.mat_inv().tolist())
    approx_eq_matrix_dynamic(se3_cmtm.mat_inv_adj(None), se3_cmtm_mr.mat_inv_adj().tolist())
    approx_eq_matrix_dynamic(se3_cmtm.tangent_mat(None), se3_cmtm_mr.tangent_mat().tolist())
    approx_eq_matrix_dynamic(
        se3_cmtm.tangent_mat_cm(None), se3_cmtm_mr.tangent_mat_cm().tolist()
    )
    approx_eq_matrix_dynamic(
        mathrobors.SE3CMTM.set_mat(se3_cmtm.mat(None)).mat(None),
        mathrobo.CMTM.set_mat(mathrobo.SE3, se3_cmtm_mr.mat()).mat().tolist(),
    )
    approx_eq_matrix_dynamic(
        mathrobors.SE3CMTM.set_mat_adj(se3_cmtm.mat_adj(None)).mat_adj(None),
        se3_cmtm.mat_adj(None),
    )
    approx_eq_matrix_dynamic(
        se3_cmtm.compose(se3_cmtm).mat(None),
        (se3_cmtm_mr @ se3_cmtm_mr).mat().tolist(),
    )


@pytest.mark.dev
def test_cmtm_cmvector_variation_matches_vendor_mathrobo():
    import numpy as np

    mathrobo = import_mathrobo()

    so3_rotation = mathrobors.SO3.from_rotation_vector((0.15, -0.1, 0.2))
    so3_derivatives = [[0.12, -0.04, 0.08], [0.03, 0.01, -0.02]]
    so3_cmtm = mathrobors.CMTM[mathrobors.SO3](so3_rotation, so3_derivatives)
    so3_cmtm_mr = mathrobo.CMTM[mathrobo.SO3](
        mathrobo.SO3.set_mat(mathrobo.SO3.exp(np.array([0.15, -0.1, 0.2], dtype=float))),
        np.array(so3_derivatives, dtype=float),
    )

    so3_arb = np.array([[0.2, -0.1, 0.3], [0.05, 0.04, -0.02], [0.01, -0.03, 0.02]], dtype=float)
    so3_tan = np.array([[0.1, 0.2, -0.1], [0.03, -0.02, 0.04], [0.02, 0.01, -0.03]], dtype=float)

    so3_res = so3_cmtm.mat_var_x_arb_vec(
        mathrobors.CMVector(so3_arb), mathrobors.CMVector(so3_tan), frame="bframe"
    ).cm_vec()
    so3_res_mr = so3_cmtm_mr.mat_var_x_arb_vec(
        mathrobo.cmvec.CMVector(so3_arb), mathrobo.cmvec.CMVector(so3_tan), frame="bframe"
    ).cm_vec()
    approx_eq(so3_res, so3_res_mr.tolist())

    so3_jacob = so3_cmtm.mat_var_x_arb_vec_jacob(mathrobors.CMVector(so3_arb), frame="bframe")
    so3_jacob_mr = so3_cmtm_mr.mat_var_x_arb_vec_jacob(
        mathrobo.cmvec.CMVector(so3_arb), frame="bframe"
    )
    approx_eq_matrix_dynamic(so3_jacob, so3_jacob_mr.tolist())

    se3_transform = mathrobors.SE3.from_parts(
        mathrobors.SO3.from_rotation_vector((0.05, -0.08, 0.12)), (0.3, -0.2, 0.4)
    )
    se3_derivatives = [
        [0.06, -0.02, 0.03, 0.4, -0.1, 0.2],
        [0.01, 0.02, -0.03, -0.2, 0.3, 0.1],
    ]
    se3_cmtm = mathrobors.CMTM[mathrobors.SE3](se3_transform, se3_derivatives)
    se3_cmtm_mr = mathrobo.CMTM[mathrobo.SE3](
        mathrobo.SE3(
            mathrobo.SO3.exp(np.array([0.05, -0.08, 0.12], dtype=float)),
            np.array([0.3, -0.2, 0.4], dtype=float),
        ),
        np.array(se3_derivatives, dtype=float),
    )

    se3_arb = np.array(
        [
            [0.1, -0.2, 0.3, 0.4, -0.5, 0.6],
            [0.03, 0.01, -0.02, -0.1, 0.2, 0.05],
            [0.02, -0.01, 0.04, 0.06, -0.03, 0.01],
        ],
        dtype=float,
    )
    se3_tan = np.array(
        [
            [0.04, -0.01, 0.02, 0.2, -0.1, 0.3],
            [0.01, 0.03, -0.02, -0.05, 0.04, 0.02],
            [0.02, -0.02, 0.01, 0.03, 0.01, -0.04],
        ],
        dtype=float,
    )

    se3_res = se3_cmtm.mat_var_x_arb_vec(
        mathrobors.CMVector(se3_arb), mathrobors.CMVector(se3_tan), frame="bframe"
    ).cm_vec()
    se3_res_mr = se3_cmtm_mr.mat_var_x_arb_vec(
        mathrobo.cmvec.CMVector(se3_arb), mathrobo.cmvec.CMVector(se3_tan), frame="bframe"
    ).cm_vec()
    approx_eq(se3_res, se3_res_mr.tolist())

    se3_jacob = se3_cmtm.mat_var_x_arb_vec_jacob(mathrobors.CMVector(se3_arb), frame="bframe")
    se3_jacob_mr = se3_cmtm_mr.mat_var_x_arb_vec_jacob(
        mathrobo.cmvec.CMVector(se3_arb), frame="bframe"
    )
    approx_eq_matrix_dynamic(se3_jacob, se3_jacob_mr.tolist())


@pytest.mark.dev
def test_wrench_and_inertia_matches_vendor_mathrobo():
    import numpy as np

    mathrobo = import_mathrobo()

    so3_vec = np.array([0.2, -0.1, 0.3], dtype=float)
    so3_inertia_vec = np.array([1.0, 2.0, 3.0, 0.4, 0.5, 0.6], dtype=float)

    approx_eq_matrix_dynamic(
        mathrobors.SO3wrench.hat(so3_vec), mathrobo.SO3wrench.hat(so3_vec).tolist()
    )
    approx_eq_matrix_dynamic(
        mathrobors.SO3wrench.hat_commute(so3_vec),
        mathrobo.SO3wrench.hat_commute(so3_vec).tolist(),
    )
    approx_eq_matrix_dynamic(
        mathrobors.SO3wrench.exp(so3_vec, 0.7), mathrobo.SO3wrench.exp(so3_vec, 0.7).tolist()
    )
    approx_eq_matrix_dynamic(
        mathrobors.SO3wrench.exp_integ(so3_vec, 0.7),
        mathrobo.SO3wrench.exp_integ(so3_vec, 0.7).tolist(),
    )

    approx_eq_matrix_dynamic(
        mathrobors.SO3inertia.hat(so3_inertia_vec), mathrobo.SO3inertia.hat(so3_inertia_vec).tolist()
    )
    approx_eq_matrix_dynamic(
        mathrobors.SO3inertia.hat_commute(so3_vec),
        mathrobo.SO3inertia.hat_commute(so3_vec).tolist(),
    )

    rotvec = np.array([0.1, -0.2, 0.15], dtype=float)
    translation = np.array([0.25, -0.4, 0.6], dtype=float)
    twist = np.array([0.2, -0.1, 0.3, 1.0, -0.5, 0.25], dtype=float)
    se3_wrench = mathrobors.SE3wrench(mathrobors.SO3.exp(rotvec.tolist(), None), translation)
    se3_wrench_mr = mathrobo.SE3wrench(mathrobo.SO3.exp(rotvec), translation)

    approx_eq_matrix_dynamic(se3_wrench.mat_adj(), se3_wrench_mr.mat_adj().tolist())
    approx_eq_matrix_dynamic(se3_wrench.mat_inv_adj(), se3_wrench_mr.mat_inv_adj().tolist())
    approx_eq_matrix_dynamic(
        mathrobors.SE3wrench.hat_adj(twist), mathrobo.SE3wrench.hat_adj(twist).tolist()
    )
    approx_eq_matrix_dynamic(
        mathrobors.SE3wrench.hat_commute(twist),
        mathrobo.SE3wrench.hat_commute(twist).tolist(),
    )
    approx_eq_matrix_dynamic(
        mathrobors.SE3wrench.hat_commute_adj(twist),
        mathrobo.SE3wrench.hat_commute_adj(twist).tolist(),
    )
    approx_eq_matrix_dynamic(
        mathrobors.SE3wrench.exp(twist, 0.5), mathrobo.SE3wrench.exp(twist, 0.5).tolist()
    )
    approx_eq_matrix_dynamic(
        mathrobors.SE3wrench.exp_integ(twist, 0.5),
        mathrobo.SE3wrench.exp_integ(twist, 0.5).tolist(),
    )
    approx_eq_matrix_dynamic(
        se3_wrench @ np.eye(6, dtype=float),
        (se3_wrench_mr @ np.eye(6, dtype=float)).tolist(),
    )

    arb_vec = np.array([0.4, -0.3, 0.2, 0.1, -0.5, 0.6], dtype=float)
    tan_var_vec = np.array([0.05, -0.02, 0.03, 0.2, -0.1, 0.04], dtype=float)
    approx_eq(
        se3_wrench.mat_var_x_arb_vec(arb_vec, tan_var_vec, frame="bframe"),
        mathrobo.SE3wrench(mathrobo.SO3.exp(rotvec), translation)
        .mat_var_x_arb_vec(arb_vec, tan_var_vec, frame="bframe")
        .tolist(),
    )
    approx_eq_matrix_dynamic(
        se3_wrench.mat_var_x_arb_vec_jacob(arb_vec, frame="bframe"),
        mathrobo.SE3wrench(mathrobo.SO3.exp(rotvec), translation)
        .mat_var_x_arb_vec_jacob(arb_vec, frame="bframe")
        .tolist(),
    )
