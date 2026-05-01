#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MATHROBO_ROOT = REPO_ROOT / "vendor" / "MathRobo"
sys.path.insert(0, str(MATHROBO_ROOT))

import mathrobo as mr  # noqa: E402
from mathrobo.lie.se3 import SE3  # noqa: E402
from mathrobo.lie.so3 import SO3  # noqa: E402


def array(value):
    return np.asarray(value, dtype=float).tolist()


def main() -> None:
    so3_vector = np.array([0.2, -0.1, 0.3], dtype=float)
    so3_scale = 0.75
    so3_left = np.array([0.4, -0.2, 0.1], dtype=float)
    so3_right = np.array([-0.3, 0.5, 0.2], dtype=float)
    quaternion = np.array([0.9238795325, 0.0, 0.3826834324, 0.0], dtype=float)

    se3_twist = np.array([0.2, -0.1, 0.3, 1.0, -2.0, 0.5], dtype=float)
    se3_scale = 0.6
    se3_left_rot = SO3.exp(np.array([0.1, 0.2, -0.15], dtype=float))
    se3_right_rot = SO3.exp(np.array([-0.05, 0.3, 0.25], dtype=float))
    se3_left_pos = np.array([0.4, -0.8, 1.2], dtype=float)
    se3_right_pos = np.array([-1.0, 0.25, 0.75], dtype=float)
    se3_point = np.array([0.2, -0.4, 1.5], dtype=float)

    so3_left_obj = SO3(SO3.exp(so3_left))
    so3_right_obj = SO3(SO3.exp(so3_right))
    se3_left_obj = SE3(se3_left_rot, se3_left_pos)
    se3_right_obj = SE3(se3_right_rot, se3_right_pos)

    cmtm_so3_derivatives = np.array(
        [[0.1, -0.2, 0.3], [0.05, 0.01, -0.04]],
        dtype=float,
    )
    cmtm_so3_right_rotvec = np.array([-0.2, 0.15, 0.35], dtype=float)
    cmtm_so3_right_derivatives = np.array(
        [[-0.02, 0.04, 0.01], [0.03, -0.01, 0.02]],
        dtype=float,
    )
    cmtm_so3 = SO3(SO3.exp(np.array([0.25, -0.35, 0.1], dtype=float)))
    cmtm_so3_right = SO3(SO3.exp(cmtm_so3_right_rotvec))
    cmtm_so3_obj = mr.CMTM[SO3](cmtm_so3, cmtm_so3_derivatives)
    cmtm_so3_other = mr.CMTM[SO3](cmtm_so3_right, cmtm_so3_right_derivatives)

    cmtm_se3_rotvec = np.array([0.15, -0.2, 0.05], dtype=float)
    cmtm_se3_translation = np.array([0.4, -0.6, 0.8], dtype=float)
    cmtm_se3_derivatives = np.array(
        [[0.1, -0.05, 0.03, 1.0, -0.4, 0.2], [0.02, 0.01, -0.04, -0.3, 0.6, 0.5]],
        dtype=float,
    )
    cmtm_se3_right_rotvec = np.array([-0.05, 0.18, 0.22], dtype=float)
    cmtm_se3_right_translation = np.array([-0.7, 0.2, 0.1], dtype=float)
    cmtm_se3_right_derivatives = np.array(
        [[-0.03, 0.06, 0.02, 0.2, 0.1, -0.5], [0.04, -0.02, 0.01, -0.1, 0.3, 0.25]],
        dtype=float,
    )
    cmtm_se3 = SE3(SO3.exp(cmtm_se3_rotvec), cmtm_se3_translation)
    cmtm_se3_right = SE3(SO3.exp(cmtm_se3_right_rotvec), cmtm_se3_right_translation)
    cmtm_se3_obj = mr.CMTM[SE3](cmtm_se3, cmtm_se3_derivatives)
    cmtm_se3_other = mr.CMTM[SE3](cmtm_se3_right, cmtm_se3_right_derivatives)

    payload = {
        "so3": {
            "vector": array(so3_vector),
            "scale": so3_scale,
            "hat": array(SO3.hat(so3_vector)),
            "vee": array(SO3.vee(SO3.hat(so3_vector))),
            "exp": array(SO3.exp(so3_vector, so3_scale)),
            "compose_left": array(so3_left),
            "compose_right": array(so3_right),
            "compose": array((so3_left_obj @ so3_right_obj).mat()),
            "quaternion": array(quaternion),
            "quaternion_matrix": array(SO3.set_quaternion(quaternion).mat()),
        },
        "se3": {
            "twist": array(se3_twist),
            "scale": se3_scale,
            "hat": array(SE3.hat(se3_twist)),
            "vee": array(SE3.vee(SE3.hat(se3_twist))),
            "exp": array(SE3.exp(se3_twist, se3_scale)),
            "left_rotvec": array(np.array([0.1, 0.2, -0.15], dtype=float)),
            "left_translation": array(se3_left_pos),
            "right_rotvec": array(np.array([-0.05, 0.3, 0.25], dtype=float)),
            "right_translation": array(se3_right_pos),
            "compose": array((se3_left_obj @ se3_right_obj).mat()),
            "inverse": array(se3_left_obj.inv().mat()),
            "adjoint": array(se3_left_obj.mat_adj()),
            "point": array(se3_point),
            "apply": array(se3_left_obj @ se3_point),
        },
        "cmtm_so3": {
            "rotation_vector": array(np.array([0.25, -0.35, 0.1], dtype=float)),
            "derivatives": array(cmtm_so3_derivatives),
            "mat": array(cmtm_so3_obj.mat()),
            "mat_adj": array(cmtm_so3_obj.mat_adj()),
            "mat_inv": array(cmtm_so3_obj.mat_inv()),
            "mat_inv_adj": array(cmtm_so3_obj.mat_inv_adj()),
            "set_mat_vecs": array(mr.CMTM.set_mat(SO3, cmtm_so3_obj.mat()).vecs()),
            "inverse_mat": array(cmtm_so3_obj.inv().mat()),
            "compose_right_rotation_vector": array(cmtm_so3_right_rotvec),
            "compose_right_derivatives": array(cmtm_so3_right_derivatives),
            "compose_mat": array((cmtm_so3_obj @ cmtm_so3_other).mat()),
            "hat": array(mr.CMTM.hat(SO3, cmtm_so3_derivatives)),
            "hat_adj": array(mr.CMTM.hat_adj(SO3, cmtm_so3_derivatives)),
            "hat_commute_adj": array(mr.CMTM.hat_commute_adj(SO3, cmtm_so3_derivatives)),
            "vee": array(mr.CMTM.vee(SO3, mr.CMTM.hat(SO3, cmtm_so3_derivatives))),
            "vee_adj": array(mr.CMTM.vee_adj(SO3, mr.CMTM.hat_adj(SO3, cmtm_so3_derivatives))),
            "tangent_mat": array(cmtm_so3_obj.tangent_mat()),
            "tangent_mat_cm": array(cmtm_so3_obj.tangent_mat_cm()),
        },
        "cmtm_se3": {
            "rotation_vector": array(cmtm_se3_rotvec),
            "translation": array(cmtm_se3_translation),
            "derivatives": array(cmtm_se3_derivatives),
            "mat": array(cmtm_se3_obj.mat()),
            "mat_adj": array(cmtm_se3_obj.mat_adj()),
            "mat_inv": array(cmtm_se3_obj.mat_inv()),
            "mat_inv_adj": array(cmtm_se3_obj.mat_inv_adj()),
            "set_mat_vecs": array(mr.CMTM.set_mat(SE3, cmtm_se3_obj.mat()).vecs()),
            "inverse_mat": array(cmtm_se3_obj.inv().mat()),
            "compose_right_rotation_vector": array(cmtm_se3_right_rotvec),
            "compose_right_translation": array(cmtm_se3_right_translation),
            "compose_right_derivatives": array(cmtm_se3_right_derivatives),
            "compose_mat": array((cmtm_se3_obj @ cmtm_se3_other).mat()),
            "hat": array(mr.CMTM.hat(SE3, cmtm_se3_derivatives)),
            "hat_adj": array(mr.CMTM.hat_adj(SE3, cmtm_se3_derivatives)),
            "hat_commute_adj": array(mr.CMTM.hat_commute_adj(SE3, cmtm_se3_derivatives)),
            "vee": array(mr.CMTM.vee(SE3, mr.CMTM.hat(SE3, cmtm_se3_derivatives))),
            "vee_adj": array(mr.CMTM.vee_adj(SE3, mr.CMTM.hat_adj(SE3, cmtm_se3_derivatives))),
            "tangent_mat": array(cmtm_se3_obj.tangent_mat()),
            "tangent_mat_cm": array(cmtm_se3_obj.tangent_mat_cm()),
        },
    }

    json.dump(payload, sys.stdout)


if __name__ == "__main__":
    main()
