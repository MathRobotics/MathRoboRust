from __future__ import annotations

import math
from pkgutil import extend_path

import numpy as np

from ._native import CMTM as _RawCMTM
from ._native import SE3 as _NativeSE3
from ._native import SO3 as _NativeSO3

__path__ = extend_path(__path__, __name__)


def _as_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _ensure_numpy_lib(lib) -> None:
    if lib not in (None, "numpy"):
        raise ValueError("Unsupported library. Choose 'numpy'.")


def _factorial_scales(order: int) -> np.ndarray:
    return np.asarray([math.factorial(index) for index in range(order)], dtype=float)


def _toeplitz_blocks(blocks) -> np.ndarray:
    if not blocks:
        return np.zeros((0, 0), dtype=float)

    blocks = [_as_array(block) for block in blocks]
    row_size, col_size = blocks[0].shape
    order = len(blocks)
    matrix = np.zeros((row_size * order, col_size * order), dtype=float)

    for index, block in enumerate(blocks):
        for column in range(order - index):
            row = index + column
            matrix[
                row * row_size : (row + 1) * row_size,
                column * col_size : (column + 1) * col_size,
            ] = block

    return matrix


def _empty_vecs(dim: int) -> np.ndarray:
    return np.zeros((0, dim), dtype=float)


def _ensure_vec_matrix(values, dim: int, order: int | None = None) -> np.ndarray:
    array = _as_array(values)
    if array.ndim == 1:
        if order is None:
            if array.size % dim != 0:
                raise ValueError("vector length is not divisible by the requested dimension")
            order = array.size // dim
        array = array.reshape(order, dim)
    if array.ndim != 2 or array.shape[1] != dim:
        raise ValueError(f"expected a 2D array with row width {dim}")
    return array


def _rotation_matrix(value=None) -> np.ndarray:
    if value is None:
        return np.identity(3, dtype=float)
    if hasattr(value, "mat"):
        value = value.mat()
    matrix = _as_array(value)
    if matrix.shape != (3, 3):
        raise ValueError("expected a 3x3 rotation matrix")
    return matrix


def _translation_vector(value=None) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=float)
    vector = _as_array(value)
    if vector.shape != (3,):
        raise ValueError("expected a 3D translation vector")
    return vector


def _vector_arg(values, dim: int):
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=float)
        if array.shape != (dim,):
            raise ValueError(f"expected a {dim}D vector")
        return array.tolist()
    if isinstance(values, (list, tuple)):
        if len(values) != dim:
            raise ValueError(f"expected a {dim}D vector")
        return values

    array = np.asarray(values, dtype=float)
    if array.shape != (dim,):
        raise ValueError(f"expected a {dim}D vector")
    return array.tolist()


def _matrix_arg(values, shape: tuple[int, int]):
    rows, cols = shape
    array = _as_array(values)
    if array.shape != shape:
        raise ValueError(f"expected a {rows}x{cols} matrix")
    return array.tolist()


def _alloc_vec(dim: int) -> np.ndarray:
    return np.empty(dim, dtype=float)


def _alloc_mat(rows: int, cols: int) -> np.ndarray:
    return np.empty((rows, cols), dtype=float)


def _homogeneous_matrix(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=float)
    matrix[0:3, 0:3] = rotation
    matrix[0:3, 3] = position
    return matrix


def _adjoint_matrix(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    matrix = np.zeros((6, 6), dtype=float)
    matrix[0:3, 0:3] = rotation
    matrix[3:6, 0:3] = _as_array(SO3.hat(position.tolist())) @ rotation
    matrix[3:6, 3:6] = rotation
    return matrix


def _adjoint_matrix_wrench(rotation: np.ndarray, position: np.ndarray) -> np.ndarray:
    matrix = np.zeros((6, 6), dtype=float)
    matrix[0:3, 0:3] = rotation
    matrix[0:3, 3:6] = _as_array(SO3.hat(position.tolist())) @ rotation
    matrix[3:6, 3:6] = rotation
    return matrix


def _vee_cm(group, hat_mat, adjoint: bool):
    matrix = _as_array(hat_mat)
    size = group.mat_adj_size() if adjoint else group.mat_size()
    vee = group.vee_adj if adjoint else group.vee
    dim = group.dof()

    if matrix.shape[0] % size != 0 or matrix.shape[1] % size != 0:
        raise ValueError("hat matrix shape is incompatible with the selected group")

    order = matrix.shape[0] // size
    vecs = np.zeros((order, dim), dtype=float)
    scales = _factorial_scales(order)

    for index in range(order):
        acc = np.zeros(dim, dtype=float)
        for column in range(index, order):
            block = matrix[
                column * size : (column + 1) * size,
                (column - index) * size : (column - index + 1) * size,
            ]
            acc += _as_array(vee(block.tolist()))
        vecs[index] = acc / (order - index) * scales[index]

    return CMVector(vecs)


def _wrap_so3(value) -> "SO3":
    if isinstance(value, SO3):
        return value
    return SO3(_inner=value)


def _wrap_se3(value) -> "SE3":
    if isinstance(value, SE3):
        return value
    return SE3(_inner=value)


def _as_native_so3(value):
    if isinstance(value, SO3):
        return value._inner
    if isinstance(value, _NativeSO3):
        return value
    if hasattr(value, "mat"):
        return _NativeSO3.set_mat(_as_array(value.mat()).tolist())
    return _NativeSO3.set_mat(_as_array(value).tolist())


def _as_native_se3(value):
    if isinstance(value, SE3):
        return value._inner
    if isinstance(value, _NativeSE3):
        return value
    if hasattr(value, "mat"):
        return _NativeSE3.set_mat(_as_array(value.mat()).tolist())
    return _NativeSE3.set_mat(_as_array(value).tolist())


def _as_native_group(group, value):
    if group is SO3:
        return _as_native_so3(value)
    if group is SE3:
        return _as_native_se3(value)
    raise TypeError("group must be SO3 or SE3")


class SO3:
    def __init__(self, matrix=None, LIB: str = "numpy", *, _inner=None):
        _ensure_numpy_lib(LIB)
        if _inner is not None:
            self._inner = _inner
        else:
            native_matrix = None if matrix is None else _as_array(matrix).tolist()
            self._inner = _NativeSO3(native_matrix, LIB)

    @property
    def lib(self) -> str:
        return "numpy"

    @staticmethod
    def dof() -> int:
        return _NativeSO3.dof()

    @staticmethod
    def mat_size() -> int:
        return _NativeSO3.mat_size()

    @staticmethod
    def mat_adj_size() -> int:
        return _NativeSO3.mat_adj_size()

    @staticmethod
    def from_axis_angle(axis, angle: float) -> "SO3":
        return _wrap_so3(_NativeSO3.from_axis_angle(_vector_arg(axis, 3), angle))

    @staticmethod
    def from_quaternion(quaternion) -> "SO3":
        return _wrap_so3(_NativeSO3.from_quaternion(_vector_arg(quaternion, 4)))

    @staticmethod
    def quaternion_to_mat(quaternion) -> np.ndarray:
        return _as_array(_NativeSO3.quaternion_to_mat(_as_array(quaternion).tolist()))

    @staticmethod
    def set_quaternion(quaternion, LIB: str = "numpy") -> "SO3":
        _ensure_numpy_lib(LIB)
        return _wrap_so3(_NativeSO3.set_quaternion(_vector_arg(quaternion, 4)))

    @staticmethod
    def from_euler_angles(roll: float, pitch: float, yaw: float) -> "SO3":
        return _wrap_so3(_NativeSO3.from_euler_angles(roll, pitch, yaw))

    @staticmethod
    def set_euler(euler, order: str = "ZYX", LIB: str = "numpy") -> "SO3":
        _ensure_numpy_lib(LIB)
        return _wrap_so3(_NativeSO3.set_euler(tuple(_vector_arg(euler, 3)), order, LIB))

    @staticmethod
    def from_rotation_vector(vector) -> "SO3":
        return _wrap_so3(_NativeSO3.from_rotation_vector(_vector_arg(vector, 3)))

    @staticmethod
    def exp(vector, a: float | None = None) -> np.ndarray:
        return _NativeSO3.exp_array(_vector_arg(vector, 3), a)

    @staticmethod
    def exp_integ(vector, a: float | None = None) -> np.ndarray:
        return _NativeSO3.exp_integ_array(_vector_arg(vector, 3), a)

    @staticmethod
    def exp_integ2nd(vector, a: float | None = None) -> np.ndarray:
        return _NativeSO3.exp_integ2nd_array(_vector_arg(vector, 3), a)

    @staticmethod
    def exp_adj(vector, a: float | None = None) -> np.ndarray:
        return _NativeSO3.exp_adj_array(_vector_arg(vector, 3), a)

    @staticmethod
    def exp_integ_adj(vector, a: float | None = None) -> np.ndarray:
        return _NativeSO3.exp_integ_adj_array(_vector_arg(vector, 3), a)

    @staticmethod
    def rand(LIB: str = "numpy") -> "SO3":
        _ensure_numpy_lib(LIB)
        return _wrap_so3(_NativeSO3.rand(LIB))

    def apply(self, vector) -> np.ndarray:
        return self._inner.apply_array(_vector_arg(vector, 3))

    def apply_into(self, vector, out) -> None:
        self._inner.apply_into(_vector_arg(vector, 3), out)

    @staticmethod
    def hat(vector) -> np.ndarray:
        return _NativeSO3.hat_array(_vector_arg(vector, 3))

    @staticmethod
    def hat_commute(vector) -> np.ndarray:
        return _NativeSO3.hat_commute_array(_vector_arg(vector, 3))

    @staticmethod
    def hat_adj(vector) -> np.ndarray:
        return _NativeSO3.hat_adj_array(_vector_arg(vector, 3))

    @staticmethod
    def hat_commute_adj(vector) -> np.ndarray:
        return _NativeSO3.hat_commute_adj_array(_vector_arg(vector, 3))

    @staticmethod
    def vee(matrix) -> np.ndarray:
        return _NativeSO3.vee_array(_matrix_arg(matrix, (3, 3)))

    @staticmethod
    def vee_adj(matrix) -> np.ndarray:
        return _NativeSO3.vee_adj_array(_matrix_arg(matrix, (3, 3)))

    @staticmethod
    def sub_tan_vec(left: "SO3", right: "SO3", frame: str = "bframe") -> np.ndarray:
        return _as_array(_NativeSO3.sub_tan_vec(_as_native_so3(left), _as_native_so3(right), frame))

    @staticmethod
    def so3_mul(left, right) -> np.ndarray:
        return _as_array(_NativeSO3.so3_mul(_as_array(left).tolist(), _as_array(right).tolist()))

    def compose(self, other: "SO3") -> "SO3":
        return _wrap_so3(self._inner.compose(_as_native_so3(other)))

    def __mul__(self, other: "SO3") -> "SO3":
        return self.compose(other)

    def __matmul__(self, other):
        if isinstance(other, (SO3, _NativeSO3)):
            return self.compose(other)
        matrix = _as_array(other)
        if matrix.shape == (3,):
            return self.apply(matrix)
        if matrix.shape == (3, 3):
            return self.mat() @ matrix
        raise TypeError("Right operand should be SO3 or a 3D vector/matrix")

    def inverse(self) -> "SO3":
        return _wrap_so3(self._inner.inverse())

    def inv(self) -> "SO3":
        return self.inverse()

    def matrix(self) -> np.ndarray:
        return self._inner.mat_array()

    def mat(self) -> np.ndarray:
        return self._inner.mat_array()

    @staticmethod
    def set_mat(matrix, LIB: str = "numpy") -> "SO3":
        _ensure_numpy_lib(LIB)
        return SO3(matrix, LIB)

    @staticmethod
    def set_mat_adj(matrix, LIB: str = "numpy") -> "SO3":
        _ensure_numpy_lib(LIB)
        return _wrap_so3(_NativeSO3.set_mat_adj(_as_array(matrix).tolist()))

    @staticmethod
    def eye(LIB: str = "numpy") -> "SO3":
        _ensure_numpy_lib(LIB)
        return _wrap_so3(_NativeSO3.eye())

    def mat_inv(self) -> np.ndarray:
        return self._inner.mat_inv_array()

    def mat_adj(self) -> np.ndarray:
        return self._inner.mat_adj_array()

    def mat_inv_adj(self) -> np.ndarray:
        return self._inner.mat_inv_adj_array()

    def quaternion(self) -> np.ndarray:
        return self._inner.quaternion_array()

    @staticmethod
    def mat_to_quaternion(matrix) -> np.ndarray:
        return _as_array(_NativeSO3.mat_to_quaternion(_as_array(matrix).tolist()))

    def euler_angles(self) -> np.ndarray:
        return _as_array(self._inner.euler_angles())

    def rotation_vector(self) -> np.ndarray:
        return self._inner.rotation_vector_array()

    def mat_var_x_arb_vec(self, arb_vec, tan_var_vec, frame: str = "bframe") -> np.ndarray:
        out = _alloc_vec(3)
        self._inner.mat_var_x_arb_vec_into(
            _vector_arg(arb_vec, 3),
            _vector_arg(tan_var_vec, 3),
            out,
            frame,
        )
        return out

    def mat_var_x_arb_vec_jacob(self, arb_vec, frame: str = "bframe") -> np.ndarray:
        out = _alloc_mat(3, 3)
        self._inner.mat_var_x_arb_vec_jacob_into(_vector_arg(arb_vec, 3), out, frame)
        return out

    def __repr__(self) -> str:
        return f"SO3({self.mat()}, LIB='numpy')"


class SE3:
    def __init__(self, rotation=None, position=None, LIB: str = "numpy", *, _inner=None):
        _ensure_numpy_lib(LIB)
        if _inner is not None:
            self._inner = _inner
        else:
            native_rotation = None if rotation is None else _rotation_matrix(rotation).tolist()
            native_position = None if position is None else _translation_vector(position).tolist()
            self._inner = _NativeSE3(native_rotation, native_position, LIB)

    @property
    def lib(self) -> str:
        return "numpy"

    @staticmethod
    def dof() -> int:
        return _NativeSE3.dof()

    @staticmethod
    def mat_size() -> int:
        return _NativeSE3.mat_size()

    @staticmethod
    def mat_adj_size() -> int:
        return _NativeSE3.mat_adj_size()

    @staticmethod
    def from_axis_angle_translation(axis, angle: float, translation) -> "SE3":
        return _wrap_se3(
            _NativeSE3.from_axis_angle_translation(
                _vector_arg(axis, 3), angle, _vector_arg(translation, 3)
            )
        )

    @staticmethod
    def from_parts(rotation, translation) -> "SE3":
        return _wrap_se3(_NativeSE3.from_parts(_as_native_so3(rotation), _vector_arg(translation, 3)))

    @staticmethod
    def from_matrix(matrix) -> "SE3":
        return _wrap_se3(_NativeSE3.from_matrix(_matrix_arg(matrix, (4, 4))))

    @staticmethod
    def set_pos_quaternion(position, quaternion, LIB: str = "numpy") -> "SE3":
        _ensure_numpy_lib(LIB)
        return _wrap_se3(
            _NativeSE3.set_pos_quaternion(_vector_arg(position, 3), _vector_arg(quaternion, 4))
        )

    @staticmethod
    def set_mat_adj(matrix, LIB: str = "numpy") -> "SE3":
        _ensure_numpy_lib(LIB)
        return _wrap_se3(_NativeSE3.set_mat_adj(_matrix_arg(matrix, (6, 6))))

    @staticmethod
    def change_class(other) -> "SE3":
        if isinstance(other, SE3):
            return _wrap_se3(other._inner.clone() if hasattr(other._inner, "clone") else other._inner)
        if isinstance(other, _NativeSE3):
            return _wrap_se3(other)
        return SE3(_rotation_matrix(other.rot()), _translation_vector(other.pos()))

    @staticmethod
    def hat(twist) -> np.ndarray:
        return _NativeSE3.hat_array(_vector_arg(twist, 6))

    @staticmethod
    def hat_commute(twist) -> np.ndarray:
        return _as_array(_NativeSE3.hat_commute(_as_array(twist).tolist()))

    @staticmethod
    def hat_adj(twist) -> np.ndarray:
        return _NativeSE3.hat_adj_array(_vector_arg(twist, 6))

    @staticmethod
    def hat_commute_adj(twist) -> np.ndarray:
        return _NativeSE3.hat_commute_adj_array(_vector_arg(twist, 6))

    @staticmethod
    def vee(matrix) -> np.ndarray:
        return _NativeSE3.vee_array(_matrix_arg(matrix, (4, 4)))

    @staticmethod
    def vee_adj(matrix) -> np.ndarray:
        return _NativeSE3.vee_adj_array(_matrix_arg(matrix, (6, 6)))

    @staticmethod
    def exp(twist, a: float | None = None) -> np.ndarray:
        return _NativeSE3.exp_array(_vector_arg(twist, 6), a)

    @staticmethod
    def exp_integ(twist, a: float | None = None) -> np.ndarray:
        return _NativeSE3.exp_integ_array(_vector_arg(twist, 6), a)

    @staticmethod
    def exp_adj(twist, a: float | None = None) -> np.ndarray:
        return _NativeSE3.exp_adj_array(_vector_arg(twist, 6), a)

    @staticmethod
    def exp_integ_adj(twist, a: float | None = None) -> np.ndarray:
        return _NativeSE3.exp_integ_adj_array(_vector_arg(twist, 6), a)

    @staticmethod
    def exp_into(twist, out, a: float | None = None) -> None:
        _NativeSE3.exp_into(_vector_arg(twist, 6), out, a)

    @staticmethod
    def rand(LIB: str = "numpy") -> "SE3":
        _ensure_numpy_lib(LIB)
        return _wrap_se3(_NativeSE3.rand(LIB))

    @staticmethod
    def sub_tan_vec(left: "SE3", right: "SE3", frame: str = "bframe") -> np.ndarray:
        return _as_array(_NativeSE3.sub_tan_vec(_as_native_se3(left), _as_native_se3(right), frame))

    @staticmethod
    def se3_mul(left_rot, left_pos, right_rot, right_pos):
        rot, pos = _NativeSE3.se3_mul(
            _rotation_matrix(left_rot).tolist(),
            _translation_vector(left_pos).tolist(),
            _rotation_matrix(right_rot).tolist(),
            _translation_vector(right_pos).tolist(),
        )
        return _as_array(rot), _as_array(pos)

    def apply(self, point) -> np.ndarray:
        return self._inner.apply_array(_vector_arg(point, 3))

    def apply_into(self, point, out) -> None:
        self._inner.apply_into(_vector_arg(point, 3), out)

    def compose(self, other: "SE3") -> "SE3":
        return _wrap_se3(self._inner.compose(_as_native_se3(other)))

    def __mul__(self, other: "SE3") -> "SE3":
        return self.compose(other)

    def __matmul__(self, other):
        if isinstance(other, (SE3, _NativeSE3)):
            return self.compose(other)
        matrix = _as_array(other)
        if matrix.shape == (3,):
            return self.apply(matrix)
        if matrix.shape == (6,):
            return self.mat_adj() @ matrix
        if matrix.shape == (4, 4):
            return self.mat() @ matrix
        if matrix.shape == (6, 6):
            return self.mat_adj() @ matrix
        raise TypeError("Right operand should be SE3 or a 3D/6D vector or a 4x4/6x6 matrix")

    def inverse(self) -> "SE3":
        return _wrap_se3(self._inner.inverse())

    def inv(self) -> "SE3":
        return self.inverse()

    def matrix(self) -> np.ndarray:
        return self._inner.mat_array()

    def mat(self) -> np.ndarray:
        return self._inner.mat_array()

    @staticmethod
    def set_mat(matrix, LIB: str = "numpy") -> "SE3":
        _ensure_numpy_lib(LIB)
        return _wrap_se3(_NativeSE3.set_mat(_as_array(matrix).tolist()))

    @staticmethod
    def eye(LIB: str = "numpy") -> "SE3":
        _ensure_numpy_lib(LIB)
        return _wrap_se3(_NativeSE3.eye())

    def mat_inv(self) -> np.ndarray:
        return self._inner.mat_inv_array()

    def mat_adj(self) -> np.ndarray:
        return self._inner.mat_adj_array()

    def mat_adj_into(self, out) -> None:
        self._inner.mat_adj_into(out)

    def mat_inv_adj(self) -> np.ndarray:
        return self._inner.mat_inv_adj_array()

    def translation(self) -> np.ndarray:
        return self._inner.translation_array()

    def position(self) -> np.ndarray:
        return self.translation()

    def pos(self) -> np.ndarray:
        return self.translation()

    def rotation(self) -> SO3:
        return _wrap_so3(self._inner.rotation())

    def rot(self) -> np.ndarray:
        return self._inner.rot_array()

    def pos_quaternion(self):
        return self._inner.translation_array(), self._inner.quaternion_array()

    def mat_var_x_arb_vec(self, arb_vec, tan_var_vec, frame: str = "bframe") -> np.ndarray:
        return _as_array(
            self._inner.mat_var_x_arb_vec(
                _as_array(arb_vec).tolist(), _as_array(tan_var_vec).tolist(), frame
            )
        )

    def mat_var_x_arb_vec_jacob(self, arb_vec, frame: str = "bframe") -> np.ndarray:
        return _as_array(self._inner.mat_var_x_arb_vec_jacob(_as_array(arb_vec).tolist(), frame))

    def __repr__(self) -> str:
        return f"SE3(rot={self.rot()}, pos={self.pos()}, LIB='numpy')"


class CMVector:
    def __init__(self, vecs):
        array = _as_array(vecs)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise TypeError("CMVector expects a 1D or 2D numpy-compatible array")
        self._vecs = array
        self._n, self._dim = array.shape
        self._len = array.size

    @staticmethod
    def set_cmvecs(cm_vecs) -> "CMVector":
        array = _as_array(cm_vecs)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise TypeError("CMVector.set_cmvecs expects a 1D or 2D numpy-compatible array")
        return CMVector(array * _factorial_scales(array.shape[0])[:, None])

    def vecs(self) -> np.ndarray:
        return self._vecs.copy()

    def cm_vecs(self) -> np.ndarray:
        return self._vecs / _factorial_scales(self._n)[:, None]

    def vec(self) -> np.ndarray:
        return self._vecs.reshape(-1).copy()

    def cm_vec(self) -> np.ndarray:
        return self.cm_vecs().reshape(-1)

    def __repr__(self) -> str:
        return f"CMVector(n={self._n}, dim={self._dim}, len={self._len})\n{self.cm_vecs()}"


class _TypedCMTM:
    _GROUP = None

    def __init__(self, elem_mat=None, elem_vecs=None, *, _inner=None):
        if _inner is None:
            if elem_mat is None:
                _inner = _RawCMTM.eye(self._GROUP, 1)
            else:
                native_elem = _as_native_group(self._GROUP, elem_mat)
                native_vecs = None
                if elem_vecs is not None:
                    native_vecs = _ensure_vec_matrix(elem_vecs, self._dof()).tolist()
                _inner = _RawCMTM(native_elem, native_vecs)
        self._inner = _inner

    @classmethod
    def _wrap(cls, inner) -> "_TypedCMTM":
        return cls(_inner=inner)

    @classmethod
    def _dof(cls) -> int:
        return cls._GROUP.dof()

    @classmethod
    def _adj_dim(cls) -> int:
        return cls._GROUP.mat_adj_size()

    @classmethod
    def _mat_dim(cls) -> int:
        return cls._GROUP.mat_size()

    @classmethod
    def _ensure_cmvector(cls, value, order: int | None = None) -> CMVector:
        if isinstance(value, CMVector):
            return value
        return CMVector(_ensure_vec_matrix(value, cls._adj_dim(), order))

    @classmethod
    def eye(cls, output_order: int = 3) -> "_TypedCMTM":
        return cls._wrap(_RawCMTM.eye(cls._GROUP, output_order))

    @classmethod
    def rand(cls, output_order: int = 3) -> "_TypedCMTM":
        return cls._wrap(_RawCMTM.rand(cls._GROUP, output_order))

    @classmethod
    def set_mat(cls, mat) -> "_TypedCMTM":
        return cls._wrap(_RawCMTM.set_mat(cls._GROUP, _as_array(mat).tolist()))

    @classmethod
    def set_mat_adj(cls, mat) -> "_TypedCMTM":
        return cls._wrap(_RawCMTM.set_mat_adj(cls._GROUP, _as_array(mat).tolist()))

    @classmethod
    def hat(cls, vecs) -> np.ndarray:
        return _as_array(_RawCMTM.hat(cls._GROUP, _ensure_vec_matrix(vecs, cls._dof()).tolist()))

    @classmethod
    def hat_adj(cls, vecs) -> np.ndarray:
        return _as_array(_RawCMTM.hat_adj(cls._GROUP, _ensure_vec_matrix(vecs, cls._adj_dim()).tolist()))

    @classmethod
    def hat_commute(cls, vecs) -> np.ndarray:
        rows = _ensure_vec_matrix(vecs, cls._dof())
        return _toeplitz_blocks([cls._GROUP.hat_commute(row.tolist()) for row in rows])

    @classmethod
    def hat_commute_adj(cls, vecs) -> np.ndarray:
        return _as_array(
            _RawCMTM.hat_commute_adj(cls._GROUP, _ensure_vec_matrix(vecs, cls._adj_dim()).tolist())
        )

    @classmethod
    def hat_cm(cls, vecs) -> np.ndarray:
        cmvec = cls._ensure_cmvector(vecs)
        return _toeplitz_blocks([cls._GROUP.hat(row.tolist()) for row in cmvec.cm_vecs()])

    @classmethod
    def hat_cm_adj(cls, vecs) -> np.ndarray:
        cmvec = cls._ensure_cmvector(vecs)
        return _toeplitz_blocks([cls._GROUP.hat_adj(row.tolist()) for row in cmvec.cm_vecs()])

    @classmethod
    def hat_cm_commute(cls, vecs) -> np.ndarray:
        cmvec = cls._ensure_cmvector(vecs)
        return _toeplitz_blocks([cls._GROUP.hat_commute(row.tolist()) for row in cmvec.cm_vecs()])

    @classmethod
    def hat_cm_commute_adj(cls, vecs) -> np.ndarray:
        cmvec = cls._ensure_cmvector(vecs)
        return _toeplitz_blocks([cls._GROUP.hat_commute_adj(row.tolist()) for row in cmvec.cm_vecs()])

    @classmethod
    def vee(cls, hat_mat) -> np.ndarray:
        return _as_array(_RawCMTM.vee(cls._GROUP, _as_array(hat_mat).tolist()))

    @classmethod
    def vee_adj(cls, hat_mat) -> np.ndarray:
        return _as_array(_RawCMTM.vee_adj(cls._GROUP, _as_array(hat_mat).tolist()))

    @classmethod
    def vee_cm(cls, hat_mat) -> CMVector:
        return _vee_cm(cls._GROUP, hat_mat, adjoint=False)

    @classmethod
    def vee_cm_adj(cls, hat_mat) -> CMVector:
        return _vee_cm(cls._GROUP, hat_mat, adjoint=True)

    @classmethod
    def sub_vec(cls, left: "_TypedCMTM", right: "_TypedCMTM", frame: str = "bframe") -> np.ndarray:
        if left.order() != right.order():
            raise TypeError("Left operand should be same order in right operand")

        left_elem = cls._GROUP.set_mat(left.elem_mat().tolist())
        right_elem = cls._GROUP.set_mat(right.elem_mat().tolist())
        head = _as_array(cls._GROUP.sub_tan_vec(left_elem, right_elem, frame))
        tail = (right.vecs() - left.vecs()).reshape(-1)
        return np.concatenate((head, tail))

    @classmethod
    def sub_tan_vec(
        cls, left: "_TypedCMTM", right: "_TypedCMTM", frame: str = "bframe"
    ) -> np.ndarray:
        if left.order() != right.order():
            raise TypeError("Left operand should be same order in right operand")

        if frame == "bframe":
            delta = right.mat() - left.mat()
            return cls.vee(left.mat_inv() @ delta).reshape(-1)
        if frame == "fframe":
            delta = right.mat() - left.mat()
            return cls.vee(delta @ left.mat_inv()).reshape(-1)
        raise ValueError("frame must be 'bframe' or 'fframe'")

    def order(self) -> int:
        return len(self._inner.vecs(None)) + 1

    def size(self) -> int:
        return self._inner.size()

    def adj_size(self) -> int:
        return self._inner.adj_size()

    def elem_mat(self) -> np.ndarray:
        return _as_array(self._inner.elem_mat())

    def elem_vecs(self, index: int):
        values = self._inner.elem_vecs(index)
        return None if values is None else _as_array(values)

    def vecs(self, output_order=None) -> np.ndarray:
        values = self._inner.vecs(output_order)
        if not values:
            return _empty_vecs(self._dof())
        return _as_array(values)

    def cmvecs(self) -> CMVector:
        return CMVector(self.vecs())

    def vecs_flatten(self, output_order=None) -> np.ndarray:
        values = self._inner.vecs_flatten(output_order)
        return _as_array(values)

    def mat(self, output_order=None) -> np.ndarray:
        return _as_array(self._inner.mat(output_order))

    def mat_adj(self, output_order=None) -> np.ndarray:
        return _as_array(self._inner.mat_adj(output_order))

    def mat_inv(self, output_order=None) -> np.ndarray:
        return _as_array(self._inner.mat_inv(output_order))

    def mat_inv_adj(self, output_order=None) -> np.ndarray:
        return _as_array(self._inner.mat_inv_adj(output_order))

    def tangent_mat(self, output_order=None) -> np.ndarray:
        return _as_array(self._inner.tangent_mat(output_order))

    def tangent_mat_inv(self, output_order=None) -> np.ndarray:
        return _as_array(self._inner.tangent_mat_inv(output_order))

    def tangent_mat_cm(self, output_order=None) -> np.ndarray:
        return _as_array(self._inner.tangent_mat_cm(output_order))

    def tangent_mat_cm_inv(self, output_order=None) -> np.ndarray:
        return _as_array(self._inner.tangent_mat_cm_inv(output_order))

    def matrix(self) -> np.ndarray:
        return self.mat()

    def inv(self) -> "_TypedCMTM":
        return self._wrap(self._inner.inv())

    def inverse(self) -> "_TypedCMTM":
        return self.inv()

    def compose(self, other: "_TypedCMTM") -> "_TypedCMTM":
        return self._wrap(self._inner.compose(other._inner))

    def __mul__(self, other: "_TypedCMTM") -> "_TypedCMTM":
        return self.compose(other)

    def __matmul__(self, other):
        if isinstance(other, _TypedCMTM):
            return self.compose(other)
        if isinstance(other, CMVector):
            result = self.mat_adj() @ other.cm_vec()
            return CMVector.set_cmvecs(result.reshape(self.order(), self._adj_dim()))
        return self.mat() @ _as_array(other)

    def mat_var_x_arb_vec(self, arb_vec, tan_var_vec, frame: str = "bframe") -> CMVector:
        arb = self._ensure_cmvector(arb_vec, self.order())
        tan = self._ensure_cmvector(tan_var_vec, self.order())

        if frame == "bframe":
            result = self.mat_adj() @ self.__class__.hat_cm_commute_adj(arb) @ tan.cm_vec()
            return CMVector.set_cmvecs(result.reshape(self.order(), self._adj_dim()))
        if frame == "fframe":
            result = self.__class__.hat_cm_commute_adj(self @ arb) @ tan.cm_vec()
            return CMVector.set_cmvecs(result.reshape(self.order(), self._adj_dim()))
        raise ValueError("frame must be 'bframe' or 'fframe'")

    def mat_var_x_arb_vec_jacob(self, arb_vec, frame: str = "bframe") -> np.ndarray:
        arb = self._ensure_cmvector(arb_vec, self.order())
        if frame == "bframe":
            return self.mat_adj() @ self.__class__.hat_cm_commute_adj(arb)
        if frame == "fframe":
            return self.__class__.hat_cm_commute_adj(self @ arb)
        raise ValueError("frame must be 'bframe' or 'fframe'")

    def mat_inv_var_x_arb_vec_jacob(self, arb_vec, frame: str = "bframe") -> np.ndarray:
        arb = self._ensure_cmvector(arb_vec, self.order())
        if frame == "bframe":
            return -self.__class__.hat_cm_commute_adj(self.inv() @ arb)
        raise NotImplementedError("Not implemented for fframe")

    def apply_omega(self, omega) -> np.ndarray:
        return _as_array(self._inner.apply_omega(omega))

    def apply_twist(self, twist) -> np.ndarray:
        return _as_array(self._inner.apply_twist(twist))

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"\telem_mat=\n{self.elem_mat()},\n"
            f"\telem_vecs=\n{self.vecs()},\n"
            "\tLIB='numpy'\n)"
        )


class SO3CMTM(_TypedCMTM):
    _GROUP = SO3

    @classmethod
    def from_so3(cls, rotation: SO3) -> "SO3CMTM":
        return cls(_inner=_RawCMTM.from_so3(_as_native_so3(rotation)))

    @classmethod
    def from_so3_with_derivatives(cls, rotation: SO3, elem_vecs) -> "SO3CMTM":
        return cls(
            _inner=_RawCMTM.from_so3_with_derivatives(
                _as_native_so3(rotation), _ensure_vec_matrix(elem_vecs, 3).tolist()
            )
        )


class SE3CMTM(_TypedCMTM):
    _GROUP = SE3

    @classmethod
    def from_se3(cls, transform: SE3) -> "SE3CMTM":
        return cls(_inner=_RawCMTM.from_se3(_as_native_se3(transform)))

    @classmethod
    def from_se3_with_derivatives(cls, transform: SE3, elem_vecs) -> "SE3CMTM":
        return cls(
            _inner=_RawCMTM.from_se3_with_derivatives(
                _as_native_se3(transform), _ensure_vec_matrix(elem_vecs, 6).tolist()
            )
        )


def _resolve_group_class(group):
    name = group if isinstance(group, str) else getattr(group, "__name__", None)
    if group in (SO3, SO3CMTM) or name == "SO3":
        return SO3CMTM
    if group in (SE3, SE3CMTM) or name == "SE3":
        return SE3CMTM
    raise TypeError("group must be SO3 or SE3")


class CMTM:
    def __new__(cls, element=None, elem_vecs=None):
        if cls is CMTM:
            raise TypeError("Use CMTM[SO3] or CMTM[SE3] for construction")
        return super().__new__(cls)

    @classmethod
    def __class_getitem__(cls, group):
        return _resolve_group_class(group)

    @staticmethod
    def eye(group, output_order: int = 3):
        return _resolve_group_class(group).eye(output_order)

    @staticmethod
    def rand(group, output_order: int = 3):
        return _resolve_group_class(group).rand(output_order)

    @staticmethod
    def set_mat(group, mat):
        return _resolve_group_class(group).set_mat(mat)

    @staticmethod
    def set_mat_adj(group, mat):
        return _resolve_group_class(group).set_mat_adj(mat)

    @staticmethod
    def hat(group, vecs):
        return _resolve_group_class(group).hat(vecs)

    @staticmethod
    def hat_adj(group, vecs):
        return _resolve_group_class(group).hat_adj(vecs)

    @staticmethod
    def hat_commute(group, vecs):
        return _resolve_group_class(group).hat_commute(vecs)

    @staticmethod
    def hat_commute_adj(group, vecs):
        return _resolve_group_class(group).hat_commute_adj(vecs)

    @staticmethod
    def hat_cm(group, vecs):
        return _resolve_group_class(group).hat_cm(vecs)

    @staticmethod
    def hat_cm_adj(group, vecs):
        return _resolve_group_class(group).hat_cm_adj(vecs)

    @staticmethod
    def hat_cm_commute(group, vecs):
        return _resolve_group_class(group).hat_cm_commute(vecs)

    @staticmethod
    def hat_cm_commute_adj(group, vecs):
        return _resolve_group_class(group).hat_cm_commute_adj(vecs)

    @staticmethod
    def vee(group, hat_mat):
        return _resolve_group_class(group).vee(hat_mat)

    @staticmethod
    def vee_adj(group, hat_mat):
        return _resolve_group_class(group).vee_adj(hat_mat)

    @staticmethod
    def vee_cm(group, hat_mat):
        return _resolve_group_class(group).vee_cm(hat_mat)

    @staticmethod
    def vee_cm_adj(group, hat_mat):
        return _resolve_group_class(group).vee_cm_adj(hat_mat)

    @staticmethod
    def sub_vec(left, right, frame: str = "bframe"):
        return left.__class__.sub_vec(left, right, frame)

    @staticmethod
    def sub_tan_vec(left, right, frame: str = "bframe"):
        return left.__class__.sub_tan_vec(left, right, frame)

class SO3wrench:
    def __init__(self, r=None, LIB: str = "numpy"):
        _ensure_numpy_lib(LIB)
        self._rot = np.eye(3, dtype=float) if r is None else _as_array(r)
        self._lib = "numpy"

    @property
    def lib(self) -> str:
        return self._lib

    @staticmethod
    def dof() -> int:
        return 3

    @staticmethod
    def mat_size() -> int:
        return 3

    @staticmethod
    def mat_adj_size() -> int:
        return 3

    def mat(self) -> np.ndarray:
        return self._rot.copy()

    def mat_adj(self) -> np.ndarray:
        return self.mat()

    def mat_inv(self) -> np.ndarray:
        return self._rot.transpose()

    def mat_inv_adj(self) -> np.ndarray:
        return self.mat_inv()

    def inv(self) -> "SO3wrench":
        return SO3wrench(self._rot.transpose(), self.lib)

    @staticmethod
    def set_mat(mat=None, LIB: str = "numpy") -> "SO3wrench":
        return SO3wrench(np.eye(3, dtype=float) if mat is None else mat, LIB)

    @staticmethod
    def hat(vec, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        return -_as_array(SO3.hat(_as_array(vec).tolist()))

    @staticmethod
    def hat_commute(vec, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        return _as_array(SO3.hat(_as_array(vec).tolist()))

    @staticmethod
    def hat_adj(vec, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        return _as_array(SO3.hat_adj(_as_array(vec).tolist()))

    @staticmethod
    def hat_commute_adj(vec, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        return _as_array(SO3.hat_commute_adj(_as_array(vec).tolist()))

    @staticmethod
    def vee(mat, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        return _as_array(SO3.vee(_as_array(mat).tolist()))

    @staticmethod
    def vee_adj(mat, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        return _as_array(SO3.vee_adj(_as_array(mat).tolist()))

    @staticmethod
    def exp(vec, a: float = 1.0, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        return _as_array(SO3.exp(_as_array(vec).tolist(), a)).transpose()

    @staticmethod
    def exp_integ(vec, a: float = 1.0, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        return _as_array(SO3.exp_integ(_as_array(vec).tolist(), a)).transpose()

    @classmethod
    def rand(cls, LIB: str = "numpy") -> "SO3wrench":
        _ensure_numpy_lib(LIB)
        return cls(_as_array(SO3.rand().mat()))

    def __matmul__(self, other):
        return self.mat() @ _as_array(other)

    def __repr__(self) -> str:
        return f"SO3wrench(\nrot=\n{self._rot},\nLIB='{self.lib}')"


class SO3inertia(SO3wrench):
    @staticmethod
    def dof() -> int:
        return 6

    @staticmethod
    def hat(vec, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        vec = _as_array(vec)
        mat = np.zeros((3, 3), dtype=float)
        mat[0, 0] = vec[0]
        mat[0, 1] = vec[5]
        mat[0, 2] = vec[4]
        mat[1, 0] = vec[5]
        mat[1, 1] = vec[1]
        mat[1, 2] = vec[3]
        mat[2, 0] = vec[4]
        mat[2, 1] = vec[3]
        mat[2, 2] = vec[2]
        return mat

    @staticmethod
    def hat_commute(vec, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        vec = _as_array(vec)
        mat = np.zeros((3, 6), dtype=float)
        mat[0, 0] = vec[0]
        mat[1, 1] = vec[1]
        mat[2, 2] = vec[2]
        mat[1, 5] = vec[0]
        mat[2, 4] = vec[0]
        mat[2, 3] = vec[1]
        mat[0, 5] = vec[1]
        mat[0, 4] = vec[2]
        mat[1, 3] = vec[2]
        return mat


class SE3wrench:
    def __init__(self, rot=None, pos=None, LIB: str = "numpy"):
        _ensure_numpy_lib(LIB)
        self._rot = np.eye(3, dtype=float) if rot is None else _as_array(rot)
        self._pos = np.zeros(3, dtype=float) if pos is None else _as_array(pos)
        self._lib = "numpy"

    @property
    def lib(self) -> str:
        return self._lib

    @staticmethod
    def dof() -> int:
        return 6

    @staticmethod
    def mat_size() -> int:
        return 4

    @staticmethod
    def mat_adj_size() -> int:
        return 6

    @staticmethod
    def set_mat(mat=None, LIB: str = "numpy") -> "SE3wrench":
        _ensure_numpy_lib(LIB)
        matrix = np.eye(4, dtype=float) if mat is None else _as_array(mat)
        return SE3wrench(matrix[0:3, 0:3], matrix[0:3, 3], LIB)

    def mat(self) -> np.ndarray:
        return _as_array(SE3(self._rot.tolist(), self._pos.tolist()).mat())

    def mat_inv(self) -> np.ndarray:
        return _as_array(SE3(self._rot.tolist(), self._pos.tolist()).mat_inv())

    def mat_adj(self) -> np.ndarray:
        mat = np.zeros((6, 6), dtype=float)
        hat_pos = _as_array(SO3.hat(self._pos.tolist()))
        mat[0:3, 0:3] = self._rot
        mat[0:3, 3:6] = hat_pos @ self._rot
        mat[3:6, 3:6] = self._rot
        return mat

    def mat_inv_adj(self) -> np.ndarray:
        mat = np.zeros((6, 6), dtype=float)
        hat_pos = _as_array(SO3.hat(self._pos.tolist()))
        rot_t = self._rot.transpose()
        mat[0:3, 0:3] = rot_t
        mat[0:3, 3:6] = -rot_t @ hat_pos
        mat[3:6, 3:6] = rot_t
        return mat

    def inv(self) -> "SE3wrench":
        rot_t = self._rot.transpose()
        return SE3wrench(rot_t, -rot_t @ self._pos, self.lib)

    @classmethod
    def change_class(cls, other) -> "SE3wrench":
        if isinstance(other, SE3wrench):
            return cls(other._rot, other._pos, other.lib)
        return cls(_as_array(other.rot()), _as_array(other.pos()), getattr(other, "lib", "numpy"))

    def rot(self) -> np.ndarray:
        return self._rot.copy()

    def pos(self) -> np.ndarray:
        return self._pos.copy()

    @staticmethod
    def exp(vec, a: float = 1.0, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        return _as_array(SE3.exp_adj(_as_array(vec).tolist(), a)).transpose()

    @staticmethod
    def exp_integ(vec, a: float = 1.0, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        return _as_array(SE3.exp_integ_adj(_as_array(vec).tolist(), a)).transpose()

    @staticmethod
    def hat_adj(vec, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        vec = _as_array(vec)
        w_hat = _as_array(SO3.hat(vec[:3].tolist()))
        v_hat = _as_array(SO3.hat(vec[3:].tolist()))
        mat = np.zeros((6, 6), dtype=float)
        mat[0:3, 0:3] = w_hat
        mat[0:3, 3:6] = v_hat
        mat[3:6, 3:6] = w_hat
        return mat

    @staticmethod
    def hat_commute(vec, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        vec = _as_array(vec)
        mat = np.zeros((4, 6), dtype=float)
        mat[0:3, 0:3] = _as_array(SO3.hat(vec[:3].tolist()))
        return -mat

    @staticmethod
    def hat_commute_adj(vec, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        vec = _as_array(vec)
        mat = np.zeros((6, 6), dtype=float)
        mat[0:3, 0:3] = _as_array(SO3.hat(vec[:3].tolist()))
        mat[0:3, 3:6] = _as_array(SO3.hat(vec[3:].tolist()))
        mat[3:6, 0:3] = _as_array(SO3.hat(vec[3:].tolist()))
        return -mat

    @classmethod
    def rand(cls, LIB: str = "numpy") -> "SE3wrench":
        _ensure_numpy_lib(LIB)
        transform = SE3.rand()
        return cls(_as_array(transform.rot()), _as_array(transform.pos()))

    def mat_var_x_arb_vec(self, arb_vec, tan_var_vec, frame: str = "bframe") -> np.ndarray:
        arb_vec = _as_array(arb_vec)
        tan_var_vec = _as_array(tan_var_vec)
        if frame == "bframe":
            return self.mat_adj() @ self.hat_commute_adj(arb_vec) @ tan_var_vec
        if frame == "fframe":
            return self.hat_commute_adj(self.mat_adj() @ arb_vec) @ tan_var_vec
        raise ValueError("frame must be 'bframe' or 'fframe'")

    def mat_var_x_arb_vec_jacob(self, arb_vec, frame: str = "bframe") -> np.ndarray:
        arb_vec = _as_array(arb_vec)
        if frame == "bframe":
            return self.mat_adj() @ self.hat_commute_adj(arb_vec)
        if frame == "fframe":
            return self.hat_commute_adj(self.mat_adj() @ arb_vec)
        raise ValueError("frame must be 'bframe' or 'fframe'")

    def compose(self, other: "SE3wrench") -> "SE3wrench":
        if not isinstance(other, SE3wrench):
            raise TypeError("other must be SE3wrench")
        rot, pos = SE3.se3_mul(self._rot, self._pos, other._rot, other._pos)
        return SE3wrench(rot, pos, self.lib)

    def __mul__(self, other: "SE3wrench") -> "SE3wrench":
        return self.compose(other)

    def __matmul__(self, other):
        if isinstance(other, SE3wrench):
            return self.compose(other)
        matrix = _as_array(other)
        if matrix.shape == (6,):
            return self.mat_adj() @ matrix
        if matrix.shape == (6, 6):
            return self.mat_adj() @ matrix
        if matrix.shape == (4, 4):
            return self.mat() @ matrix
        raise TypeError("Right operand should be SE3wrench or a 6D vector or a 4x4/6x6 matrix")

    def __repr__(self) -> str:
        return f"SE3wrench(\nrot=\n{self._rot},\npos=\n{self._pos},\nLIB='{self.lib}')"


class SE3inertia:
    def __init__(self, rot=None, pos=None, LIB: str = "numpy"):
        _ensure_numpy_lib(LIB)
        self._rot = np.eye(3, dtype=float) if rot is None else _as_array(rot)
        self._pos = np.zeros(3, dtype=float) if pos is None else _as_array(pos)
        self._lib = "numpy"

    @property
    def lib(self) -> str:
        return self._lib

    @staticmethod
    def dof() -> int:
        return 10

    @staticmethod
    def mat_size() -> int:
        return 6

    @staticmethod
    def mat_adj_size() -> int:
        return 6

    @staticmethod
    def hat(vec, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        vec = _as_array(vec)
        mat = np.zeros((6, 6), dtype=float)
        mpg = vec[1:4]
        mat[0:3, 0:3] = SO3inertia.hat(vec[4:10], LIB)
        mat[0:3, 3:6] = SO3wrench.hat(mpg, LIB)
        mat[3:6, 0:3] = _as_array(SO3.hat(mpg.tolist()))
        mat[3:6, 3:6] = vec[0] * np.identity(3)
        return mat

    @staticmethod
    def hat_commute(vec, LIB: str = "numpy") -> np.ndarray:
        _ensure_numpy_lib(LIB)
        vec = _as_array(vec)
        mat = np.zeros((6, 10), dtype=float)
        v = vec[3:6]
        w = vec[0:3]
        mat[3:6, 0] = v
        mat[0:3, 1:4] = SO3wrench.hat_commute(v, LIB)
        mat[3:6, 1:4] = _as_array(SO3.hat_commute(w.tolist()))
        mat[0:3, 4:10] = SO3inertia.hat_commute(w, LIB)
        return mat

    def __repr__(self) -> str:
        return f"SE3inertia(\nrot=\n{self._rot},\npos=\n{self._pos},\nLIB='{self.lib}')"


__all__ = [
    "SO3",
    "SE3",
    "SO3wrench",
    "SO3inertia",
    "SE3wrench",
    "SE3inertia",
    "CMTM",
    "SO3CMTM",
    "SE3CMTM",
    "CMVector",
]
