use mathroborust::Se3;
use mathroborust::lie::{LieGroup, matrix_to_array};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

use crate::buffer::{
    ensure_numpy, mat_to_numpy, s_matrix6_to_array, vec_to_numpy, write_mat3, write_mat4,
    write_mat6, write_vec3, write_vec4, write_vec6,
};
use crate::convert::{matrix4_from_array, matrix6_from_array};
use crate::py_so3::PySo3;

#[pyclass(name = "SE3")]
#[derive(Clone)]
pub struct PySe3 {
    pub(crate) inner: Se3,
}

#[pymethods]
impl PySe3 {
    #[new]
    #[pyo3(signature = (rotation=None, position=None, lib=None))]
    pub fn new(
        rotation: Option<[[f64; 3]; 3]>,
        position: Option<[f64; 3]>,
        lib: Option<&str>,
    ) -> PyResult<Self> {
        ensure_numpy(lib)?;
        let rotation = rotation.map(mathroborust::So3::from_matrix);
        let position = position.unwrap_or([0.0, 0.0, 0.0]);
        match rotation {
            Some(rotation) => Ok(Self {
                inner: Se3::from_parts(rotation, position),
            }),
            None => Ok(Self {
                inner: Se3::from_parts(mathroborust::So3::identity(), position),
            }),
        }
    }

    #[getter]
    pub fn lib(&self) -> &'static str {
        "numpy"
    }

    #[staticmethod]
    pub fn dof() -> usize {
        Se3::dof()
    }

    #[staticmethod]
    pub fn mat_size() -> usize {
        Se3::mat_size()
    }

    #[staticmethod]
    pub fn mat_adj_size() -> usize {
        Se3::mat_adj_size()
    }

    #[staticmethod]
    pub fn from_axis_angle_translation(axis: [f64; 3], angle: f64, translation: [f64; 3]) -> Self {
        Self {
            inner: Se3::from_axis_angle_translation(axis, angle, translation),
        }
    }

    #[staticmethod]
    pub fn from_parts(rotation: &PySo3, translation: [f64; 3]) -> Self {
        Self {
            inner: Se3::from_parts(rotation.inner.clone(), translation),
        }
    }

    #[staticmethod]
    pub fn from_matrix(matrix: [[f64; 4]; 4]) -> Self {
        Self {
            inner: Se3::from_matrix(matrix),
        }
    }

    #[staticmethod]
    pub fn set_pos_quaternion(position: [f64; 3], quaternion: [f64; 4]) -> Self {
        Self {
            inner: Se3::from_pos_quaternion(position, quaternion),
        }
    }

    #[staticmethod]
    pub fn set_mat_adj(matrix: [[f64; 6]; 6]) -> Self {
        Self {
            inner: Se3::from_adjoint_matrix(matrix),
        }
    }

    #[staticmethod]
    pub fn change_class(other: &PySe3) -> Self {
        other.clone()
    }

    #[staticmethod]
    pub fn hat(twist: [f64; 6]) -> [[f64; 4]; 4] {
        Se3::hat(twist)
    }

    #[staticmethod]
    pub fn hat_array<'py>(py: Python<'py>, twist: [f64; 6]) -> Bound<'py, PyAny> {
        mat_to_numpy(py, Se3::hat(twist)).into_any()
    }

    #[staticmethod]
    pub fn hat_into(py: Python<'_>, twist: [f64; 6], out: PyBuffer<f64>) -> PyResult<()> {
        write_mat4(py, out, Se3::hat(twist))
    }

    #[staticmethod]
    pub fn hat_commute(twist: [f64; 6]) -> [[f64; 6]; 4] {
        Se3::hat_commute(twist)
    }

    #[staticmethod]
    pub fn hat_adj(twist: [f64; 6]) -> [[f64; 6]; 6] {
        Se3::hat_adj(twist)
    }

    #[staticmethod]
    pub fn hat_adj_array<'py>(py: Python<'py>, twist: [f64; 6]) -> Bound<'py, PyAny> {
        mat_to_numpy(py, Se3::hat_adj(twist)).into_any()
    }

    #[staticmethod]
    pub fn hat_adj_into(py: Python<'_>, twist: [f64; 6], out: PyBuffer<f64>) -> PyResult<()> {
        write_mat6(py, out, Se3::hat_adj(twist))
    }

    #[staticmethod]
    pub fn hat_commute_adj(twist: [f64; 6]) -> [[f64; 6]; 6] {
        Se3::hat_commute_adj(twist)
    }

    #[staticmethod]
    pub fn hat_commute_adj_array<'py>(py: Python<'py>, twist: [f64; 6]) -> Bound<'py, PyAny> {
        mat_to_numpy(py, Se3::hat_commute_adj(twist)).into_any()
    }

    #[staticmethod]
    pub fn hat_commute_adj_into(
        py: Python<'_>,
        twist: [f64; 6],
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        write_mat6(py, out, Se3::hat_commute_adj(twist))
    }

    #[staticmethod]
    pub fn vee(matrix: [[f64; 4]; 4]) -> [f64; 6] {
        Se3::vee(matrix)
    }

    #[staticmethod]
    pub fn vee_array<'py>(py: Python<'py>, matrix: [[f64; 4]; 4]) -> Bound<'py, PyAny> {
        vec_to_numpy(py, Se3::vee(matrix)).into_any()
    }

    #[staticmethod]
    pub fn vee_into(py: Python<'_>, matrix: [[f64; 4]; 4], out: PyBuffer<f64>) -> PyResult<()> {
        write_vec6(py, out, Se3::vee(matrix))
    }

    #[staticmethod]
    pub fn vee_adj(matrix: [[f64; 6]; 6]) -> [f64; 6] {
        Se3::vee_adj(matrix)
    }

    #[staticmethod]
    pub fn vee_adj_array<'py>(py: Python<'py>, matrix: [[f64; 6]; 6]) -> Bound<'py, PyAny> {
        vec_to_numpy(py, Se3::vee_adj(matrix)).into_any()
    }

    #[staticmethod]
    pub fn vee_adj_into(
        py: Python<'_>,
        matrix: [[f64; 6]; 6],
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        write_vec6(py, out, Se3::vee_adj(matrix))
    }

    #[staticmethod]
    #[pyo3(signature = (twist, a=None))]
    pub fn exp(twist: [f64; 6], a: Option<f64>) -> [[f64; 4]; 4] {
        Se3::exp(twist, a)
    }

    #[staticmethod]
    #[pyo3(signature = (twist, a=None))]
    pub fn exp_array<'py>(py: Python<'py>, twist: [f64; 6], a: Option<f64>) -> Bound<'py, PyAny> {
        mat_to_numpy(py, Se3::exp(twist, a)).into_any()
    }

    #[staticmethod]
    #[pyo3(signature = (twist, a=None))]
    pub fn exp_integ(twist: [f64; 6], a: Option<f64>) -> [[f64; 4]; 4] {
        Se3::exp_integ(twist, a)
    }

    #[staticmethod]
    #[pyo3(signature = (twist, a=None))]
    pub fn exp_integ_array<'py>(
        py: Python<'py>,
        twist: [f64; 6],
        a: Option<f64>,
    ) -> Bound<'py, PyAny> {
        mat_to_numpy(py, Se3::exp_integ(twist, a)).into_any()
    }

    #[staticmethod]
    #[pyo3(signature = (twist, out, a=None))]
    pub fn exp_integ_into(
        py: Python<'_>,
        twist: [f64; 6],
        out: PyBuffer<f64>,
        a: Option<f64>,
    ) -> PyResult<()> {
        write_mat4(py, out, Se3::exp_integ(twist, a))
    }

    #[staticmethod]
    #[pyo3(signature = (twist, a=None))]
    pub fn exp_adj(twist: [f64; 6], a: Option<f64>) -> [[f64; 6]; 6] {
        Se3::exp_adj(twist, a)
    }

    #[staticmethod]
    #[pyo3(signature = (twist, a=None))]
    pub fn exp_adj_array<'py>(
        py: Python<'py>,
        twist: [f64; 6],
        a: Option<f64>,
    ) -> Bound<'py, PyAny> {
        mat_to_numpy(py, Se3::exp_adj(twist, a)).into_any()
    }

    #[staticmethod]
    #[pyo3(signature = (twist, out, a=None))]
    pub fn exp_adj_into(
        py: Python<'_>,
        twist: [f64; 6],
        out: PyBuffer<f64>,
        a: Option<f64>,
    ) -> PyResult<()> {
        write_mat6(py, out, Se3::exp_adj(twist, a))
    }

    #[staticmethod]
    #[pyo3(signature = (twist, a=None))]
    pub fn exp_integ_adj(twist: [f64; 6], a: Option<f64>) -> [[f64; 6]; 6] {
        Se3::exp_integ_adj(twist, a)
    }

    #[staticmethod]
    #[pyo3(signature = (twist, a=None))]
    pub fn exp_integ_adj_array<'py>(
        py: Python<'py>,
        twist: [f64; 6],
        a: Option<f64>,
    ) -> Bound<'py, PyAny> {
        mat_to_numpy(py, Se3::exp_integ_adj(twist, a)).into_any()
    }

    #[staticmethod]
    #[pyo3(signature = (twist, out, a=None))]
    pub fn exp_integ_adj_into(
        py: Python<'_>,
        twist: [f64; 6],
        out: PyBuffer<f64>,
        a: Option<f64>,
    ) -> PyResult<()> {
        write_mat6(py, out, Se3::exp_integ_adj(twist, a))
    }

    #[staticmethod]
    #[pyo3(signature = (twist, out, a=None))]
    pub fn exp_into(
        py: Python<'_>,
        twist: [f64; 6],
        out: PyBuffer<f64>,
        a: Option<f64>,
    ) -> PyResult<()> {
        write_mat4(py, out, Se3::exp(twist, a))
    }

    #[staticmethod]
    #[pyo3(signature = (lib=None))]
    pub fn rand(lib: Option<&str>) -> PyResult<Self> {
        ensure_numpy(lib)?;
        Ok(Self {
            inner: Se3::from_axis_angle_translation(
                [
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                ],
                1.0,
                [
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                    rand::random::<f64>(),
                ],
            ),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (left, right, frame="bframe"))]
    pub fn sub_tan_vec(left: &PySe3, right: &PySe3, frame: &str) -> [f64; 6] {
        Se3::sub_tan_vec(&left.inner, &right.inner, Some(frame))
    }

    #[staticmethod]
    pub fn se3_mul(
        left_rot: [[f64; 3]; 3],
        left_pos: [f64; 3],
        right_rot: [[f64; 3]; 3],
        right_pos: [f64; 3],
    ) -> ([[f64; 3]; 3], [f64; 3]) {
        let left = Se3::from_parts(mathroborust::So3::from_matrix(left_rot), left_pos);
        let right = Se3::from_parts(mathroborust::So3::from_matrix(right_rot), right_pos);
        let result = left.compose(&right);
        (result.rotation().to_matrix(), result.translation())
    }

    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        self.inner.apply(point)
    }

    pub fn apply_array<'py>(&self, py: Python<'py>, point: [f64; 3]) -> Bound<'py, PyAny> {
        vec_to_numpy(py, self.inner.apply(point)).into_any()
    }

    pub fn apply_into(&self, py: Python<'_>, point: [f64; 3], out: PyBuffer<f64>) -> PyResult<()> {
        write_vec3(py, out, self.inner.apply(point))
    }

    pub fn compose(&self, other: &PySe3) -> PySe3 {
        PySe3 {
            inner: self.inner.compose(&other.inner),
        }
    }

    #[pyo3(name = "__mul__")]
    pub fn mul(&self, other: &PySe3) -> PyResult<PySe3> {
        Ok(self.compose(other))
    }

    #[pyo3(name = "__matmul__")]
    pub fn matmul(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(other) = other.extract::<PyRef<'_, PySe3>>() {
            return Ok(Py::new(
                py,
                PySe3 {
                    inner: self.inner.compose(&other.inner),
                },
            )?
            .into_py(py));
        }

        if let Ok(point) = other.extract::<[f64; 3]>() {
            return Ok(self.inner.apply(point).into_py(py));
        }

        if let Ok(twist) = other.extract::<[f64; 6]>() {
            let result = self.inner.adjoint() * nalgebra::SVector::<f64, 6>::from_row_slice(&twist);
            let output: [f64; 6] = result.into();
            return Ok(output.into_py(py));
        }

        if let Ok(matrix) = other.extract::<[[f64; 4]; 4]>() {
            let result = self.inner.as_matrix() * matrix4_from_array(matrix);
            return Ok(matrix_to_array(&result).into_py(py));
        }

        if let Ok(matrix) = other.extract::<[[f64; 6]; 6]>() {
            let result = self.inner.adjoint() * matrix6_from_array(matrix);
            return Ok(s_matrix6_to_array(&result).into_py(py));
        }

        Err(PyTypeError::new_err(
            "Right operand should be SE3 or a 3D/6D vector or a 4x4/6x6 matrix",
        ))
    }

    pub fn inverse(&self) -> PySe3 {
        PySe3 {
            inner: self.inner.inverse(),
        }
    }

    pub fn inv(&self) -> PySe3 {
        self.inverse()
    }

    pub fn matrix(&self) -> [[f64; 4]; 4] {
        self.inner.to_matrix()
    }

    pub fn matrix_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        mat_to_numpy(py, self.inner.to_matrix()).into_any()
    }

    pub fn mat(&self) -> [[f64; 4]; 4] {
        self.inner.to_matrix()
    }

    pub fn mat_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        mat_to_numpy(py, self.inner.to_matrix()).into_any()
    }

    pub fn mat_into(&self, py: Python<'_>, out: PyBuffer<f64>) -> PyResult<()> {
        write_mat4(py, out, self.inner.to_matrix())
    }

    #[staticmethod]
    pub fn set_mat(matrix: [[f64; 4]; 4]) -> Self {
        Self {
            inner: Se3::from_matrix(matrix),
        }
    }

    #[staticmethod]
    pub fn eye() -> Self {
        Self {
            inner: Se3::identity(),
        }
    }

    pub fn mat_inv(&self) -> [[f64; 4]; 4] {
        self.inner.inverse().to_matrix()
    }

    pub fn mat_inv_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        mat_to_numpy(py, self.inner.inverse().to_matrix()).into_any()
    }

    pub fn mat_inv_into(&self, py: Python<'_>, out: PyBuffer<f64>) -> PyResult<()> {
        write_mat4(py, out, self.inner.inverse().to_matrix())
    }

    pub fn mat_adj(&self) -> [[f64; 6]; 6] {
        s_matrix6_to_array(&self.inner.adjoint())
    }

    pub fn mat_adj_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        mat_to_numpy(py, s_matrix6_to_array(&self.inner.adjoint())).into_any()
    }

    pub fn mat_adj_into(&self, py: Python<'_>, out: PyBuffer<f64>) -> PyResult<()> {
        write_mat6(py, out, s_matrix6_to_array(&self.inner.adjoint()))
    }

    pub fn mat_inv_adj(&self) -> [[f64; 6]; 6] {
        s_matrix6_to_array(&self.inner.inverse().adjoint())
    }

    pub fn mat_inv_adj_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        mat_to_numpy(py, s_matrix6_to_array(&self.inner.inverse().adjoint())).into_any()
    }

    pub fn mat_inv_adj_into(&self, py: Python<'_>, out: PyBuffer<f64>) -> PyResult<()> {
        write_mat6(py, out, s_matrix6_to_array(&self.inner.inverse().adjoint()))
    }

    pub fn translation(&self) -> [f64; 3] {
        self.inner.translation()
    }

    pub fn translation_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        vec_to_numpy(py, self.inner.translation()).into_any()
    }

    pub fn translation_into(&self, py: Python<'_>, out: PyBuffer<f64>) -> PyResult<()> {
        write_vec3(py, out, self.inner.translation())
    }

    pub fn position(&self) -> [f64; 3] {
        self.inner.translation()
    }

    pub fn pos(&self) -> [f64; 3] {
        self.inner.translation()
    }

    pub fn rotation(&self) -> PySo3 {
        PySo3 {
            inner: self.inner.rotation().clone(),
        }
    }

    pub fn rot(&self) -> [[f64; 3]; 3] {
        self.inner.rotation().to_matrix()
    }

    pub fn rot_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        mat_to_numpy(py, self.inner.rotation().to_matrix()).into_any()
    }

    pub fn rot_into(&self, py: Python<'_>, out: PyBuffer<f64>) -> PyResult<()> {
        write_mat3(py, out, self.inner.rotation().to_matrix())
    }

    pub fn pos_quaternion(&self) -> ([f64; 3], [f64; 4]) {
        self.inner.pos_quaternion()
    }

    pub fn quaternion_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        let (_, quaternion) = self.inner.pos_quaternion();
        vec_to_numpy(py, quaternion).into_any()
    }

    pub fn quaternion_into(&self, py: Python<'_>, out: PyBuffer<f64>) -> PyResult<()> {
        let (_, quaternion) = self.inner.pos_quaternion();
        write_vec4(py, out, quaternion)
    }

    #[pyo3(signature = (arb_vec, tan_var_vec, frame="bframe"))]
    pub fn mat_var_x_arb_vec(
        &self,
        arb_vec: [f64; 6],
        tan_var_vec: [f64; 6],
        frame: &str,
    ) -> [f64; 6] {
        self.inner
            .mat_var_x_arb_vec(arb_vec, tan_var_vec, Some(frame))
    }

    #[pyo3(signature = (arb_vec, frame="bframe"))]
    pub fn mat_var_x_arb_vec_jacob(&self, arb_vec: [f64; 6], frame: &str) -> [[f64; 6]; 6] {
        self.inner.mat_var_x_arb_vec_jacob(arb_vec, Some(frame))
    }

    pub fn __repr__(&self) -> String {
        format!(
            "SE3(rot={:?}, pos={:?}, LIB='numpy')",
            self.inner.rotation().to_matrix(),
            self.inner.translation()
        )
    }
}
