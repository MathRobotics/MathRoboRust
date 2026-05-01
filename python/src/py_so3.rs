use mathroborust::So3;
use mathroborust::lie::{LieGroup, matrix_to_array};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::buffer::{ensure_numpy, mat_to_numpy, vec_to_numpy, write_mat3, write_vec3, write_vec4};
use crate::convert::matrix3_from_array;

#[pyclass(name = "SO3")]
#[derive(Clone)]
pub struct PySo3 {
    pub(crate) inner: So3,
}

fn euler_matrix(euler: (f64, f64, f64), order: &str) -> PyResult<[[f64; 3]; 3]> {
    let (roll, pitch, yaw) = euler;
    let (cr, sr) = (roll.cos(), roll.sin());
    let (cp, sp) = (pitch.cos(), pitch.sin());
    let (cy, sy) = (yaw.cos(), yaw.sin());

    let matrix = match order {
        "ZYX" => [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        "ZXY" => [
            [cy * cp + sy * sp * sr, -cy * sp + sy * cp * sr, sy * cr],
            [sy * cp - cy * sp * sr, -sy * sp - cy * cp * sr, -cy * cr],
            [-cp * sr, cp * cr, sp],
        ],
        "YXZ" => [
            [cp * cy + sp * sr * sy, -cr * sy, sp * cy - cp * sr * sy],
            [cp * sy - sp * sr * cy, cr * cy, sp * sy + cp * sr * cy],
            [-sp * cr, sr, cp * cr],
        ],
        "YZX" => [
            [cp * cy, sr * sp - cr * cy * sy, cr * sp + sr * cy * sy],
            [sp, sr * cp, cr * cp],
            [-sy * cp, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
        ],
        "XYZ" => [
            [cp * cy, -cp * sy, sp],
            [sr * sp * cy + cr * sy, -sr * sp * sy + cr * cy, -sr * cp],
            [-cr * sp * cy + sr * sy, cr * sp * sy + sr * cy, cr * cp],
        ],
        "XZY" => [
            [cp * cy, -sy, sp * cy],
            [sr * sp + cr * cp * sy, cr * cy, -sr * cp + cr * sp * sy],
            [-cr * sp + sr * cp * sy, sr * cy, cr * cp + sr * sp * sy],
        ],
        _ => {
            return Err(PyValueError::new_err(
                "Unsupported order. Choose from 'ZYX', 'ZXY', 'YXZ', 'YZX', 'XYZ', 'XZY'.",
            ));
        }
    };

    Ok(matrix)
}

#[pymethods]
impl PySo3 {
    #[new]
    #[pyo3(signature = (matrix=None, lib=None))]
    pub fn new(matrix: Option<[[f64; 3]; 3]>, lib: Option<&str>) -> PyResult<Self> {
        ensure_numpy(lib)?;
        match matrix {
            Some(matrix) => Ok(Self {
                inner: So3::from_matrix(matrix),
            }),
            None => Ok(Self {
                inner: So3::identity(),
            }),
        }
    }

    #[getter]
    pub fn lib(&self) -> &'static str {
        "numpy"
    }

    #[staticmethod]
    pub fn dof() -> usize {
        So3::dof()
    }

    #[staticmethod]
    pub fn mat_size() -> usize {
        So3::mat_size()
    }

    #[staticmethod]
    pub fn mat_adj_size() -> usize {
        So3::mat_adj_size()
    }

    #[staticmethod]
    pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Self {
        Self {
            inner: So3::from_axis_angle(axis, angle),
        }
    }

    #[staticmethod]
    pub fn from_quaternion(quaternion: [f64; 4]) -> Self {
        Self {
            inner: So3::from_quaternion(quaternion),
        }
    }

    #[staticmethod]
    pub fn quaternion_to_mat(quaternion: [f64; 4]) -> [[f64; 3]; 3] {
        So3::from_quaternion(quaternion).to_matrix()
    }

    #[staticmethod]
    pub fn set_quaternion(quaternion: [f64; 4]) -> Self {
        Self::from_quaternion(quaternion)
    }

    #[staticmethod]
    pub fn from_euler_angles(roll: f64, pitch: f64, yaw: f64) -> Self {
        Self {
            inner: So3::from_euler_angles(roll, pitch, yaw),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (euler, order="ZYX", lib=None))]
    pub fn set_euler(euler: (f64, f64, f64), order: &str, lib: Option<&str>) -> PyResult<Self> {
        ensure_numpy(lib)?;
        Ok(Self {
            inner: So3::from_matrix(euler_matrix(euler, order)?),
        })
    }

    #[staticmethod]
    pub fn from_rotation_vector(vector: [f64; 3]) -> Self {
        Self {
            inner: So3::from_rotation_vector(vector),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (vector, a=None))]
    pub fn exp(vector: [f64; 3], a: Option<f64>) -> [[f64; 3]; 3] {
        So3::exp(vector, a)
    }

    #[staticmethod]
    #[pyo3(signature = (vector, a=None))]
    pub fn exp_array<'py>(py: Python<'py>, vector: [f64; 3], a: Option<f64>) -> Bound<'py, PyAny> {
        mat_to_numpy(py, So3::exp(vector, a)).into_any()
    }

    #[staticmethod]
    #[pyo3(signature = (vector, out, a=None))]
    pub fn exp_into(
        py: Python<'_>,
        vector: [f64; 3],
        out: PyBuffer<f64>,
        a: Option<f64>,
    ) -> PyResult<()> {
        write_mat3(py, out, So3::exp(vector, a))
    }

    #[staticmethod]
    #[pyo3(signature = (vector, a=None))]
    pub fn exp_integ(vector: [f64; 3], a: Option<f64>) -> [[f64; 3]; 3] {
        So3::exp_integ(vector, a)
    }

    #[staticmethod]
    #[pyo3(signature = (vector, a=None))]
    pub fn exp_integ_array<'py>(
        py: Python<'py>,
        vector: [f64; 3],
        a: Option<f64>,
    ) -> Bound<'py, PyAny> {
        mat_to_numpy(py, So3::exp_integ(vector, a)).into_any()
    }

    #[staticmethod]
    #[pyo3(signature = (vector, out, a=None))]
    pub fn exp_integ_into(
        py: Python<'_>,
        vector: [f64; 3],
        out: PyBuffer<f64>,
        a: Option<f64>,
    ) -> PyResult<()> {
        write_mat3(py, out, So3::exp_integ(vector, a))
    }

    #[staticmethod]
    #[pyo3(signature = (vector, a=None))]
    pub fn exp_integ2nd(vector: [f64; 3], a: Option<f64>) -> [[f64; 3]; 3] {
        So3::exp_integ2nd(vector, a)
    }

    #[staticmethod]
    #[pyo3(signature = (vector, a=None))]
    pub fn exp_integ2nd_array<'py>(
        py: Python<'py>,
        vector: [f64; 3],
        a: Option<f64>,
    ) -> Bound<'py, PyAny> {
        mat_to_numpy(py, So3::exp_integ2nd(vector, a)).into_any()
    }

    #[staticmethod]
    #[pyo3(signature = (vector, out, a=None))]
    pub fn exp_integ2nd_into(
        py: Python<'_>,
        vector: [f64; 3],
        out: PyBuffer<f64>,
        a: Option<f64>,
    ) -> PyResult<()> {
        write_mat3(py, out, So3::exp_integ2nd(vector, a))
    }

    #[staticmethod]
    #[pyo3(signature = (vector, a=None))]
    pub fn exp_adj(vector: [f64; 3], a: Option<f64>) -> [[f64; 3]; 3] {
        So3::exp_adj(vector, a)
    }

    #[staticmethod]
    #[pyo3(signature = (vector, a=None))]
    pub fn exp_adj_array<'py>(
        py: Python<'py>,
        vector: [f64; 3],
        a: Option<f64>,
    ) -> Bound<'py, PyAny> {
        mat_to_numpy(py, So3::exp_adj(vector, a)).into_any()
    }

    #[staticmethod]
    #[pyo3(signature = (vector, out, a=None))]
    pub fn exp_adj_into(
        py: Python<'_>,
        vector: [f64; 3],
        out: PyBuffer<f64>,
        a: Option<f64>,
    ) -> PyResult<()> {
        write_mat3(py, out, So3::exp_adj(vector, a))
    }

    #[staticmethod]
    #[pyo3(signature = (vector, a=None))]
    pub fn exp_integ_adj(vector: [f64; 3], a: Option<f64>) -> [[f64; 3]; 3] {
        So3::exp_integ_adj(vector, a)
    }

    #[staticmethod]
    #[pyo3(signature = (vector, a=None))]
    pub fn exp_integ_adj_array<'py>(
        py: Python<'py>,
        vector: [f64; 3],
        a: Option<f64>,
    ) -> Bound<'py, PyAny> {
        mat_to_numpy(py, So3::exp_integ_adj(vector, a)).into_any()
    }

    #[staticmethod]
    #[pyo3(signature = (vector, out, a=None))]
    pub fn exp_integ_adj_into(
        py: Python<'_>,
        vector: [f64; 3],
        out: PyBuffer<f64>,
        a: Option<f64>,
    ) -> PyResult<()> {
        write_mat3(py, out, So3::exp_integ_adj(vector, a))
    }

    #[staticmethod]
    #[pyo3(signature = (lib=None))]
    pub fn rand(lib: Option<&str>) -> PyResult<Self> {
        ensure_numpy(lib)?;
        Ok(Self {
            inner: So3::from_rotation_vector([
                rand::random::<f64>(),
                rand::random::<f64>(),
                rand::random::<f64>(),
            ]),
        })
    }

    pub fn apply(&self, vector: [f64; 3]) -> [f64; 3] {
        self.inner.apply(vector)
    }

    pub fn apply_array<'py>(&self, py: Python<'py>, vector: [f64; 3]) -> Bound<'py, PyAny> {
        vec_to_numpy(py, self.inner.apply(vector)).into_any()
    }

    pub fn apply_into(&self, py: Python<'_>, vector: [f64; 3], out: PyBuffer<f64>) -> PyResult<()> {
        write_vec3(py, out, self.inner.apply(vector))
    }

    #[staticmethod]
    pub fn hat(vector: [f64; 3]) -> [[f64; 3]; 3] {
        So3::hat(vector)
    }

    #[staticmethod]
    pub fn hat_array<'py>(py: Python<'py>, vector: [f64; 3]) -> Bound<'py, PyAny> {
        mat_to_numpy(py, So3::hat(vector)).into_any()
    }

    #[staticmethod]
    pub fn hat_into(py: Python<'_>, vector: [f64; 3], out: PyBuffer<f64>) -> PyResult<()> {
        write_mat3(py, out, So3::hat(vector))
    }

    #[staticmethod]
    pub fn hat_commute(vector: [f64; 3]) -> [[f64; 3]; 3] {
        So3::hat_commute(vector)
    }

    #[staticmethod]
    pub fn hat_commute_array<'py>(py: Python<'py>, vector: [f64; 3]) -> Bound<'py, PyAny> {
        mat_to_numpy(py, So3::hat_commute(vector)).into_any()
    }

    #[staticmethod]
    pub fn hat_commute_into(
        py: Python<'_>,
        vector: [f64; 3],
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        write_mat3(py, out, So3::hat_commute(vector))
    }

    #[staticmethod]
    pub fn hat_adj(vector: [f64; 3]) -> [[f64; 3]; 3] {
        So3::hat_adj(vector)
    }

    #[staticmethod]
    pub fn hat_adj_array<'py>(py: Python<'py>, vector: [f64; 3]) -> Bound<'py, PyAny> {
        mat_to_numpy(py, So3::hat_adj(vector)).into_any()
    }

    #[staticmethod]
    pub fn hat_adj_into(py: Python<'_>, vector: [f64; 3], out: PyBuffer<f64>) -> PyResult<()> {
        write_mat3(py, out, So3::hat_adj(vector))
    }

    #[staticmethod]
    pub fn hat_commute_adj(vector: [f64; 3]) -> [[f64; 3]; 3] {
        So3::hat_commute_adj(vector)
    }

    #[staticmethod]
    pub fn hat_commute_adj_array<'py>(py: Python<'py>, vector: [f64; 3]) -> Bound<'py, PyAny> {
        mat_to_numpy(py, So3::hat_commute_adj(vector)).into_any()
    }

    #[staticmethod]
    pub fn hat_commute_adj_into(
        py: Python<'_>,
        vector: [f64; 3],
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        write_mat3(py, out, So3::hat_commute_adj(vector))
    }

    #[staticmethod]
    pub fn vee(matrix: [[f64; 3]; 3]) -> [f64; 3] {
        So3::vee(matrix)
    }

    #[staticmethod]
    pub fn vee_array<'py>(py: Python<'py>, matrix: [[f64; 3]; 3]) -> Bound<'py, PyAny> {
        vec_to_numpy(py, So3::vee(matrix)).into_any()
    }

    #[staticmethod]
    pub fn vee_into(py: Python<'_>, matrix: [[f64; 3]; 3], out: PyBuffer<f64>) -> PyResult<()> {
        write_vec3(py, out, So3::vee(matrix))
    }

    #[staticmethod]
    pub fn vee_adj(matrix: [[f64; 3]; 3]) -> [f64; 3] {
        So3::vee_adj(matrix)
    }

    #[staticmethod]
    pub fn vee_adj_array<'py>(py: Python<'py>, matrix: [[f64; 3]; 3]) -> Bound<'py, PyAny> {
        vec_to_numpy(py, So3::vee_adj(matrix)).into_any()
    }

    #[staticmethod]
    pub fn vee_adj_into(
        py: Python<'_>,
        matrix: [[f64; 3]; 3],
        out: PyBuffer<f64>,
    ) -> PyResult<()> {
        write_vec3(py, out, So3::vee_adj(matrix))
    }

    #[staticmethod]
    #[pyo3(signature = (left, right, frame="bframe"))]
    pub fn sub_tan_vec(left: &PySo3, right: &PySo3, frame: &str) -> [f64; 3] {
        So3::sub_tan_vec(&left.inner, &right.inner, Some(frame))
    }

    #[staticmethod]
    pub fn so3_mul(left: [[f64; 3]; 3], right: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
        matrix_to_array(&(matrix3_from_array(left) * matrix3_from_array(right)))
    }

    pub fn compose(&self, other: &PySo3) -> PySo3 {
        PySo3 {
            inner: self.inner.compose(&other.inner),
        }
    }

    #[pyo3(name = "__mul__")]
    pub fn mul(&self, other: &PySo3) -> PyResult<PySo3> {
        Ok(self.compose(other))
    }

    #[pyo3(name = "__matmul__")]
    pub fn matmul(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(other) = other.extract::<PyRef<'_, PySo3>>() {
            return Ok(Py::new(
                py,
                PySo3 {
                    inner: self.inner.compose(&other.inner),
                },
            )?
            .into_py(py));
        }

        if let Ok(vector) = other.extract::<[f64; 3]>() {
            return Ok(self.inner.apply(vector).into_py(py));
        }

        if let Ok(matrix) = other.extract::<[[f64; 3]; 3]>() {
            let result = self.inner.as_matrix() * matrix3_from_array(matrix);
            return Ok(matrix_to_array(&result).into_py(py));
        }

        Err(PyTypeError::new_err(
            "Right operand should be SO3 or a 3D vector/matrix",
        ))
    }

    pub fn inverse(&self) -> PySo3 {
        PySo3 {
            inner: self.inner.inverse(),
        }
    }

    pub fn inv(&self) -> PySo3 {
        self.inverse()
    }

    pub fn matrix(&self) -> [[f64; 3]; 3] {
        self.inner.to_matrix()
    }

    pub fn matrix_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        mat_to_numpy(py, self.inner.to_matrix()).into_any()
    }

    pub fn mat(&self) -> [[f64; 3]; 3] {
        self.inner.to_matrix()
    }

    pub fn mat_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        mat_to_numpy(py, self.inner.to_matrix()).into_any()
    }

    pub fn mat_into(&self, py: Python<'_>, out: PyBuffer<f64>) -> PyResult<()> {
        write_mat3(py, out, self.inner.to_matrix())
    }

    #[staticmethod]
    pub fn set_mat(matrix: [[f64; 3]; 3]) -> Self {
        Self {
            inner: So3::from_matrix(matrix),
        }
    }

    #[staticmethod]
    pub fn set_mat_adj(matrix: [[f64; 3]; 3]) -> Self {
        Self::set_mat(matrix)
    }

    #[staticmethod]
    pub fn eye() -> Self {
        Self {
            inner: So3::identity(),
        }
    }

    pub fn mat_inv(&self) -> [[f64; 3]; 3] {
        self.inner.inverse().to_matrix()
    }

    pub fn mat_inv_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        mat_to_numpy(py, self.inner.inverse().to_matrix()).into_any()
    }

    pub fn mat_inv_into(&self, py: Python<'_>, out: PyBuffer<f64>) -> PyResult<()> {
        write_mat3(py, out, self.inner.inverse().to_matrix())
    }

    pub fn mat_adj(&self) -> [[f64; 3]; 3] {
        self.inner.to_matrix()
    }

    pub fn mat_adj_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        mat_to_numpy(py, self.inner.to_matrix()).into_any()
    }

    pub fn mat_adj_into(&self, py: Python<'_>, out: PyBuffer<f64>) -> PyResult<()> {
        write_mat3(py, out, self.inner.to_matrix())
    }

    pub fn mat_inv_adj(&self) -> [[f64; 3]; 3] {
        self.inner.inverse().to_matrix()
    }

    pub fn mat_inv_adj_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        mat_to_numpy(py, self.inner.inverse().to_matrix()).into_any()
    }

    pub fn mat_inv_adj_into(&self, py: Python<'_>, out: PyBuffer<f64>) -> PyResult<()> {
        write_mat3(py, out, self.inner.inverse().to_matrix())
    }

    pub fn quaternion(&self) -> [f64; 4] {
        self.inner.to_quaternion()
    }

    pub fn quaternion_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        vec_to_numpy(py, self.inner.to_quaternion()).into_any()
    }

    pub fn quaternion_into(&self, py: Python<'_>, out: PyBuffer<f64>) -> PyResult<()> {
        write_vec4(py, out, self.inner.to_quaternion())
    }

    #[staticmethod]
    pub fn mat_to_quaternion(matrix: [[f64; 3]; 3]) -> [f64; 4] {
        So3::from_matrix(matrix).to_quaternion()
    }

    pub fn euler_angles(&self) -> (f64, f64, f64) {
        self.inner.to_euler_angles()
    }

    pub fn rotation_vector(&self) -> [f64; 3] {
        self.inner.to_rotation_vector()
    }

    pub fn rotation_vector_array<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        vec_to_numpy(py, self.inner.to_rotation_vector()).into_any()
    }

    pub fn rotation_vector_into(&self, py: Python<'_>, out: PyBuffer<f64>) -> PyResult<()> {
        write_vec3(py, out, self.inner.to_rotation_vector())
    }

    #[pyo3(signature = (arb_vec, tan_var_vec, frame="bframe"))]
    pub fn mat_var_x_arb_vec(
        &self,
        arb_vec: [f64; 3],
        tan_var_vec: [f64; 3],
        frame: &str,
    ) -> [f64; 3] {
        self.inner
            .mat_var_x_arb_vec(arb_vec, tan_var_vec, Some(frame))
    }

    #[pyo3(signature = (arb_vec, frame="bframe"))]
    pub fn mat_var_x_arb_vec_jacob(&self, arb_vec: [f64; 3], frame: &str) -> [[f64; 3]; 3] {
        self.inner.mat_var_x_arb_vec_jacob(arb_vec, Some(frame))
    }

    #[pyo3(signature = (arb_vec, tan_var_vec, out, frame="bframe"))]
    pub fn mat_var_x_arb_vec_into(
        &self,
        py: Python<'_>,
        arb_vec: [f64; 3],
        tan_var_vec: [f64; 3],
        out: PyBuffer<f64>,
        frame: &str,
    ) -> PyResult<()> {
        write_vec3(
            py,
            out,
            self.inner.mat_var_x_arb_vec(arb_vec, tan_var_vec, Some(frame)),
        )
    }

    #[pyo3(signature = (arb_vec, out, frame="bframe"))]
    pub fn mat_var_x_arb_vec_jacob_into(
        &self,
        py: Python<'_>,
        arb_vec: [f64; 3],
        out: PyBuffer<f64>,
        frame: &str,
    ) -> PyResult<()> {
        write_mat3(py, out, self.inner.mat_var_x_arb_vec_jacob(arb_vec, Some(frame)))
    }

    pub fn __repr__(&self) -> String {
        format!("SO3({:?}, LIB='numpy')", self.inner.to_matrix())
    }
}
