use mathroborust::{Cmtm, Se3, So3};
use mathroborust::lie::LieGroup;
use pyo3::prelude::*;

#[pymodule]
pub fn mathrobors(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PySo3>()?;
    module.add_class::<PySe3>()?;
    module.add_class::<PyCmtm>()?;
    Ok(())
}

#[pyclass(name = "SO3")]
pub struct PySo3 {
    inner: So3,
}

#[pymethods]
impl PySo3 {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: So3::identity(),
        }
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
        Self {
            inner: So3::from_quaternion(quaternion),
        }
    }

    #[staticmethod]
    pub fn from_euler_angles(roll: f64, pitch: f64, yaw: f64) -> Self {
        Self {
            inner: So3::from_euler_angles(roll, pitch, yaw),
        }
    }

    #[staticmethod]
    pub fn set_euler(euler: (f64, f64, f64)) -> Self {
        Self {
            inner: So3::from_euler_angles(euler.0, euler.1, euler.2),
        }
    }

    #[staticmethod]
    pub fn from_rotation_vector(vector: [f64; 3]) -> Self {
        Self {
            inner: So3::from_rotation_vector(vector),
        }
    }

    #[staticmethod]
    pub fn exp(vector: [f64; 3], a: Option<f64>) -> [[f64; 3]; 3] {
        let scale = a.unwrap_or(1.0);
        let scaled = [vector[0] * scale, vector[1] * scale, vector[2] * scale];
        So3::from_rotation_vector(scaled).to_matrix()
    }

    #[staticmethod]
    pub fn exp_adj(vector: [f64; 3], a: Option<f64>) -> [[f64; 3]; 3] {
        PySo3::exp(vector, a)
    }

    pub fn apply(&self, vector: [f64; 3]) -> [f64; 3] {
        self.inner.apply(vector)
    }

    #[staticmethod]
    pub fn hat(vector: [f64; 3]) -> [[f64; 3]; 3] {
        So3::hat(vector)
    }

    #[staticmethod]
    pub fn hat_commute(vector: [f64; 3]) -> [[f64; 3]; 3] {
        let mut hat = So3::hat(vector);
        for r in 0..3 {
            for c in 0..3 {
                hat[r][c] = -hat[r][c];
            }
        }
        hat
    }

    #[staticmethod]
    pub fn hat_adj(vector: [f64; 3]) -> [[f64; 3]; 3] {
        So3::hat(vector)
    }

    #[staticmethod]
    pub fn hat_commute_adj(vector: [f64; 3]) -> [[f64; 3]; 3] {
        PySo3::hat_commute(vector)
    }

    #[staticmethod]
    pub fn vee(matrix: [[f64; 3]; 3]) -> [f64; 3] {
        So3::vee(matrix)
    }

    #[staticmethod]
    pub fn vee_adj(matrix: [[f64; 3]; 3]) -> [f64; 3] {
        So3::vee(matrix)
    }

    pub fn compose(&self, other: &PySo3) -> PySo3 {
        PySo3 {
            inner: self.inner.compose(&other.inner),
        }
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

    pub fn mat(&self) -> [[f64; 3]; 3] {
        self.inner.to_matrix()
    }

    #[staticmethod]
    pub fn set_mat(matrix: [[f64; 3]; 3]) -> Self {
        Self {
            inner: So3::from_matrix(matrix),
        }
    }

    #[staticmethod]
    pub fn set_mat_adj(matrix: [[f64; 3]; 3]) -> Self {
        Self {
            inner: So3::from_matrix(matrix),
        }
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

    pub fn mat_adj(&self) -> [[f64; 3]; 3] {
        self.inner.to_matrix()
    }

    pub fn mat_inv_adj(&self) -> [[f64; 3]; 3] {
        self.inner.inverse().to_matrix()
    }

    pub fn quaternion(&self) -> [f64; 4] {
        self.inner.to_quaternion()
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
}

#[pyclass(name = "SE3")]
pub struct PySe3 {
    inner: Se3,
}

#[pymethods]
impl PySe3 {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Se3::identity(),
        }
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

    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        self.inner.apply(point)
    }

    pub fn compose(&self, other: &PySe3) -> PySe3 {
        PySe3 {
            inner: self.inner.compose(&other.inner),
        }
    }

    pub fn inverse(&self) -> PySe3 {
        PySe3 {
            inner: self.inner.inverse(),
        }
    }

    pub fn matrix(&self) -> [[f64; 4]; 4] {
        self.inner.to_matrix()
    }

    pub fn translation(&self) -> [f64; 3] {
        self.inner.translation()
    }
}

#[pyclass(name = "CMTM")]
pub struct PyCmtm {
    inner: Cmtm,
}

#[pymethods]
impl PyCmtm {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: Cmtm::identity(),
        }
    }

    #[staticmethod]
    pub fn from_se3(transform: &PySe3) -> Self {
        Self {
            inner: Cmtm::from_se3(&transform.inner),
        }
    }

    pub fn apply_twist(&self, twist: [f64; 6]) -> [f64; 6] {
        self.inner.apply_twist(twist)
    }

    pub fn matrix(&self) -> [[f64; 6]; 6] {
        self.inner.to_matrix()
    }
}
