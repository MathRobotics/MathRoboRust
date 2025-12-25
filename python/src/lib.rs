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

    pub fn apply(&self, vector: [f64; 3]) -> [f64; 3] {
        self.inner.apply(vector)
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

    pub fn matrix(&self) -> [[f64; 3]; 3] {
        self.inner.to_matrix()
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
