use mathroborust::lie::LieGroup;
use mathroborust::{RotationalCmtm, Se3, So3, SpatialCmtm};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::convert::{dmatrix_from_vec, dmatrix_to_vec};
use crate::py_se3::PySe3;
use crate::py_so3::PySo3;

#[derive(Clone)]
enum CmtmInner {
    So3(RotationalCmtm),
    Se3(SpatialCmtm),
}

#[pyclass(name = "CMTM")]
#[derive(Clone)]
pub struct PyCmtm {
    inner: CmtmInner,
}

fn group_name(group: &Bound<'_, PyAny>) -> PyResult<String> {
    group.getattr("__name__")?.extract::<String>()
}

fn parse_so3_vecs(vecs: &[Vec<f64>]) -> PyResult<Vec<[f64; 3]>> {
    vecs.iter()
        .map(|row| match row.as_slice() {
            [a, b, c] => Ok([*a, *b, *c]),
            _ => Err(PyValueError::new_err(
                "SO3 derivative rows must have length 3",
            )),
        })
        .collect()
}

fn parse_se3_vecs(vecs: &[Vec<f64>]) -> PyResult<Vec<[f64; 6]>> {
    vecs.iter()
        .map(|row| match row.as_slice() {
            [a, b, c, d, e, f] => Ok([*a, *b, *c, *d, *e, *f]),
            _ => Err(PyValueError::new_err(
                "SE3 derivative rows must have length 6",
            )),
        })
        .collect()
}

fn so3_vecs_to_vec(vecs: Vec<[f64; 3]>) -> Vec<Vec<f64>> {
    vecs.into_iter()
        .map(|row| vec![row[0], row[1], row[2]])
        .collect()
}

fn se3_vecs_to_vec(vecs: Vec<[f64; 6]>) -> Vec<Vec<f64>> {
    vecs.into_iter()
        .map(|row| vec![row[0], row[1], row[2], row[3], row[4], row[5]])
        .collect()
}

impl PyCmtm {
    fn from_element(
        element: &Bound<'_, PyAny>,
        elem_vecs: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        if let Ok(rotation) = element.extract::<PyRef<'_, PySo3>>() {
            let derivatives = parse_so3_vecs(elem_vecs.as_deref().unwrap_or(&[]))?;
            return Ok(Self {
                inner: CmtmInner::So3(RotationalCmtm::from_so3_with_derivatives(
                    &rotation.inner,
                    derivatives,
                )),
            });
        }

        if let Ok(transform) = element.extract::<PyRef<'_, PySe3>>() {
            let derivatives = parse_se3_vecs(elem_vecs.as_deref().unwrap_or(&[]))?;
            return Ok(Self {
                inner: CmtmInner::Se3(SpatialCmtm::from_se3_with_derivatives(
                    &transform.inner,
                    derivatives,
                )),
            });
        }

        Err(PyTypeError::new_err(
            "CMTM expects an SO3 or SE3 element as the first argument",
        ))
    }

    fn group_matches(&self, other: &Self) -> bool {
        matches!(
            (&self.inner, &other.inner),
            (CmtmInner::So3(_), CmtmInner::So3(_)) | (CmtmInner::Se3(_), CmtmInner::Se3(_))
        )
    }
}

#[pymethods]
impl PyCmtm {
    #[new]
    #[pyo3(signature = (element=None, elem_vecs=None))]
    pub fn new(
        element: Option<&Bound<'_, PyAny>>,
        elem_vecs: Option<Vec<Vec<f64>>>,
    ) -> PyResult<Self> {
        match element {
            Some(element) => Self::from_element(element, elem_vecs),
            None => Ok(Self {
                inner: CmtmInner::Se3(SpatialCmtm::identity()),
            }),
        }
    }

    #[classmethod]
    pub fn __class_getitem__(cls: &Bound<'_, PyType>, _item: &Bound<'_, PyAny>) -> Py<PyAny> {
        cls.clone().into_any().unbind()
    }

    #[staticmethod]
    pub fn from_so3(rotation: &PySo3) -> Self {
        Self {
            inner: CmtmInner::So3(RotationalCmtm::from_so3(&rotation.inner)),
        }
    }

    #[staticmethod]
    pub fn from_so3_with_derivatives(rotation: &PySo3, elem_vecs: Vec<Vec<f64>>) -> PyResult<Self> {
        Ok(Self {
            inner: CmtmInner::So3(RotationalCmtm::from_so3_with_derivatives(
                &rotation.inner,
                parse_so3_vecs(&elem_vecs)?,
            )),
        })
    }

    #[staticmethod]
    pub fn from_se3(transform: &PySe3) -> Self {
        Self {
            inner: CmtmInner::Se3(SpatialCmtm::from_se3(&transform.inner)),
        }
    }

    #[staticmethod]
    pub fn from_se3_with_derivatives(transform: &PySe3, elem_vecs: Vec<Vec<f64>>) -> PyResult<Self> {
        Ok(Self {
            inner: CmtmInner::Se3(SpatialCmtm::from_se3_with_derivatives(
                &transform.inner,
                parse_se3_vecs(&elem_vecs)?,
            )),
        })
    }

    #[staticmethod]
    #[pyo3(signature = (group, output_order=3))]
    pub fn eye(group: &Bound<'_, PyAny>, output_order: usize) -> PyResult<Self> {
        if output_order == 0 {
            return Err(PyValueError::new_err("output_order must be positive"));
        }

        match group_name(group)?.as_str() {
            "SO3" => Ok(Self {
                inner: CmtmInner::So3(RotationalCmtm::from_so3_with_derivatives(
                    &So3::identity(),
                    vec![[0.0; 3]; output_order - 1],
                )),
            }),
            "SE3" => Ok(Self {
                inner: CmtmInner::Se3(SpatialCmtm::from_se3_with_derivatives(
                    &Se3::identity(),
                    vec![[0.0; 6]; output_order - 1],
                )),
            }),
            _ => Err(PyTypeError::new_err("group must be SO3 or SE3")),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (group, output_order=3))]
    pub fn rand(group: &Bound<'_, PyAny>, output_order: usize) -> PyResult<Self> {
        if output_order == 0 {
            return Err(PyValueError::new_err("output_order must be positive"));
        }

        match group_name(group)?.as_str() {
            "SO3" => Ok(Self {
                inner: CmtmInner::So3(RotationalCmtm::from_so3_with_derivatives(
                    &So3::from_rotation_vector([
                        rand::random::<f64>(),
                        rand::random::<f64>(),
                        rand::random::<f64>(),
                    ]),
                    (0..output_order - 1)
                        .map(|_| {
                            [
                                rand::random::<f64>(),
                                rand::random::<f64>(),
                                rand::random::<f64>(),
                            ]
                        })
                        .collect(),
                )),
            }),
            "SE3" => Ok(Self {
                inner: CmtmInner::Se3(SpatialCmtm::from_se3_with_derivatives(
                    &Se3::from_axis_angle_translation(
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
                    (0..output_order - 1)
                        .map(|_| {
                            [
                                rand::random::<f64>(),
                                rand::random::<f64>(),
                                rand::random::<f64>(),
                                rand::random::<f64>(),
                                rand::random::<f64>(),
                                rand::random::<f64>(),
                            ]
                        })
                        .collect(),
                )),
            }),
            _ => Err(PyTypeError::new_err("group must be SO3 or SE3")),
        }
    }

    #[staticmethod]
    pub fn set_mat(group: &Bound<'_, PyAny>, mat: Vec<Vec<f64>>) -> PyResult<Self> {
        let matrix = dmatrix_from_vec(&mat);
        match group_name(group)?.as_str() {
            "SO3" => Ok(Self {
                inner: CmtmInner::So3(RotationalCmtm::set_mat(&matrix)),
            }),
            "SE3" => Ok(Self {
                inner: CmtmInner::Se3(SpatialCmtm::set_mat(&matrix)),
            }),
            _ => Err(PyTypeError::new_err("group must be SO3 or SE3")),
        }
    }

    #[staticmethod]
    pub fn set_mat_adj(group: &Bound<'_, PyAny>, mat: Vec<Vec<f64>>) -> PyResult<Self> {
        let matrix = dmatrix_from_vec(&mat);
        match group_name(group)?.as_str() {
            "SO3" => Ok(Self {
                inner: CmtmInner::So3(RotationalCmtm::set_mat_adj(&matrix)),
            }),
            "SE3" => Ok(Self {
                inner: CmtmInner::Se3(SpatialCmtm::set_mat_adj(&matrix)),
            }),
            _ => Err(PyTypeError::new_err("group must be SO3 or SE3")),
        }
    }

    #[staticmethod]
    pub fn hat(group: &Bound<'_, PyAny>, vecs: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        match group_name(group)?.as_str() {
            "SO3" => Ok(dmatrix_to_vec(&RotationalCmtm::hat(&parse_so3_vecs(
                &vecs,
            )?))),
            "SE3" => Ok(dmatrix_to_vec(&SpatialCmtm::hat(&parse_se3_vecs(&vecs)?))),
            _ => Err(PyTypeError::new_err("group must be SO3 or SE3")),
        }
    }

    #[staticmethod]
    pub fn hat_adj(group: &Bound<'_, PyAny>, vecs: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        match group_name(group)?.as_str() {
            "SO3" => Ok(dmatrix_to_vec(&RotationalCmtm::hat_adj(&parse_so3_vecs(
                &vecs,
            )?))),
            "SE3" => Ok(dmatrix_to_vec(&SpatialCmtm::hat_adj(&parse_se3_vecs(
                &vecs,
            )?))),
            _ => Err(PyTypeError::new_err("group must be SO3 or SE3")),
        }
    }

    #[staticmethod]
    pub fn hat_commute_adj(
        group: &Bound<'_, PyAny>,
        vecs: Vec<Vec<f64>>,
    ) -> PyResult<Vec<Vec<f64>>> {
        match group_name(group)?.as_str() {
            "SO3" => Ok(dmatrix_to_vec(&RotationalCmtm::hat_commute_adj(
                &parse_so3_vecs(&vecs)?,
            ))),
            "SE3" => Ok(dmatrix_to_vec(&SpatialCmtm::hat_commute_adj(
                &parse_se3_vecs(&vecs)?,
            ))),
            _ => Err(PyTypeError::new_err("group must be SO3 or SE3")),
        }
    }

    #[staticmethod]
    pub fn vee(group: &Bound<'_, PyAny>, hat_mat: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let matrix = dmatrix_from_vec(&hat_mat);
        match group_name(group)?.as_str() {
            "SO3" => Ok(so3_vecs_to_vec(RotationalCmtm::vee(&matrix))),
            "SE3" => Ok(se3_vecs_to_vec(SpatialCmtm::vee(&matrix))),
            _ => Err(PyTypeError::new_err("group must be SO3 or SE3")),
        }
    }

    #[staticmethod]
    pub fn vee_adj(group: &Bound<'_, PyAny>, hat_mat: Vec<Vec<f64>>) -> PyResult<Vec<Vec<f64>>> {
        let matrix = dmatrix_from_vec(&hat_mat);
        match group_name(group)?.as_str() {
            "SO3" => Ok(so3_vecs_to_vec(RotationalCmtm::vee_adj(&matrix))),
            "SE3" => Ok(se3_vecs_to_vec(SpatialCmtm::vee_adj(&matrix))),
            _ => Err(PyTypeError::new_err("group must be SO3 or SE3")),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (left, right, frame="bframe"))]
    pub fn sub_vec(left: &PyCmtm, right: &PyCmtm, frame: &str) -> PyResult<Vec<f64>> {
        if !left.group_matches(right) {
            return Err(PyTypeError::new_err("CMTM groups must match"));
        }

        match (&left.inner, &right.inner) {
            (CmtmInner::So3(left), CmtmInner::So3(right)) => {
                if left.order() != right.order() {
                    return Err(PyValueError::new_err("CMTM orders must match"));
                }
                let mut out = Vec::new();
                out.extend(So3::sub_tan_vec(
                    left.element(),
                    right.element(),
                    Some(frame),
                ));
                for (l, r) in left.vecs(None).iter().zip(right.vecs(None).iter()) {
                    out.extend((0..3).map(|idx| r[idx] - l[idx]));
                }
                Ok(out)
            }
            (CmtmInner::Se3(left), CmtmInner::Se3(right)) => {
                if left.order() != right.order() {
                    return Err(PyValueError::new_err("CMTM orders must match"));
                }
                let mut out = Vec::new();
                out.extend(Se3::sub_tan_vec(
                    left.element(),
                    right.element(),
                    Some(frame),
                ));
                for (l, r) in left.vecs(None).iter().zip(right.vecs(None).iter()) {
                    out.extend((0..6).map(|idx| r[idx] - l[idx]));
                }
                Ok(out)
            }
            _ => Err(PyTypeError::new_err("CMTM groups must match")),
        }
    }

    pub fn size(&self) -> usize {
        match &self.inner {
            CmtmInner::So3(inner) => inner.mat(None).nrows(),
            CmtmInner::Se3(inner) => inner.mat(None).nrows(),
        }
    }

    pub fn adj_size(&self) -> usize {
        match &self.inner {
            CmtmInner::So3(inner) => inner.mat_adj(None).nrows(),
            CmtmInner::Se3(inner) => inner.mat_adj(None).nrows(),
        }
    }

    pub fn elem_mat(&self) -> Vec<Vec<f64>> {
        match &self.inner {
            CmtmInner::So3(inner) => inner
                .to_elem_matrix()
                .into_iter()
                .map(|row| row.into_iter().collect())
                .collect(),
            CmtmInner::Se3(inner) => inner
                .to_elem_matrix()
                .into_iter()
                .map(|row| row.into_iter().collect())
                .collect(),
        }
    }

    pub fn elem_vecs(&self, index: usize) -> Option<Vec<f64>> {
        match &self.inner {
            CmtmInner::So3(inner) => inner.elem_vecs(index).map(|row| row.into_iter().collect()),
            CmtmInner::Se3(inner) => inner.elem_vecs(index).map(|row| row.into_iter().collect()),
        }
    }

    #[pyo3(signature = (output_order=None))]
    pub fn vecs(&self, output_order: Option<usize>) -> Vec<Vec<f64>> {
        match &self.inner {
            CmtmInner::So3(inner) => so3_vecs_to_vec(inner.vecs(output_order)),
            CmtmInner::Se3(inner) => se3_vecs_to_vec(inner.vecs(output_order)),
        }
    }

    #[pyo3(signature = (output_order=None))]
    pub fn vecs_flatten(&self, output_order: Option<usize>) -> Vec<f64> {
        match &self.inner {
            CmtmInner::So3(inner) => inner.vecs_flatten(output_order),
            CmtmInner::Se3(inner) => inner.vecs_flatten(output_order),
        }
    }

    #[pyo3(signature = (output_order=None))]
    pub fn mat(&self, output_order: Option<usize>) -> Vec<Vec<f64>> {
        match &self.inner {
            CmtmInner::So3(inner) => dmatrix_to_vec(&inner.mat(output_order)),
            CmtmInner::Se3(inner) => dmatrix_to_vec(&inner.mat(output_order)),
        }
    }

    #[pyo3(signature = (output_order=None))]
    pub fn mat_adj(&self, output_order: Option<usize>) -> Vec<Vec<f64>> {
        match &self.inner {
            CmtmInner::So3(inner) => dmatrix_to_vec(&inner.mat_adj(output_order)),
            CmtmInner::Se3(inner) => dmatrix_to_vec(&inner.mat_adj(output_order)),
        }
    }

    #[pyo3(signature = (output_order=None))]
    pub fn mat_inv(&self, output_order: Option<usize>) -> Vec<Vec<f64>> {
        match &self.inner {
            CmtmInner::So3(inner) => dmatrix_to_vec(&inner.mat_inv(output_order)),
            CmtmInner::Se3(inner) => dmatrix_to_vec(&inner.mat_inv(output_order)),
        }
    }

    #[pyo3(signature = (output_order=None))]
    pub fn mat_inv_adj(&self, output_order: Option<usize>) -> Vec<Vec<f64>> {
        match &self.inner {
            CmtmInner::So3(inner) => dmatrix_to_vec(&inner.mat_inv_adj(output_order)),
            CmtmInner::Se3(inner) => dmatrix_to_vec(&inner.mat_inv_adj(output_order)),
        }
    }

    #[pyo3(signature = (output_order=None))]
    pub fn tangent_mat(&self, output_order: Option<usize>) -> Vec<Vec<f64>> {
        match &self.inner {
            CmtmInner::So3(inner) => dmatrix_to_vec(&inner.tangent_mat(output_order)),
            CmtmInner::Se3(inner) => dmatrix_to_vec(&inner.tangent_mat(output_order)),
        }
    }

    #[pyo3(signature = (output_order=None))]
    pub fn tangent_mat_inv(&self, output_order: Option<usize>) -> Vec<Vec<f64>> {
        match &self.inner {
            CmtmInner::So3(inner) => dmatrix_to_vec(&inner.tangent_mat_inv(output_order)),
            CmtmInner::Se3(inner) => dmatrix_to_vec(&inner.tangent_mat_inv(output_order)),
        }
    }

    #[pyo3(signature = (output_order=None))]
    pub fn tangent_mat_cm(&self, output_order: Option<usize>) -> Vec<Vec<f64>> {
        match &self.inner {
            CmtmInner::So3(inner) => dmatrix_to_vec(&inner.tangent_mat_cm(output_order)),
            CmtmInner::Se3(inner) => dmatrix_to_vec(&inner.tangent_mat_cm(output_order)),
        }
    }

    #[pyo3(signature = (output_order=None))]
    pub fn tangent_mat_cm_inv(&self, output_order: Option<usize>) -> Vec<Vec<f64>> {
        match &self.inner {
            CmtmInner::So3(inner) => dmatrix_to_vec(&inner.tangent_mat_cm_inv(output_order)),
            CmtmInner::Se3(inner) => dmatrix_to_vec(&inner.tangent_mat_cm_inv(output_order)),
        }
    }

    pub fn inv(&self) -> PyCmtm {
        match &self.inner {
            CmtmInner::So3(inner) => PyCmtm {
                inner: CmtmInner::So3(inner.inverse()),
            },
            CmtmInner::Se3(inner) => PyCmtm {
                inner: CmtmInner::Se3(inner.inverse()),
            },
        }
    }

    pub fn inverse(&self) -> PyCmtm {
        self.inv()
    }

    pub fn compose(&self, other: &PyCmtm) -> PyResult<PyCmtm> {
        match (&self.inner, &other.inner) {
            (CmtmInner::So3(left), CmtmInner::So3(right)) => Ok(PyCmtm {
                inner: CmtmInner::So3(left.compose(right)),
            }),
            (CmtmInner::Se3(left), CmtmInner::Se3(right)) => Ok(PyCmtm {
                inner: CmtmInner::Se3(left.compose(right)),
            }),
            _ => Err(PyTypeError::new_err("CMTM groups must match")),
        }
    }

    #[pyo3(name = "__mul__")]
    pub fn mul(&self, other: &PyCmtm) -> PyResult<PyCmtm> {
        self.compose(other)
    }

    #[pyo3(name = "__matmul__")]
    pub fn matmul(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(other) = other.extract::<PyRef<'_, PyCmtm>>() {
            return Ok(Py::new(py, self.compose(&other)?)?.into_any());
        }

        if let Ok(vector) = other.extract::<Vec<f64>>() {
            let matrix = dmatrix_from_vec(&self.mat_adj(None));
            let result = matrix * nalgebra::DVector::from_vec(vector);
            return Ok(result
                .as_slice()
                .to_vec()
                .into_pyobject(py)?
                .unbind()
                .into());
        }

        if let Ok(matrix) = other.extract::<Vec<Vec<f64>>>() {
            let lhs = dmatrix_from_vec(&self.mat(None));
            let rhs = dmatrix_from_vec(&matrix);
            return Ok(dmatrix_to_vec(&(lhs * rhs))
                .into_pyobject(py)?
                .unbind()
                .into());
        }

        Err(PyTypeError::new_err(
            "Right operand should be CMTM, a vector, or a matrix",
        ))
    }

    pub fn matrix(&self) -> Vec<Vec<f64>> {
        self.mat_adj(None)
    }

    pub fn apply_twist(&self, twist: [f64; 6]) -> PyResult<[f64; 6]> {
        match &self.inner {
            CmtmInner::Se3(inner) => Ok(inner.apply_twist(twist)),
            CmtmInner::So3(_) => Err(PyTypeError::new_err(
                "apply_twist is only available for SE3-backed CMTM",
            )),
        }
    }

    pub fn apply_omega(&self, omega: [f64; 3]) -> PyResult<[f64; 3]> {
        match &self.inner {
            CmtmInner::So3(inner) => Ok(inner.apply_omega(omega)),
            CmtmInner::Se3(_) => Err(PyTypeError::new_err(
                "apply_omega is only available for SO3-backed CMTM",
            )),
        }
    }
}
