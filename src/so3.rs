use nalgebra::{Rotation3, Vector3};

use crate::lie::{LieGroup, apply_linear, matrix_to_array};
use crate::util::vector3_from_array;

#[derive(Debug, Clone, PartialEq)]
pub struct So3 {
    rotation: Rotation3<f64>,
}

impl So3 {
    pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Self {
        let axis_vector = vector3_from_array(axis);
        if axis_vector.norm() == 0.0 {
            return Self::identity();
        }

        Self {
            rotation: Rotation3::new(axis_vector.normalize() * angle),
        }
    }

    pub fn compose(&self, other: &Self) -> Self {
        Self {
            rotation: self.rotation * other.rotation,
        }
    }

    pub fn inverse(&self) -> Self {
        Self {
            rotation: self.rotation.inverse(),
        }
    }

    pub fn apply(&self, vector: [f64; 3]) -> [f64; 3] {
        apply_linear(&self.rotation.matrix().clone_owned(), vector)
    }

    pub fn to_matrix(&self) -> [[f64; 3]; 3] {
        matrix_to_array(&self.rotation.matrix().clone_owned())
    }

    pub fn rotation(&self) -> &Rotation3<f64> {
        &self.rotation
    }
}

impl LieGroup<3> for So3 {
    fn identity() -> Self {
        Self {
            rotation: Rotation3::identity(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        self.compose(other)
    }

    fn inverse(&self) -> Self {
        self.inverse()
    }

    fn as_matrix(&self) -> nalgebra::SMatrix<f64, 3, 3> {
        self.rotation.matrix().clone_owned()
    }
}
