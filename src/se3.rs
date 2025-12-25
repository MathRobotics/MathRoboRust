use nalgebra::{Matrix4, SMatrix, Translation3};

use crate::{
    lie::{HasAdjoint, LieGroup, matrix_to_array},
    so3::So3,
    util::{vector3_from_array, vector3_to_array},
};

#[derive(Debug, Clone, PartialEq)]
pub struct Se3 {
    rotation: So3,
    translation: Translation3<f64>,
}

impl Se3 {
    pub fn from_parts(rotation: So3, translation: [f64; 3]) -> Self {
        Self {
            rotation,
            translation: Translation3::new(translation[0], translation[1], translation[2]),
        }
    }

    pub fn from_axis_angle_translation(axis: [f64; 3], angle: f64, translation: [f64; 3]) -> Self {
        let rotation = So3::from_axis_angle(axis, angle);
        Self::from_parts(rotation, translation)
    }

    pub fn compose(&self, other: &Self) -> Self {
        let new_rotation = self.rotation.compose(&other.rotation);
        let translated =
            self.translation.vector + self.rotation.rotation() * other.translation.vector;
        Self {
            rotation: new_rotation,
            translation: Translation3::from(translated),
        }
    }

    pub fn inverse(&self) -> Self {
        let inv_rotation = self.rotation.inverse();
        let inv_translation = -(inv_rotation.rotation() * self.translation.vector);
        Self {
            rotation: inv_rotation,
            translation: Translation3::from(inv_translation),
        }
    }

    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        let point_vec = vector3_from_array(point);
        let rotated = self.rotation.rotation() * point_vec;
        let translated = rotated + self.translation.vector;
        vector3_to_array(&translated)
    }

    pub fn to_matrix(&self) -> [[f64; 4]; 4] {
        let mut matrix = Matrix4::<f64>::identity();
        let rotation_matrix = self.rotation.rotation().matrix();
        for row in 0..3 {
            for col in 0..3 {
                matrix[(row, col)] = rotation_matrix[(row, col)];
            }
            matrix[(row, 3)] = self.translation.vector[row];
        }
        matrix_to_array(&matrix)
    }

    pub fn rotation(&self) -> &So3 {
        &self.rotation
    }

    pub fn translation(&self) -> [f64; 3] {
        vector3_to_array(&self.translation.vector)
    }

    pub fn adjoint(&self) -> SMatrix<f64, 6, 6> {
        let rotation = self.rotation.rotation();
        let translation = self.translation();
        let translation_vec = vector3_from_array(translation);
        let skew = crate::util::skew_symmetric(&translation_vec);

        let mut matrix = SMatrix::<f64, 6, 6>::zeros();
        for r in 0..3 {
            for c in 0..3 {
                matrix[(r, c)] = rotation.matrix()[(r, c)];
                matrix[(r + 3, c + 3)] = rotation.matrix()[(r, c)];
                matrix[(r + 3, c)] = (skew * rotation.matrix())[(r, c)];
            }
        }

        matrix
    }
}

impl LieGroup<4> for Se3 {
    fn identity() -> Self {
        Self {
            rotation: So3::identity(),
            translation: Translation3::identity(),
        }
    }

    fn compose(&self, other: &Self) -> Self {
        self.compose(other)
    }

    fn inverse(&self) -> Self {
        self.inverse()
    }

    fn as_matrix(&self) -> SMatrix<f64, 4, 4> {
        let mut matrix = Matrix4::<f64>::identity();
        let rotation_matrix = self.rotation.rotation().matrix();
        for row in 0..3 {
            for col in 0..3 {
                matrix[(row, col)] = rotation_matrix[(row, col)];
            }
            matrix[(row, 3)] = self.translation.vector[row];
        }
        matrix.clone_owned()
    }
}

impl HasAdjoint<6> for Se3 {
    fn adjoint_matrix(&self) -> SMatrix<f64, 6, 6> {
        self.adjoint()
    }
}
