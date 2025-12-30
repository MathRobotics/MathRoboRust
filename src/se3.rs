use nalgebra::{Matrix3, Matrix4, Rotation3, SMatrix, Translation3, Vector3};

use crate::{
    lie::{HasAdjoint, LieGroup, matrix_to_array},
    so3::So3,
    util::{vector3_from_array, vector3_to_array},
};

/// A rigid-body transform in the special Euclidean group \(\mathrm{SE}(3)\),
/// storing a rotation and translation.
#[derive(Debug, Clone, PartialEq)]
pub struct Se3 {
    rotation: So3,
    translation: Translation3<f64>,
}

impl Se3 {
    /// Build an SE(3) element directly from a 4×4 homogeneous matrix.
    /// The bottom row is assumed to be `[0, 0, 0, 1]` and the top-left
    /// 3×3 block is interpreted as a rotation matrix.
    pub fn from_matrix(matrix: [[f64; 4]; 4]) -> Self {
        let flat: [f64; 16] = [
            matrix[0][0],
            matrix[0][1],
            matrix[0][2],
            matrix[0][3],
            matrix[1][0],
            matrix[1][1],
            matrix[1][2],
            matrix[1][3],
            matrix[2][0],
            matrix[2][1],
            matrix[2][2],
            matrix[2][3],
            matrix[3][0],
            matrix[3][1],
            matrix[3][2],
            matrix[3][3],
        ];
        let mat = Matrix4::from_row_slice(&flat);

        let rotation_matrix: [[f64; 3]; 3] = [
            [mat[(0, 0)], mat[(0, 1)], mat[(0, 2)]],
            [mat[(1, 0)], mat[(1, 1)], mat[(1, 2)]],
            [mat[(2, 0)], mat[(2, 1)], mat[(2, 2)]],
        ];

        let translation = [mat[(0, 3)], mat[(1, 3)], mat[(2, 3)]];

        Self::from_parts(So3::from_matrix(rotation_matrix), translation)
    }

    /// Construct the Lie-algebra hat operator mapping a 6D twist vector
    /// into a 4×4 matrix in `se(3)`.
    pub fn hat(twist: [f64; 6]) -> [[f64; 4]; 4] {
        let omega = Vector3::new(twist[0], twist[1], twist[2]);
        let v = Vector3::new(twist[3], twist[4], twist[5]);
        let mut mat = Matrix4::<f64>::zeros();

        let skew = crate::util::skew_symmetric(&omega);
        for r in 0..3 {
            for c in 0..3 {
                mat[(r, c)] = skew[(r, c)];
            }
            mat[(r, 3)] = v[r];
        }

        matrix_to_array(&mat)
    }

    /// Inverse of [`Se3::hat`], recovering a 6D twist vector from a matrix
    /// representation in `se(3)`.
    pub fn vee(matrix: [[f64; 4]; 4]) -> [f64; 6] {
        let rotation_block = [
            [matrix[0][0], matrix[0][1], matrix[0][2]],
            [matrix[1][0], matrix[1][1], matrix[1][2]],
            [matrix[2][0], matrix[2][1], matrix[2][2]],
        ];

        let omega = So3::vee(rotation_block);
        [
            omega[0],
            omega[1],
            omega[2],
            matrix[0][3],
            matrix[1][3],
            matrix[2][3],
        ]
    }

    /// Compute the exponential map from a 6D twist to an SE(3) transform.
    /// The optional scale factor `a` can be used to scale the twist prior to
    /// exponentiation.
    pub fn exp(twist: [f64; 6], a: Option<f64>) -> [[f64; 4]; 4] {
        let scale = a.unwrap_or(1.0);
        let omega = Vector3::new(twist[0] * scale, twist[1] * scale, twist[2] * scale);
        let v = Vector3::new(twist[3] * scale, twist[4] * scale, twist[5] * scale);

        let theta = omega.norm();
        let rotation = Rotation3::new(omega);
        let rotation_matrix = rotation.matrix();

        let mut hat = Matrix3::<f64>::zeros();
        let mut hat_sq = Matrix3::<f64>::zeros();

        if theta != 0.0 {
            hat = crate::util::skew_symmetric(&omega);
            hat_sq = hat * hat;
        }

        let v_matrix = if theta.abs() < 1e-12 {
            Matrix3::<f64>::identity() + 0.5 * hat
        } else {
            let theta_sq = theta * theta;
            let theta_cu = theta_sq * theta;
            Matrix3::<f64>::identity()
                + (1.0 - theta.cos()) / theta_sq * hat
                + (theta - theta.sin()) / theta_cu * hat_sq
        };

        let translated = v_matrix * v;

        let mut matrix = Matrix4::<f64>::identity();
        for r in 0..3 {
            for c in 0..3 {
                matrix[(r, c)] = rotation_matrix[(r, c)];
            }
            matrix[(r, 3)] = translated[r];
        }

        matrix_to_array(&matrix)
    }

    pub fn from_parts(rotation: So3, translation: [f64; 3]) -> Self {
        Self {
            rotation,
            translation: Translation3::new(translation[0], translation[1], translation[2]),
        }
    }

    /// Construct an SE(3) transform from an axis–angle rotation and a translation
    /// vector, yielding the homogeneous transform \(T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}\).
    pub fn from_axis_angle_translation(axis: [f64; 3], angle: f64, translation: [f64; 3]) -> Self {
        let rotation = So3::from_axis_angle(axis, angle);
        Self::from_parts(rotation, translation)
    }

    /// Left-multiply two transforms so that the result maps a point by `other`
    /// and then by `self`.
    pub fn compose(&self, other: &Self) -> Self {
        let new_rotation = self.rotation.compose(&other.rotation);
        let translated =
            self.translation.vector + self.rotation.rotation() * other.translation.vector;
        Self {
            rotation: new_rotation,
            translation: Translation3::from(translated),
        }
    }

    /// Compute the inverse rigid motion: \(T^{-1} = [R^T, -R^T t]\).
    pub fn inverse(&self) -> Self {
        let inv_rotation = self.rotation.inverse();
        let inv_translation = -(inv_rotation.rotation() * self.translation.vector);
        Self {
            rotation: inv_rotation,
            translation: Translation3::from(inv_translation),
        }
    }

    /// Apply the rigid transform to a 3D point (rotate, then translate).
    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        let point_vec = vector3_from_array(point);
        let rotated = self.rotation.rotation() * point_vec;
        let translated = rotated + self.translation.vector;
        vector3_to_array(&translated)
    }

    /// Export the 4×4 homogeneous transform matrix.
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

    /// Return the translation vector in \(\mathbb{R}^3\).
    pub fn translation(&self) -> [f64; 3] {
        vector3_to_array(&self.translation.vector)
    }

    /// Compute the adjoint representation \(\mathrm{Ad}_T\) that maps twists
    /// from the child frame into the parent frame.
    pub fn adjoint(&self) -> SMatrix<f64, 6, 6> {
        let rotation = self.rotation.rotation();
        let translation = self.translation();
        let translation_vec = vector3_from_array(translation);
        let skew = crate::util::skew_symmetric(&translation_vec);

        // The SE(3) adjoint has the block structure
        // \(\mathrm{Ad}_T = \begin{bmatrix} R & [t]_\times R \\ 0 & R \end{bmatrix}\)\,
        // where \(R\) is the rotation and \([t]_\times\) is the skew-symmetric
        // matrix built from the translation vector \(t\).

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
