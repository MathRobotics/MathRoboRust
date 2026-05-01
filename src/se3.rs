use nalgebra::{Matrix3, Matrix4, Rotation3, SMatrix, SVector, Translation3, Vector3};
use std::ops::Mul;

use crate::{
    lie::{HasAdjoint, LieGroup, matrix_exp_integral_series, matrix_to_array},
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
    pub const fn dof() -> usize {
        6
    }

    pub const fn mat_size() -> usize {
        4
    }

    pub const fn mat_adj_size() -> usize {
        6
    }

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

        let rotation_matrix = [
            [mat[(0, 0)], mat[(0, 1)], mat[(0, 2)]],
            [mat[(1, 0)], mat[(1, 1)], mat[(1, 2)]],
            [mat[(2, 0)], mat[(2, 1)], mat[(2, 2)]],
        ];
        let translation = [mat[(0, 3)], mat[(1, 3)], mat[(2, 3)]];

        Self::from_parts(So3::from_matrix(rotation_matrix), translation)
    }

    /// Reconstruct an SE(3) element from its 6×6 adjoint matrix.
    pub fn from_adjoint_matrix(matrix: [[f64; 6]; 6]) -> Self {
        let rotation_matrix = [
            [
                0.5 * (matrix[0][0] + matrix[3][3]),
                0.5 * (matrix[0][1] + matrix[3][4]),
                0.5 * (matrix[0][2] + matrix[3][5]),
            ],
            [
                0.5 * (matrix[1][0] + matrix[4][3]),
                0.5 * (matrix[1][1] + matrix[4][4]),
                0.5 * (matrix[1][2] + matrix[4][5]),
            ],
            [
                0.5 * (matrix[2][0] + matrix[5][3]),
                0.5 * (matrix[2][1] + matrix[5][4]),
                0.5 * (matrix[2][2] + matrix[5][5]),
            ],
        ];

        let rotation = So3::from_matrix(rotation_matrix);
        let rotation_t = rotation.rotation().matrix().transpose();
        let bottom_left = Matrix3::from_row_slice(&[
            matrix[3][0],
            matrix[3][1],
            matrix[3][2],
            matrix[4][0],
            matrix[4][1],
            matrix[4][2],
            matrix[5][0],
            matrix[5][1],
            matrix[5][2],
        ]);
        let position_hat = bottom_left * rotation_t;
        let translation = So3::vee(matrix_to_array(&position_hat));

        Self::from_parts(rotation, translation)
    }

    /// Construct the Lie-algebra hat operator mapping a 6D twist vector
    /// into a 4×4 matrix in `se(3)`.
    pub fn hat(twist: [f64; 6]) -> [[f64; 4]; 4] {
        let omega = Vector3::new(twist[0], twist[1], twist[2]);
        let v = Vector3::new(twist[3], twist[4], twist[5]);
        let mut matrix = Matrix4::<f64>::zeros();
        let skew = crate::util::skew_symmetric(&omega);

        for row in 0..3 {
            for col in 0..3 {
                matrix[(row, col)] = skew[(row, col)];
            }
            matrix[(row, 3)] = v[row];
        }

        matrix_to_array(&matrix)
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

    /// Construct the adjoint-space hat operator on a 6D twist vector.
    pub fn hat_adj(twist: [f64; 6]) -> [[f64; 6]; 6] {
        let omega_hat = So3::hat([twist[0], twist[1], twist[2]]);
        let v_hat = So3::hat([twist[3], twist[4], twist[5]]);
        let mut matrix = SMatrix::<f64, 6, 6>::zeros();

        for row in 0..3 {
            for col in 0..3 {
                matrix[(row, col)] = omega_hat[row][col];
                matrix[(row + 3, col)] = v_hat[row][col];
                matrix[(row + 3, col + 3)] = omega_hat[row][col];
            }
        }

        matrix_to_array(&matrix)
    }

    /// The element-space hat-commute operator for SE(3).
    pub fn hat_commute(twist: [f64; 6]) -> [[f64; 6]; 4] {
        let omega_hat = So3::hat([twist[0], twist[1], twist[2]]);
        let mut matrix = [[0.0_f64; 6]; 4];
        for row in 0..3 {
            for col in 0..3 {
                matrix[row][col] = -omega_hat[row][col];
            }
        }
        matrix
    }

    /// The adjoint-space hat-commute operator for SE(3).
    pub fn hat_commute_adj(twist: [f64; 6]) -> [[f64; 6]; 6] {
        let hat = Self::hat_adj(twist);
        let mut out = [[0.0_f64; 6]; 6];
        for row in 0..6 {
            for col in 0..6 {
                out[row][col] = -hat[row][col];
            }
        }
        out
    }

    /// Inverse of [`Se3::hat_adj`], recovering a 6D twist from the adjoint
    /// representation matrix.
    pub fn vee_adj(matrix: [[f64; 6]; 6]) -> [f64; 6] {
        let omega_top = [
            [matrix[0][0], matrix[0][1], matrix[0][2]],
            [matrix[1][0], matrix[1][1], matrix[1][2]],
            [matrix[2][0], matrix[2][1], matrix[2][2]],
        ];
        let omega_bottom = [
            [matrix[3][3], matrix[3][4], matrix[3][5]],
            [matrix[4][3], matrix[4][4], matrix[4][5]],
            [matrix[5][3], matrix[5][4], matrix[5][5]],
        ];
        let v_block = [
            [matrix[3][0], matrix[3][1], matrix[3][2]],
            [matrix[4][0], matrix[4][1], matrix[4][2]],
            [matrix[5][0], matrix[5][1], matrix[5][2]],
        ];

        let omega_a = So3::vee(omega_top);
        let omega_b = So3::vee(omega_bottom);
        let v = So3::vee(v_block);

        [
            0.5 * (omega_a[0] + omega_b[0]),
            0.5 * (omega_a[1] + omega_b[1]),
            0.5 * (omega_a[2] + omega_b[2]),
            v[0],
            v[1],
            v[2],
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
        for row in 0..3 {
            for col in 0..3 {
                matrix[(row, col)] = rotation_matrix[(row, col)];
            }
            matrix[(row, 3)] = translated[row];
        }

        matrix_to_array(&matrix)
    }

    /// Compute the vendor-compatible integrated SE(3) exponential.
    pub fn exp_integ(twist: [f64; 6], a: Option<f64>) -> [[f64; 4]; 4] {
        let scale = a.unwrap_or(1.0);
        let rotation = So3::exp_integ([twist[0], twist[1], twist[2]], Some(scale));
        let v_matrix = So3::exp_integ2nd([twist[0], twist[1], twist[2]], Some(scale));
        let v_matrix = Matrix3::from_row_slice(&[
            v_matrix[0][0],
            v_matrix[0][1],
            v_matrix[0][2],
            v_matrix[1][0],
            v_matrix[1][1],
            v_matrix[1][2],
            v_matrix[2][0],
            v_matrix[2][1],
            v_matrix[2][2],
        ]);
        let translation = v_matrix * Vector3::new(twist[3], twist[4], twist[5]);

        let mut matrix = Matrix4::<f64>::zeros();
        for row in 0..3 {
            for col in 0..3 {
                matrix[(row, col)] = rotation[row][col];
            }
            matrix[(row, 3)] = translation[row];
        }
        matrix[(3, 3)] = 1.0;

        matrix_to_array(&matrix)
    }

    /// Exponential in the adjoint representation.
    pub fn exp_adj(twist: [f64; 6], a: Option<f64>) -> [[f64; 6]; 6] {
        let transform = Self::from_matrix(Self::exp(twist, a));
        matrix_to_array(&transform.adjoint())
    }

    /// Compute \(\int_0^a \exp(s \operatorname{ad}_{\xi}) ds\).
    pub fn exp_integ_adj(twist: [f64; 6], a: Option<f64>) -> [[f64; 6]; 6] {
        let upper = a.unwrap_or(1.0);
        let hat_adj = Self::hat_adj(twist);
        let generator = SMatrix::<f64, 6, 6>::from_row_slice(&[
            hat_adj[0][0],
            hat_adj[0][1],
            hat_adj[0][2],
            hat_adj[0][3],
            hat_adj[0][4],
            hat_adj[0][5],
            hat_adj[1][0],
            hat_adj[1][1],
            hat_adj[1][2],
            hat_adj[1][3],
            hat_adj[1][4],
            hat_adj[1][5],
            hat_adj[2][0],
            hat_adj[2][1],
            hat_adj[2][2],
            hat_adj[2][3],
            hat_adj[2][4],
            hat_adj[2][5],
            hat_adj[3][0],
            hat_adj[3][1],
            hat_adj[3][2],
            hat_adj[3][3],
            hat_adj[3][4],
            hat_adj[3][5],
            hat_adj[4][0],
            hat_adj[4][1],
            hat_adj[4][2],
            hat_adj[4][3],
            hat_adj[4][4],
            hat_adj[4][5],
            hat_adj[5][0],
            hat_adj[5][1],
            hat_adj[5][2],
            hat_adj[5][3],
            hat_adj[5][4],
            hat_adj[5][5],
        ]);
        matrix_to_array(&matrix_exp_integral_series(&generator, upper, 1))
    }

    pub fn from_parts(rotation: So3, translation: [f64; 3]) -> Self {
        Self {
            rotation,
            translation: Translation3::new(translation[0], translation[1], translation[2]),
        }
    }

    /// Construct an SE(3) transform from an axis–angle rotation and a
    /// translation vector.
    pub fn from_axis_angle_translation(axis: [f64; 3], angle: f64, translation: [f64; 3]) -> Self {
        let rotation = So3::from_axis_angle(axis, angle);
        Self::from_parts(rotation, translation)
    }

    /// Construct an SE(3) transform from a translation and quaternion.
    pub fn from_pos_quaternion(position: [f64; 3], quaternion: [f64; 4]) -> Self {
        Self::from_parts(So3::from_quaternion(quaternion), position)
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

    /// Return the translation and quaternion tuple used by the Python API.
    pub fn pos_quaternion(&self) -> ([f64; 3], [f64; 4]) {
        (self.translation(), self.rotation.to_quaternion())
    }

    /// Compute the adjoint representation \(\mathrm{Ad}_T\) that maps twists
    /// from the child frame into the parent frame.
    pub fn adjoint(&self) -> SMatrix<f64, 6, 6> {
        let rotation = self.rotation.rotation();
        let translation_vec = vector3_from_array(self.translation());
        let skew = crate::util::skew_symmetric(&translation_vec);

        let mut matrix = SMatrix::<f64, 6, 6>::zeros();
        for row in 0..3 {
            for col in 0..3 {
                matrix[(row, col)] = rotation.matrix()[(row, col)];
                matrix[(row + 3, col + 3)] = rotation.matrix()[(row, col)];
                matrix[(row + 3, col)] = (skew * rotation.matrix())[(row, col)];
            }
        }

        matrix
    }

    /// Approximate the tangent-space difference between two nearby transforms.
    pub fn sub_tan_vec(val0: &Self, val1: &Self, frame: Option<&str>) -> [f64; 6] {
        let angular = So3::sub_tan_vec(&val0.rotation, &val1.rotation, frame);
        let translation = match frame.unwrap_or("bframe") {
            "bframe" => {
                val0.rotation.rotation().matrix().transpose()
                    * (vector3_from_array(val1.translation())
                        - vector3_from_array(val0.translation()))
            }
            "fframe" => {
                let tmp = (val1.rotation.as_matrix() - val0.rotation.as_matrix())
                    * val0.rotation.as_matrix().transpose();
                vector3_from_array(val1.translation())
                    - vector3_from_array(val0.translation())
                    - tmp * vector3_from_array(val0.translation())
            }
            other => panic!("Unsupported frame: {other}"),
        };

        [
            angular[0],
            angular[1],
            angular[2],
            translation[0],
            translation[1],
            translation[2],
        ]
    }

    /// First-order variation of `Ad_T * arb_vec` with respect to a tangent
    /// perturbation.
    pub fn mat_var_x_arb_vec(
        &self,
        arb_vec: [f64; 6],
        tan_var_vec: [f64; 6],
        frame: Option<&str>,
    ) -> [f64; 6] {
        let jacobian = self.mat_var_x_arb_vec_jacob(arb_vec, frame);
        let jacobian = SMatrix::<f64, 6, 6>::from_row_slice(&[
            jacobian[0][0],
            jacobian[0][1],
            jacobian[0][2],
            jacobian[0][3],
            jacobian[0][4],
            jacobian[0][5],
            jacobian[1][0],
            jacobian[1][1],
            jacobian[1][2],
            jacobian[1][3],
            jacobian[1][4],
            jacobian[1][5],
            jacobian[2][0],
            jacobian[2][1],
            jacobian[2][2],
            jacobian[2][3],
            jacobian[2][4],
            jacobian[2][5],
            jacobian[3][0],
            jacobian[3][1],
            jacobian[3][2],
            jacobian[3][3],
            jacobian[3][4],
            jacobian[3][5],
            jacobian[4][0],
            jacobian[4][1],
            jacobian[4][2],
            jacobian[4][3],
            jacobian[4][4],
            jacobian[4][5],
            jacobian[5][0],
            jacobian[5][1],
            jacobian[5][2],
            jacobian[5][3],
            jacobian[5][4],
            jacobian[5][5],
        ]);
        let result = jacobian * SVector::<f64, 6>::from_row_slice(&tan_var_vec);
        result.into()
    }

    /// Jacobian of the first-order variation of `Ad_T * arb_vec`.
    pub fn mat_var_x_arb_vec_jacob(&self, arb_vec: [f64; 6], frame: Option<&str>) -> [[f64; 6]; 6] {
        let matrix = match frame.unwrap_or("bframe") {
            "bframe" => {
                let commute =
                    SMatrix::<f64, 6, 6>::from_row_slice(&Self::hat_commute_adj(arb_vec).concat());
                self.adjoint() * commute
            }
            "fframe" => {
                let transformed = self.adjoint() * SVector::<f64, 6>::from_row_slice(&arb_vec);
                SMatrix::<f64, 6, 6>::from_row_slice(
                    &Self::hat_commute_adj(transformed.into()).concat(),
                )
            }
            other => panic!("Unsupported frame: {other}"),
        };
        matrix_to_array(&matrix)
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
        matrix
    }
}

impl HasAdjoint<6> for Se3 {
    fn adjoint_matrix(&self) -> SMatrix<f64, 6, 6> {
        self.adjoint()
    }
}

impl Mul for Se3 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.compose(&rhs)
    }
}

impl<'a> Mul<&'a Se3> for Se3 {
    type Output = Se3;

    fn mul(self, rhs: &'a Se3) -> Self::Output {
        self.compose(rhs)
    }
}

impl<'a> Mul<Se3> for &'a Se3 {
    type Output = Se3;

    fn mul(self, rhs: Se3) -> Self::Output {
        self.compose(&rhs)
    }
}

impl<'a, 'b> Mul<&'a Se3> for &'b Se3 {
    type Output = Se3;

    fn mul(self, rhs: &'a Se3) -> Self::Output {
        self.compose(rhs)
    }
}
