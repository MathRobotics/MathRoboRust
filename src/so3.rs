use nalgebra::{Matrix3, Quaternion, Rotation3, UnitQuaternion};
use std::ops::Mul;

use crate::lie::{HasAdjoint, LieGroup, apply_linear, matrix_to_array};
use crate::util::{skew_symmetric, vector3_from_array, vector3_to_array};

/// A 3D rotation represented as an element of the special orthogonal group
/// \(\mathrm{SO}(3)\).
#[derive(Debug, Clone, PartialEq)]
pub struct So3 {
    rotation: Rotation3<f64>,
}

impl So3 {
    pub const fn dof() -> usize {
        3
    }

    pub const fn mat_size() -> usize {
        3
    }

    pub const fn mat_adj_size() -> usize {
        3
    }

    /// Build an element of SO(3) from an axis and angle using Rodrigues'
    /// rotation formula. Zero-length axes fall back to the identity
    /// so the caller can safely pass unnormalized vectors.
    pub fn from_axis_angle(axis: [f64; 3], angle: f64) -> Self {
        let axis_vector = vector3_from_array(axis);
        if axis_vector.norm() == 0.0 {
            return Self::identity();
        }

        Self {
            rotation: Rotation3::new(axis_vector.normalize() * angle),
        }
    }

    /// Compose two rotations using matrix multiplication: \(R_1 R_2\).
    pub fn compose(&self, other: &Self) -> Self {
        Self {
            rotation: self.rotation * other.rotation,
        }
    }

    /// Construct a rotation directly from a 3×3 matrix. The input is assumed to
    /// already be a valid rotation matrix; no orthonormality checks are
    /// performed.
    pub fn from_matrix(matrix: [[f64; 3]; 3]) -> Self {
        let flat: [f64; 9] = [
            matrix[0][0],
            matrix[0][1],
            matrix[0][2],
            matrix[1][0],
            matrix[1][1],
            matrix[1][2],
            matrix[2][0],
            matrix[2][1],
            matrix[2][2],
        ];
        let mat = Matrix3::from_row_slice(&flat);
        Self {
            rotation: Rotation3::from_matrix_unchecked(mat),
        }
    }

    /// Return the inverse rotation, i.e. the transpose of the rotation matrix.
    pub fn inverse(&self) -> Self {
        Self {
            rotation: self.rotation.inverse(),
        }
    }

    /// Apply the rotation to a 3D vector.
    pub fn apply(&self, vector: [f64; 3]) -> [f64; 3] {
        apply_linear(&self.rotation.matrix().clone_owned(), vector)
    }

    /// Construct an SO(3) element from a unit quaternion specified as
    /// \([w, x, y, z]\). The quaternion is normalized before use so callers do
    /// not need to pre-normalize inputs.
    pub fn from_quaternion(quaternion: [f64; 4]) -> Self {
        let mut quat = Quaternion::new(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
        if quat.norm() != 0.0 {
            quat = quat.normalize();
        }
        Self {
            rotation: UnitQuaternion::from_quaternion(quat).to_rotation_matrix(),
        }
    }

    /// Export the rotation as a normalized quaternion \([w, x, y, z]\).
    pub fn to_quaternion(&self) -> [f64; 4] {
        let unit = UnitQuaternion::from_rotation_matrix(&self.rotation);
        let quat = unit.quaternion();
        [quat.w, quat.i, quat.j, quat.k]
    }

    /// Build a rotation from roll–pitch–yaw angles applied in ZYX order.
    pub fn from_euler_angles(roll: f64, pitch: f64, yaw: f64) -> Self {
        Self {
            rotation: Rotation3::from_euler_angles(roll, pitch, yaw),
        }
    }

    /// Return the roll–pitch–yaw angles (ZYX order) that generate this
    /// rotation.
    pub fn to_euler_angles(&self) -> (f64, f64, f64) {
        self.rotation.euler_angles()
    }

    /// Build a rotation directly from the so(3) tangent vector using the
    /// exponential map.
    pub fn from_rotation_vector(vector: [f64; 3]) -> Self {
        let axis_angle = vector3_from_array(vector);
        Self {
            rotation: Rotation3::new(axis_angle),
        }
    }

    /// Recover the tangent vector representation (logarithm map) using the
    /// Rodrigues rotation vector.
    pub fn to_rotation_vector(&self) -> [f64; 3] {
        vector3_to_array(&self.rotation.scaled_axis())
    }

    /// Compute the exponential map from a tangent vector to an SO(3) rotation
    /// matrix. The optional scale factor `a` scales the tangent vector prior to
    /// exponentiation.
    pub fn exp(vector: [f64; 3], a: Option<f64>) -> [[f64; 3]; 3] {
        let scale = a.unwrap_or(1.0);
        Self::from_rotation_vector([vector[0] * scale, vector[1] * scale, vector[2] * scale])
            .to_matrix()
    }

    /// Compute \(\int_0^a \exp(s \hat{\omega}) ds\).
    pub fn exp_integ(vector: [f64; 3], a: Option<f64>) -> [[f64; 3]; 3] {
        let scale = a.unwrap_or(1.0);
        let omega = vector3_from_array(vector);
        let theta = omega.norm();
        let hat = skew_symmetric(&omega);
        let hat_sq = hat * hat;

        let matrix = if theta.abs() < 1e-12 {
            Matrix3::<f64>::identity() * scale
                + hat * (0.5 * scale * scale)
                + hat_sq * (scale * scale * scale / 6.0)
        } else {
            let angle = theta * scale;
            let theta_sq = theta * theta;
            let theta_cu = theta_sq * theta;
            Matrix3::<f64>::identity() * scale
                + hat * ((1.0 - angle.cos()) / theta_sq)
                + hat_sq * ((angle - angle.sin()) / theta_cu)
        };

        matrix_to_array(&matrix)
    }

    /// Compute \(\int_0^a \int_0^s \exp(u \hat{\omega}) du\, ds\).
    pub fn exp_integ2nd(vector: [f64; 3], a: Option<f64>) -> [[f64; 3]; 3] {
        let scale = a.unwrap_or(1.0);
        let omega = vector3_from_array(vector);
        let theta = omega.norm();
        let hat = skew_symmetric(&omega);
        let hat_sq = hat * hat;

        let matrix = if theta.abs() < 1e-12 {
            Matrix3::<f64>::identity() * scale
        } else {
            let angle = theta * scale;
            let theta_sq = theta * theta;
            let theta_cu = theta_sq * theta;
            let theta_4 = theta_sq * theta_sq;
            Matrix3::<f64>::identity() * ((1.0 - angle.cos()) / theta_sq)
                + hat * ((angle - angle.sin()) / theta_cu)
                + hat_sq * ((0.5 * angle * angle - 1.0 + angle.cos()) / theta_4)
        };

        matrix_to_array(&matrix)
    }

    /// The adjoint exponential map for SO(3), identical to [`So3::exp`].
    pub fn exp_adj(vector: [f64; 3], a: Option<f64>) -> [[f64; 3]; 3] {
        Self::exp(vector, a)
    }

    /// The adjoint exponential integral for SO(3), identical to
    /// [`So3::exp_integ`].
    pub fn exp_integ_adj(vector: [f64; 3], a: Option<f64>) -> [[f64; 3]; 3] {
        Self::exp_integ(vector, a)
    }

    /// Create the skew-symmetric matrix associated with a 3D vector.
    pub fn hat(vector: [f64; 3]) -> [[f64; 3]; 3] {
        matrix_to_array(&skew_symmetric(&vector3_from_array(vector)))
    }

    /// The adjoint-space hat operator for SO(3), identical to [`So3::hat`].
    pub fn hat_adj(vector: [f64; 3]) -> [[f64; 3]; 3] {
        Self::hat(vector)
    }

    /// The element-space hat-commute operator for SO(3).
    pub fn hat_commute(vector: [f64; 3]) -> [[f64; 3]; 3] {
        Self::hat_commute_adj(vector)
    }

    /// The adjoint-space hat-commute operator for SO(3).
    pub fn hat_commute_adj(vector: [f64; 3]) -> [[f64; 3]; 3] {
        let hat = Self::hat_adj(vector);
        let mut out = [[0.0_f64; 3]; 3];
        for row in 0..3 {
            for col in 0..3 {
                out[row][col] = -hat[row][col];
            }
        }
        out
    }

    /// Recover the vector that generated a skew-symmetric matrix. The inputs do
    /// not need to be perfectly skew-symmetric; the off-diagonal elements are
    /// symmetrized.
    pub fn vee(matrix: [[f64; 3]; 3]) -> [f64; 3] {
        let flat: [f64; 9] = [
            matrix[0][0],
            matrix[0][1],
            matrix[0][2],
            matrix[1][0],
            matrix[1][1],
            matrix[1][2],
            matrix[2][0],
            matrix[2][1],
            matrix[2][2],
        ];
        let mat = Matrix3::from_row_slice(&flat);
        [
            0.5 * (mat[(2, 1)] - mat[(1, 2)]),
            0.5 * (mat[(0, 2)] - mat[(2, 0)]),
            0.5 * (mat[(1, 0)] - mat[(0, 1)]),
        ]
    }

    /// The adjoint-space vee operator for SO(3), identical to [`So3::vee`].
    pub fn vee_adj(matrix: [[f64; 3]; 3]) -> [f64; 3] {
        Self::vee(matrix)
    }

    /// Approximate the tangent-space difference between two nearby rotations.
    pub fn sub_tan_vec(val0: &Self, val1: &Self, frame: Option<&str>) -> [f64; 3] {
        let diff = val1.as_matrix() - val0.as_matrix();
        let matrix = match frame.unwrap_or("bframe") {
            "bframe" => val0.inverse().as_matrix() * diff,
            "fframe" => diff * val0.inverse().as_matrix(),
            other => panic!("Unsupported frame: {other}"),
        };
        Self::vee(matrix_to_array(&matrix))
    }

    /// First-order variation of `R * arb_vec` with respect to a tangent
    /// perturbation.
    pub fn mat_var_x_arb_vec(
        &self,
        arb_vec: [f64; 3],
        tan_var_vec: [f64; 3],
        frame: Option<&str>,
    ) -> [f64; 3] {
        let jacobian = self.mat_var_x_arb_vec_jacob(arb_vec, frame);
        let jacobian = Matrix3::<f64>::from_row_slice(&[
            jacobian[0][0],
            jacobian[0][1],
            jacobian[0][2],
            jacobian[1][0],
            jacobian[1][1],
            jacobian[1][2],
            jacobian[2][0],
            jacobian[2][1],
            jacobian[2][2],
        ]);
        apply_linear(&jacobian, tan_var_vec)
    }

    /// Jacobian of the first-order variation of `R * arb_vec`.
    pub fn mat_var_x_arb_vec_jacob(&self, arb_vec: [f64; 3], frame: Option<&str>) -> [[f64; 3]; 3] {
        let matrix = match frame.unwrap_or("bframe") {
            "bframe" => {
                let commute =
                    Matrix3::<f64>::from_row_slice(&Self::hat_commute_adj(arb_vec).concat());
                self.as_matrix() * commute
            }
            "fframe" => {
                Matrix3::<f64>::from_row_slice(&Self::hat_commute_adj(self.apply(arb_vec)).concat())
            }
            other => panic!("Unsupported frame: {other}"),
        };
        matrix_to_array(&matrix)
    }

    /// Export the underlying 3×3 rotation matrix.
    pub fn to_matrix(&self) -> [[f64; 3]; 3] {
        matrix_to_array(&self.rotation.matrix().clone_owned())
    }

    /// Access the nalgebra `Rotation3` backing this object.
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

impl HasAdjoint<3> for So3 {
    fn adjoint_matrix(&self) -> nalgebra::SMatrix<f64, 3, 3> {
        self.rotation.matrix().clone_owned()
    }
}

impl Mul for So3 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.compose(&rhs)
    }
}

impl<'a> Mul<&'a So3> for So3 {
    type Output = So3;

    fn mul(self, rhs: &'a So3) -> Self::Output {
        self.compose(rhs)
    }
}

impl<'a> Mul<So3> for &'a So3 {
    type Output = So3;

    fn mul(self, rhs: So3) -> Self::Output {
        self.compose(&rhs)
    }
}

impl<'a, 'b> Mul<&'a So3> for &'b So3 {
    type Output = So3;

    fn mul(self, rhs: &'a So3) -> Self::Output {
        self.compose(rhs)
    }
}
