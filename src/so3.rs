use nalgebra::{Matrix3, Quaternion, Rotation3, UnitQuaternion};

use crate::lie::{LieGroup, apply_linear, matrix_to_array};
use crate::util::{skew_symmetric, vector3_from_array, vector3_to_array};

/// A 3D rotation represented as an element of the special orthogonal group
/// \(\mathrm{SO}(3)\).
#[derive(Debug, Clone, PartialEq)]
pub struct So3 {
    rotation: Rotation3<f64>,
}

impl So3 {
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

    /// Create the skew-symmetric matrix associated with a 3D vector.
    pub fn hat(vector: [f64; 3]) -> [[f64; 3]; 3] {
        matrix_to_array(&skew_symmetric(&vector3_from_array(vector)))
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
