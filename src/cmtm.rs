use nalgebra::{DMatrix, SMatrix, SVector};
use std::ops::Mul;

use crate::lie::{apply_linear, matrix_to_array, HasAdjoint};
use crate::se3::Se3;
use crate::so3::So3;
use crate::util::{skew_symmetric, vector3_from_array};

pub type Matrix6 = SMatrix<f64, 6, 6>;
pub type Vector6 = SVector<f64, 6>;

/// Composite Motion Transformation Matrix (CMTM) used to move spatial
/// velocities between coordinate frames.
#[derive(Debug, Clone, PartialEq)]
pub struct GenericCmtm<const DIM: usize> {
    matrix: SMatrix<f64, DIM, DIM>,
    derivatives: Vec<SVector<f64, DIM>>, // holds time-derivative pseudo tangent vectors up to order n-1
}

/// Convenience alias for the spatial (SE(3)) adjoint representation.
pub type SpatialCmtm = GenericCmtm<6>;
/// Convenience alias for the rotational (SO(3)) adjoint representation.
pub type RotationalCmtm = GenericCmtm<3>;
/// Backwards-compatible alias that keeps the original 6×6 CMTM name.
pub type Cmtm6 = SpatialCmtm;
pub type Cmtm = SpatialCmtm;

impl<const DIM: usize> GenericCmtm<DIM> {
    /// The identity transformation, which leaves tangent vectors unchanged.
    pub fn identity() -> Self {
        Self {
            matrix: SMatrix::<f64, DIM, DIM>::identity(),
            derivatives: Vec::new(),
        }
    }

    /// Construct a CMTM from an adjoint matrix and a list of derivative vectors
    /// (up to the \(n-1\)-th derivative). The resulting block matrix supports
    /// orders up to `derivatives.len() + 1`.
    pub fn with_derivatives(
        matrix: SMatrix<f64, DIM, DIM>,
        derivatives: Vec<[f64; DIM]>,
    ) -> Self {
        Self {
            matrix,
            derivatives: derivatives
                .into_iter()
                .map(|v| SVector::<f64, DIM>::from_row_slice(&v))
                .collect(),
        }
    }

    /// Export the transformation matrix as a nested array for FFI use.
    pub fn to_matrix(&self) -> [[f64; DIM]; DIM] {
        matrix_to_array(&self.matrix)
    }

    /// Borrow the internal nalgebra matrix for advanced operations.
    pub fn matrix(&self) -> &SMatrix<f64, DIM, DIM> {
        &self.matrix
    }

    /// Highest derivative order supported by this CMTM (1 means only the base matrix).
    pub fn order(&self) -> usize {
        self.derivatives.len() + 1
    }

    fn check_output_order(&self, output_order: Option<usize>) -> usize {
        match output_order {
            Some(o) if o == 0 => panic!("Output order must be positive"),
            Some(o) if o > self.order() => panic!("Output order exceeds available derivatives"),
            Some(o) => o,
            None => self.order(),
        }
    }

    fn factorial(n: usize) -> f64 {
        (1..=n).fold(1.0, |acc, v| acc * v as f64)
    }
}

impl GenericCmtm<3> {
    /// Build the composite motion transformation matrix directly from an SO(3)
    /// rotation. This is the 3×3 adjoint representation that maps angular
    /// velocities between frames.
    pub fn from_so3(rotation: &So3) -> Self {
        let matrix = rotation.rotation().matrix().clone_owned();
        Self {
            matrix,
            derivatives: Vec::new(),
        }
    }

    /// Create an SO(3) CMTM that tracks derivatives up to order `n`.
    pub fn from_so3_with_derivatives(rotation: &So3, derivatives: Vec<[f64; 3]>) -> Self {
        let matrix = rotation.rotation().matrix().clone_owned();
        Self::with_derivatives(matrix, derivatives)
    }

    /// Apply the 3×3 transformation to an angular velocity vector.
    pub fn apply_omega(&self, omega: [f64; 3]) -> [f64; 3] {
        apply_linear(&self.matrix, omega)
    }

    /// Build the block CMTM up to the requested derivative order.
    pub fn to_block_matrix(&self, output_order: Option<usize>) -> DMatrix<f64> {
        self.build_block_matrix(output_order)
    }
}

impl GenericCmtm<6> {
    /// Build the composite motion transformation matrix from an SE(3)
    /// transform. This corresponds to the adjoint representation that maps
    /// spatial twists between frames.
    pub fn from_se3(transform: &Se3) -> Self {
        Self {
            matrix: transform.adjoint_matrix(),
            derivatives: Vec::new(),
        }
    }

    /// Create an SE(3) CMTM that tracks derivatives up to order `n`.
    pub fn from_se3_with_derivatives(transform: &Se3, derivatives: Vec<[f64; 6]>) -> Self {
        let matrix = transform.adjoint_matrix();
        Self::with_derivatives(matrix, derivatives)
    }

    /// Apply the 6×6 transformation to a twist vector \([\omega, v]\), returning
    /// the transformed angular and linear velocity components.
    pub fn apply_twist(&self, twist: [f64; 6]) -> [f64; 6] {
        apply_linear(&self.matrix, twist)
    }

    /// Build the block CMTM up to the requested derivative order.
    pub fn to_block_matrix(&self, output_order: Option<usize>) -> DMatrix<f64> {
        self.build_block_matrix(output_order)
    }
}

impl<const DIM: usize> GenericCmtm<DIM> {
    fn hat_adj(&self, vec: &SVector<f64, DIM>) -> SMatrix<f64, DIM, DIM> {
        match DIM {
            3 => {
                let omega_hat = skew_symmetric(&vector3_from_array([vec[0], vec[1], vec[2]]));
                let mut data = vec![0.0_f64; DIM * DIM];
                for r in 0..3 {
                    for c in 0..3 {
                        data[r * DIM + c] = omega_hat[(r, c)];
                    }
                }

                SMatrix::<f64, DIM, DIM>::from_row_slice(&data)
            }
            6 => {
                let omega = vector3_from_array([vec[0], vec[1], vec[2]]);
                let v = vector3_from_array([vec[3], vec[4], vec[5]]);
                let omega_hat = skew_symmetric(&omega);
                let v_hat = skew_symmetric(&v);

                let mut data = vec![0.0_f64; DIM * DIM];
                for r in 0..3 {
                    for c in 0..3 {
                        data[r * DIM + c] = omega_hat[(r, c)];
                        data[(r + 3) * DIM + (c + 3)] = omega_hat[(r, c)];
                        data[(r + 3) * DIM + c] = v_hat[(r, c)];
                    }
                }

                SMatrix::<f64, DIM, DIM>::from_row_slice(&data)
            }
            _ => panic!("hat_adj not implemented for dimension {DIM}"),
        }
    }

    /// Compose two CMTMs by multiplying their base matrices and pairing the
    /// derivative vectors order-wise. Missing derivative orders on either side
    /// are treated as zero, so the resulting order matches the larger operand.
    pub fn compose(&self, other: &Self) -> Self {
        let matrix = self.matrix * other.matrix;
        let max_order = usize::max(self.derivatives.len(), other.derivatives.len());

        let derivatives = (0..max_order)
            .map(|i| {
                let left = self
                    .derivatives
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| SVector::<f64, DIM>::zeros());
                let right = other
                    .derivatives
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| SVector::<f64, DIM>::zeros());
                left + right
            })
            .collect();

        Self { matrix, derivatives }
    }

    fn mat_elem(&self, p: usize) -> SMatrix<f64, DIM, DIM> {
        if p == 0 {
            return self.matrix;
        }

        let mut mat = SMatrix::<f64, DIM, DIM>::zeros();
        for i in 0..p {
            let prev = self.mat_elem(p - i - 1);
            let scaled = self.derivatives[i] / Self::factorial(i);
            let hat = self.hat_adj(&scaled);
            mat += prev * hat;
        }

        mat / p as f64
    }

    fn build_block_matrix(&self, output_order: Option<usize>) -> DMatrix<f64> {
        let order = self.check_output_order(output_order);
        let size = DIM * order;

        let mut mat = DMatrix::<f64>::zeros(size, size);
        let tmp: Vec<SMatrix<f64, DIM, DIM>> = (0..order).map(|i| self.mat_elem(i)).collect();

        for i in 0..order {
            for j in i..order {
                let row_offset = j * DIM;
                let col_offset = (j - i) * DIM;
                for r in 0..DIM {
                    for c in 0..DIM {
                        mat[(row_offset + r, col_offset + c)] = tmp[i][(r, c)];
                    }
                }
            }
        }

        mat
    }
}

impl<const DIM: usize> Mul for GenericCmtm<DIM> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.compose(&rhs)
    }
}

impl<'a, const DIM: usize> Mul<&'a GenericCmtm<DIM>> for GenericCmtm<DIM> {
    type Output = GenericCmtm<DIM>;

    fn mul(self, rhs: &'a GenericCmtm<DIM>) -> Self::Output {
        self.compose(rhs)
    }
}

impl<'a, const DIM: usize> Mul<GenericCmtm<DIM>> for &'a GenericCmtm<DIM> {
    type Output = GenericCmtm<DIM>;

    fn mul(self, rhs: GenericCmtm<DIM>) -> Self::Output {
        self.compose(&rhs)
    }
}

impl<'a, 'b, const DIM: usize> Mul<&'a GenericCmtm<DIM>> for &'b GenericCmtm<DIM> {
    type Output = GenericCmtm<DIM>;

    fn mul(self, rhs: &'a GenericCmtm<DIM>) -> Self::Output {
        self.compose(rhs)
    }
}
