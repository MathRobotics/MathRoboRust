use nalgebra::{SMatrix, SVector};

use crate::lie::HasAdjoint;
use crate::se3::Se3;

pub type Matrix6 = SMatrix<f64, 6, 6>;
pub type Vector6 = SVector<f64, 6>;

#[derive(Debug, Clone, PartialEq)]
pub struct Cmtm {
    matrix: Matrix6,
}

impl Cmtm {
    pub fn from_se3(transform: &Se3) -> Self {
        Self {
            matrix: transform.adjoint_matrix(),
        }
    }

    pub fn identity() -> Self {
        Self {
            matrix: Matrix6::identity(),
        }
    }

    pub fn apply_twist(&self, twist: [f64; 6]) -> [f64; 6] {
        let twist_vec = Vector6::from_row_slice(&twist);
        let result = &self.matrix * twist_vec;
        let array: [f64; 6] = result.into();
        array
    }

    pub fn to_matrix(&self) -> [[f64; 6]; 6] {
        let mut array = [[0.0_f64; 6]; 6];
        for r in 0..6 {
            for c in 0..6 {
                array[r][c] = self.matrix[(r, c)];
            }
        }
        array
    }

    pub fn matrix(&self) -> &Matrix6 {
        &self.matrix
    }
}
