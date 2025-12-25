use nalgebra::{SMatrix, SVector};

/// Common functionality shared by Lie groups that admit a matrix representation.
pub trait LieGroup<const MAT_DIM: usize>: Sized {
    fn identity() -> Self;
    fn compose(&self, other: &Self) -> Self;
    fn inverse(&self) -> Self;
    fn as_matrix(&self) -> SMatrix<f64, MAT_DIM, MAT_DIM>;
}

/// Provides the adjoint action `Ad_g` as a matrix on the Lie algebra.
pub trait HasAdjoint<const ADJ_DIM: usize> {
    fn adjoint_matrix(&self) -> SMatrix<f64, ADJ_DIM, ADJ_DIM>;
}

/// Apply a matrix-valued group action to a vector using static dimensions.
pub fn apply_linear<const DIM: usize>(
    matrix: &SMatrix<f64, DIM, DIM>,
    vector: [f64; DIM],
) -> [f64; DIM] {
    let vec = SVector::<f64, DIM>::from_row_slice(&vector);
    let result = matrix * vec;
    result.into()
}

/// Convert a statically sized matrix into a nested array for FFI-friendly use.
pub fn matrix_to_array<const DIM: usize>(matrix: &SMatrix<f64, DIM, DIM>) -> [[f64; DIM]; DIM] {
    let mut array = [[0.0_f64; DIM]; DIM];
    for r in 0..DIM {
        for c in 0..DIM {
            array[r][c] = matrix[(r, c)];
        }
    }
    array
}
