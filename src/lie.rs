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

/// Compute repeated integrals of a matrix exponential via the power series
/// \(\sum_{n=0}^{\infty} \frac{a^{n+k}}{(n+k)!} A^n\), where `k` is the
/// integral order (1 for \(\int e^{sA} ds\), 2 for \(\int\int e^{sA} ds^2\)).
pub fn matrix_exp_integral_series<const DIM: usize>(
    generator: &SMatrix<f64, DIM, DIM>,
    upper: f64,
    integral_order: usize,
) -> SMatrix<f64, DIM, DIM> {
    assert!(integral_order > 0, "integral_order must be positive");

    let mut scale = 1.0_f64;
    for denom in 1..=integral_order {
        scale *= upper / denom as f64;
    }

    let mut term = SMatrix::<f64, DIM, DIM>::identity() * scale;
    let mut sum = term;

    for index in 0..63 {
        let denom = (integral_order + index + 1) as f64;
        term = term * generator * (upper / denom);
        sum += term;

        if term.norm() <= 1e-14 {
            break;
        }
    }

    sum
}
