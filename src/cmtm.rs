use nalgebra::{DMatrix, SMatrix, SVector};
use std::ops::Mul;

use crate::lie::{HasAdjoint, LieGroup, apply_linear, matrix_to_array};
use crate::se3::Se3;
use crate::so3::So3;

pub type Matrix6 = SMatrix<f64, 6, 6>;
pub type Vector6 = SVector<f64, 6>;

pub trait CmtmElement<const MAT_DIM: usize, const ADJ_DIM: usize>:
    LieGroup<MAT_DIM> + HasAdjoint<ADJ_DIM> + Clone + PartialEq
{
    fn from_matrix(matrix: [[f64; MAT_DIM]; MAT_DIM]) -> Self;
    fn from_adjoint_matrix(matrix: [[f64; ADJ_DIM]; ADJ_DIM]) -> Self;
    fn hat(vector: [f64; ADJ_DIM]) -> [[f64; MAT_DIM]; MAT_DIM];
    fn vee(matrix: [[f64; MAT_DIM]; MAT_DIM]) -> [f64; ADJ_DIM];
    fn hat_adj(vector: [f64; ADJ_DIM]) -> [[f64; ADJ_DIM]; ADJ_DIM];
    fn hat_commute_adj(vector: [f64; ADJ_DIM]) -> [[f64; ADJ_DIM]; ADJ_DIM];
    fn vee_adj(matrix: [[f64; ADJ_DIM]; ADJ_DIM]) -> [f64; ADJ_DIM];
}

/// Composite Motion Transformation Matrix (CMTM) that lifts a Lie-group element
/// and its higher-order tangent vectors into lower block-Toeplitz form.
#[derive(Debug, Clone, PartialEq)]
pub struct GenericCmtm<G, const MAT_DIM: usize, const ADJ_DIM: usize>
where
    G: CmtmElement<MAT_DIM, ADJ_DIM>,
{
    element: G,
    derivatives: Vec<SVector<f64, ADJ_DIM>>,
}

/// Convenience alias for the spatial (SE(3)) CMTM.
pub type SpatialCmtm = GenericCmtm<Se3, 4, 6>;
/// Convenience alias for the rotational (SO(3)) CMTM.
pub type RotationalCmtm = GenericCmtm<So3, 3, 3>;
/// Backwards-compatible aliases.
pub type Cmtm6 = SpatialCmtm;
pub type Cmtm = SpatialCmtm;

fn vector_to_array<const DIM: usize>(vector: &SVector<f64, DIM>) -> [f64; DIM] {
    std::array::from_fn(|idx| vector[idx])
}

fn smatrix_from_array<const DIM: usize>(matrix: [[f64; DIM]; DIM]) -> SMatrix<f64, DIM, DIM> {
    let mut flat = Vec::with_capacity(DIM * DIM);
    for row in matrix {
        flat.extend(row);
    }
    SMatrix::<f64, DIM, DIM>::from_row_slice(&flat)
}

fn lower_toeplitz<const DIM: usize>(blocks: &[SMatrix<f64, DIM, DIM>]) -> DMatrix<f64> {
    let order = blocks.len();
    let mut matrix = DMatrix::<f64>::zeros(DIM * order, DIM * order);
    for offset in 0..order {
        for block_row in offset..order {
            let row_offset = block_row * DIM;
            let col_offset = (block_row - offset) * DIM;
            for row in 0..DIM {
                for col in 0..DIM {
                    matrix[(row_offset + row, col_offset + col)] = blocks[offset][(row, col)];
                }
            }
        }
    }
    matrix
}

fn lower_triangular<const DIM: usize>(
    table: &[Vec<SMatrix<f64, DIM, DIM>>],
    col_scales: Option<&[f64]>,
) -> DMatrix<f64> {
    let order = table.len();
    let mut matrix = DMatrix::<f64>::zeros(DIM * order, DIM * order);
    for block_row in 0..order {
        for block_col in 0..=block_row {
            let scale = col_scales.map(|values| values[block_col]).unwrap_or(1.0);
            let row_offset = block_row * DIM;
            let col_offset = block_col * DIM;
            for row in 0..DIM {
                for col in 0..DIM {
                    matrix[(row_offset + row, col_offset + col)] =
                        table[block_row][block_col][(row, col)] * scale;
                }
            }
        }
    }
    matrix
}

fn averaged_blocks<const DIM: usize>(matrix: &DMatrix<f64>) -> Vec<SMatrix<f64, DIM, DIM>> {
    assert_eq!(matrix.nrows(), matrix.ncols(), "Matrix must be square");
    assert!(
        matrix.nrows() % DIM == 0,
        "Matrix size must be divisible by block dimension"
    );

    let order = matrix.nrows() / DIM;
    let mut blocks = Vec::with_capacity(order);
    for offset in 0..order {
        let mut block = SMatrix::<f64, DIM, DIM>::zeros();
        let count = (order - offset) as f64;
        for base in 0..(order - offset) {
            let row_offset = (base + offset) * DIM;
            let col_offset = base * DIM;
            for row in 0..DIM {
                for col in 0..DIM {
                    block[(row, col)] += matrix[(row_offset + row, col_offset + col)];
                }
            }
        }
        blocks.push(block / count);
    }
    blocks
}

fn hat_toeplitz<const ADJ_DIM: usize, const BLOCK_DIM: usize, F>(
    vecs: &[[f64; ADJ_DIM]],
    hat_fn: F,
) -> DMatrix<f64>
where
    F: Fn([f64; ADJ_DIM]) -> [[f64; BLOCK_DIM]; BLOCK_DIM],
{
    let order = vecs.len();
    let mut matrix = DMatrix::<f64>::zeros(BLOCK_DIM * order, BLOCK_DIM * order);
    for offset in 0..order {
        let block = smatrix_from_array(hat_fn(vecs[offset]));
        for block_row in offset..order {
            let row_offset = block_row * BLOCK_DIM;
            let col_offset = (block_row - offset) * BLOCK_DIM;
            for row in 0..BLOCK_DIM {
                for col in 0..BLOCK_DIM {
                    matrix[(row_offset + row, col_offset + col)] = block[(row, col)];
                }
            }
        }
    }
    matrix
}

fn vee_from_matrix<const ADJ_DIM: usize, const BLOCK_DIM: usize, F>(
    matrix: &DMatrix<f64>,
    vee_fn: F,
) -> Vec<[f64; ADJ_DIM]>
where
    F: Fn([[f64; BLOCK_DIM]; BLOCK_DIM]) -> [f64; ADJ_DIM],
{
    assert_eq!(matrix.nrows(), matrix.ncols(), "Matrix must be square");
    assert!(
        matrix.nrows() % BLOCK_DIM == 0,
        "Matrix size must be divisible by block dimension"
    );

    let order = matrix.nrows() / BLOCK_DIM;
    let mut out = Vec::with_capacity(order);
    for offset in 0..order {
        let mut acc = [0.0_f64; ADJ_DIM];
        let count = (order - offset) as f64;
        for base in offset..order {
            let row_offset = base * BLOCK_DIM;
            let col_offset = (base - offset) * BLOCK_DIM;
            let mut block = SMatrix::<f64, BLOCK_DIM, BLOCK_DIM>::zeros();
            for row in 0..BLOCK_DIM {
                for col in 0..BLOCK_DIM {
                    block[(row, col)] = matrix[(row_offset + row, col_offset + col)];
                }
            }
            let vec = vee_fn(matrix_to_array(&block));
            for idx in 0..ADJ_DIM {
                acc[idx] += vec[idx];
            }
        }
        for value in &mut acc {
            *value /= count;
        }
        out.push(acc);
    }
    out
}

fn invert_matrix(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    matrix
        .clone()
        .try_inverse()
        .expect("Matrix should be invertible")
}

impl<G, const MAT_DIM: usize, const ADJ_DIM: usize> GenericCmtm<G, MAT_DIM, ADJ_DIM>
where
    G: CmtmElement<MAT_DIM, ADJ_DIM>,
{
    /// Identity CMTM with no higher-order derivatives.
    pub fn identity() -> Self {
        Self {
            element: G::identity(),
            derivatives: Vec::new(),
        }
    }

    /// Construct a CMTM directly from an element and derivative vectors.
    pub fn with_derivatives(element: G, derivatives: Vec<[f64; ADJ_DIM]>) -> Self {
        Self {
            element,
            derivatives: derivatives
                .into_iter()
                .map(|vector| SVector::<f64, ADJ_DIM>::from_row_slice(&vector))
                .collect(),
        }
    }

    /// Borrow the underlying Lie-group element.
    pub fn element(&self) -> &G {
        &self.element
    }

    /// Highest supported output order.
    pub fn order(&self) -> usize {
        self.derivatives.len() + 1
    }

    /// Element matrix of the base Lie-group element.
    pub fn elem_mat(&self) -> [[f64; MAT_DIM]; MAT_DIM] {
        matrix_to_array(&self.element.as_matrix())
    }

    /// Backwards-compatible accessor for the base adjoint matrix.
    pub fn to_matrix(&self) -> [[f64; ADJ_DIM]; ADJ_DIM] {
        self.to_adjoint_matrix()
    }

    /// Base element matrix as an array.
    pub fn to_elem_matrix(&self) -> [[f64; MAT_DIM]; MAT_DIM] {
        self.elem_mat()
    }

    /// Base adjoint matrix as an array.
    pub fn to_adjoint_matrix(&self) -> [[f64; ADJ_DIM]; ADJ_DIM] {
        matrix_to_array(&self.element.adjoint_matrix())
    }

    /// Base adjoint matrix as a statically sized nalgebra matrix.
    pub fn matrix(&self) -> SMatrix<f64, ADJ_DIM, ADJ_DIM> {
        self.element.adjoint_matrix()
    }

    /// Return one stored derivative vector, if present.
    pub fn elem_vecs(&self, index: usize) -> Option<[f64; ADJ_DIM]> {
        self.derivatives.get(index).map(vector_to_array)
    }

    /// Return derivative vectors up to the requested output order.
    pub fn vecs(&self, output_order: Option<usize>) -> Vec<[f64; ADJ_DIM]> {
        let order = self.check_output_order(output_order);
        self.derivatives[..order.saturating_sub(1)]
            .iter()
            .map(vector_to_array)
            .collect()
    }

    /// Flatten derivative vectors up to the requested output order.
    pub fn vecs_flatten(&self, output_order: Option<usize>) -> Vec<f64> {
        self.vecs(output_order)
            .into_iter()
            .flat_map(|vector| vector.into_iter())
            .collect()
    }

    /// Lower block-Toeplitz matrix built from the element-space representation.
    pub fn mat(&self, output_order: Option<usize>) -> DMatrix<f64> {
        lower_toeplitz(&self.mat_blocks(self.check_output_order(output_order)))
    }

    /// Lower block-Toeplitz matrix built from the adjoint representation.
    pub fn mat_adj(&self, output_order: Option<usize>) -> DMatrix<f64> {
        lower_toeplitz(&self.mat_adj_blocks(self.check_output_order(output_order)))
    }

    /// Lower block-Toeplitz inverse matrix in the element-space representation.
    pub fn mat_inv(&self, output_order: Option<usize>) -> DMatrix<f64> {
        lower_toeplitz(&self.mat_inv_blocks(self.check_output_order(output_order)))
    }

    /// Lower block-Toeplitz inverse matrix in the adjoint representation.
    pub fn mat_inv_adj(&self, output_order: Option<usize>) -> DMatrix<f64> {
        lower_toeplitz(&self.mat_inv_adj_blocks(self.check_output_order(output_order)))
    }

    /// Reconstruct a CMTM from its lower block-Toeplitz element matrix.
    pub fn set_mat(matrix: &DMatrix<f64>) -> Self {
        Self::from_mat_blocks(&averaged_blocks::<MAT_DIM>(matrix))
    }

    /// Reconstruct a CMTM from its lower block-Toeplitz adjoint matrix.
    pub fn set_mat_adj(matrix: &DMatrix<f64>) -> Self {
        Self::from_mat_adj_blocks(&averaged_blocks::<ADJ_DIM>(matrix))
    }

    /// Invert the CMTM while preserving the output order.
    pub fn inverse(&self) -> Self {
        Self::set_mat(&self.mat_inv(None))
    }

    /// Compose two CMTMs of the same order.
    pub fn compose(&self, other: &Self) -> Self {
        assert_eq!(
            self.order(),
            other.order(),
            "CMTM composition requires matching orders"
        );
        Self::set_mat(&(self.mat(None) * other.mat(None)))
    }

    /// Block-Toeplitz hat operator in the element-space representation.
    pub fn hat(vecs: &[[f64; ADJ_DIM]]) -> DMatrix<f64> {
        hat_toeplitz(vecs, G::hat)
    }

    /// Block-Toeplitz hat operator in the adjoint representation.
    pub fn hat_adj(vecs: &[[f64; ADJ_DIM]]) -> DMatrix<f64> {
        hat_toeplitz(vecs, G::hat_adj)
    }

    /// Block-Toeplitz hat-commute operator in the adjoint representation.
    pub fn hat_commute_adj(vecs: &[[f64; ADJ_DIM]]) -> DMatrix<f64> {
        hat_toeplitz(vecs, G::hat_commute_adj)
    }

    /// Inverse of [`GenericCmtm::hat`] on lower block-Toeplitz matrices.
    pub fn vee(matrix: &DMatrix<f64>) -> Vec<[f64; ADJ_DIM]> {
        vee_from_matrix(matrix, G::vee)
    }

    /// Inverse of [`GenericCmtm::hat_adj`] on lower block-Toeplitz matrices.
    pub fn vee_adj(matrix: &DMatrix<f64>) -> Vec<[f64; ADJ_DIM]> {
        vee_from_matrix(matrix, G::vee_adj)
    }

    /// Tangent propagation matrix used in higher-order variation formulas.
    pub fn tangent_mat(&self, output_order: Option<usize>) -> DMatrix<f64> {
        let order = self.check_output_order(output_order);
        let table = self.tangent_table(order);
        let mut scales = vec![1.0_f64; order];
        for index in 2..order {
            scales[index] = 1.0 / Self::factorial(index - 1);
        }
        lower_triangular(&table, Some(&scales))
    }

    /// Inverse of [`GenericCmtm::tangent_mat`].
    pub fn tangent_mat_inv(&self, output_order: Option<usize>) -> DMatrix<f64> {
        invert_matrix(&self.tangent_mat(output_order))
    }

    /// CM-scaled tangent propagation matrix.
    pub fn tangent_mat_cm(&self, output_order: Option<usize>) -> DMatrix<f64> {
        let order = self.check_output_order(output_order);
        lower_triangular(&self.tangent_cm_table(order), None)
    }

    /// Inverse of [`GenericCmtm::tangent_mat_cm`].
    pub fn tangent_mat_cm_inv(&self, output_order: Option<usize>) -> DMatrix<f64> {
        invert_matrix(&self.tangent_mat_cm(output_order))
    }

    fn check_output_order(&self, output_order: Option<usize>) -> usize {
        match output_order {
            Some(0) => panic!("Output order must be positive"),
            Some(order) if order > self.order() => {
                panic!("Output order exceeds available derivatives")
            }
            Some(order) => order,
            None => self.order(),
        }
    }

    fn factorial(index: usize) -> f64 {
        (1..=index).fold(1.0, |acc, value| acc * value as f64)
    }

    fn scaled_derivative(&self, index: usize) -> [f64; ADJ_DIM] {
        let scale = Self::factorial(index);
        std::array::from_fn(|component| self.derivatives[index][component] / scale)
    }

    fn hat_series<const BLOCK_DIM: usize, F>(
        &self,
        output_order: usize,
        hat_fn: F,
    ) -> Vec<SMatrix<f64, BLOCK_DIM, BLOCK_DIM>>
    where
        F: Fn([f64; ADJ_DIM]) -> [[f64; BLOCK_DIM]; BLOCK_DIM],
    {
        let mut hats = Vec::with_capacity(output_order.saturating_sub(1));
        for index in 0..output_order.saturating_sub(1) {
            hats.push(smatrix_from_array(hat_fn(self.scaled_derivative(index))));
        }
        hats
    }

    fn mat_blocks(&self, output_order: usize) -> Vec<SMatrix<f64, MAT_DIM, MAT_DIM>> {
        let hats = self.hat_series(output_order, G::hat);
        let mut blocks = vec![SMatrix::<f64, MAT_DIM, MAT_DIM>::zeros(); output_order];
        blocks[0] = self.element.as_matrix();

        for order in 1..output_order {
            let mut acc = SMatrix::<f64, MAT_DIM, MAT_DIM>::zeros();
            for index in 0..order {
                acc += blocks[order - index - 1] * hats[index];
            }
            blocks[order] = acc / order as f64;
        }

        blocks
    }

    fn mat_adj_blocks(&self, output_order: usize) -> Vec<SMatrix<f64, ADJ_DIM, ADJ_DIM>> {
        let hats = self.hat_series(output_order, G::hat_adj);
        let mut blocks = vec![SMatrix::<f64, ADJ_DIM, ADJ_DIM>::zeros(); output_order];
        blocks[0] = self.element.adjoint_matrix();

        for order in 1..output_order {
            let mut acc = SMatrix::<f64, ADJ_DIM, ADJ_DIM>::zeros();
            for index in 0..order {
                acc += blocks[order - index - 1] * hats[index];
            }
            blocks[order] = acc / order as f64;
        }

        blocks
    }

    fn mat_inv_blocks(&self, output_order: usize) -> Vec<SMatrix<f64, MAT_DIM, MAT_DIM>> {
        let hats = self.hat_series(output_order, G::hat);
        let mut blocks = vec![SMatrix::<f64, MAT_DIM, MAT_DIM>::zeros(); output_order];
        blocks[0] = self.element.inverse().as_matrix();

        for order in 1..output_order {
            let mut acc = SMatrix::<f64, MAT_DIM, MAT_DIM>::zeros();
            for index in 0..order {
                acc -= hats[index] * blocks[order - index - 1];
            }
            blocks[order] = acc / order as f64;
        }

        blocks
    }

    fn mat_inv_adj_blocks(&self, output_order: usize) -> Vec<SMatrix<f64, ADJ_DIM, ADJ_DIM>> {
        let hats = self.hat_series(output_order, G::hat_adj);
        let mut blocks = vec![SMatrix::<f64, ADJ_DIM, ADJ_DIM>::zeros(); output_order];
        blocks[0] = self.element.inverse().adjoint_matrix();

        for order in 1..output_order {
            let mut acc = SMatrix::<f64, ADJ_DIM, ADJ_DIM>::zeros();
            for index in 0..order {
                acc -= hats[index] * blocks[order - index - 1];
            }
            blocks[order] = acc / order as f64;
        }

        blocks
    }

    fn from_mat_blocks(blocks: &[SMatrix<f64, MAT_DIM, MAT_DIM>]) -> Self {
        assert!(!blocks.is_empty(), "At least one block is required");

        let element = G::from_matrix(matrix_to_array(&blocks[0]));
        let inverse = element.inverse().as_matrix();
        let mut derivatives = Vec::with_capacity(blocks.len().saturating_sub(1));
        let mut hats =
            Vec::<SMatrix<f64, MAT_DIM, MAT_DIM>>::with_capacity(blocks.len().saturating_sub(1));

        for index in 0..blocks.len().saturating_sub(1) {
            let mut correction = SMatrix::<f64, MAT_DIM, MAT_DIM>::zeros();
            for inner in 0..index {
                correction += blocks[index - inner] * hats[inner];
            }

            let delta = inverse * (blocks[index + 1] * (index + 1) as f64 - correction);
            let vector = G::vee(matrix_to_array(&delta));
            let scale = Self::factorial(index);
            let derivative = SVector::<f64, ADJ_DIM>::from_fn(|row, _| vector[row] * scale);
            hats.push(smatrix_from_array(G::hat(std::array::from_fn(|row| {
                derivative[row] / scale
            }))));
            derivatives.push(derivative);
        }

        Self {
            element,
            derivatives,
        }
    }

    fn from_mat_adj_blocks(blocks: &[SMatrix<f64, ADJ_DIM, ADJ_DIM>]) -> Self {
        assert!(!blocks.is_empty(), "At least one block is required");

        let element = G::from_adjoint_matrix(matrix_to_array(&blocks[0]));
        let inverse = element.inverse().adjoint_matrix();
        let mut derivatives = Vec::with_capacity(blocks.len().saturating_sub(1));
        let mut hats =
            Vec::<SMatrix<f64, ADJ_DIM, ADJ_DIM>>::with_capacity(blocks.len().saturating_sub(1));

        for index in 0..blocks.len().saturating_sub(1) {
            let mut correction = SMatrix::<f64, ADJ_DIM, ADJ_DIM>::zeros();
            for inner in 0..index {
                correction += blocks[index - inner] * hats[inner];
            }

            let delta = inverse * (blocks[index + 1] * (index + 1) as f64 - correction);
            let vector = G::vee_adj(matrix_to_array(&delta));
            let scale = Self::factorial(index);
            let derivative = SVector::<f64, ADJ_DIM>::from_fn(|row, _| vector[row] * scale);
            hats.push(smatrix_from_array(G::hat_adj(std::array::from_fn(|row| {
                derivative[row] / scale
            }))));
            derivatives.push(derivative);
        }

        Self {
            element,
            derivatives,
        }
    }

    fn tangent_table(&self, output_order: usize) -> Vec<Vec<SMatrix<f64, ADJ_DIM, ADJ_DIM>>> {
        let hats = self.hat_series(output_order, G::hat_adj);
        let eye = SMatrix::<f64, ADJ_DIM, ADJ_DIM>::identity();
        let mut table =
            vec![vec![SMatrix::<f64, ADJ_DIM, ADJ_DIM>::zeros(); output_order]; output_order];
        table[0][0] = eye;

        for row in 1..output_order {
            table[row][row] = eye / row as f64;
            for col in 0..row {
                let mut acc = SMatrix::<f64, ADJ_DIM, ADJ_DIM>::zeros();
                for inner in 0..(row - col) {
                    acc -= hats[inner] * table[row - inner - 1][col];
                }
                table[row][col] = acc / row as f64;
            }
        }

        table
    }

    fn tangent_cm_table(&self, output_order: usize) -> Vec<Vec<SMatrix<f64, ADJ_DIM, ADJ_DIM>>> {
        let hats = self.hat_series(output_order, G::hat_adj);
        let eye = SMatrix::<f64, ADJ_DIM, ADJ_DIM>::identity();
        let mut table =
            vec![vec![SMatrix::<f64, ADJ_DIM, ADJ_DIM>::zeros(); output_order]; output_order];
        table[0][0] = eye;

        for row in 1..output_order {
            table[row][row] = eye / row as f64;
            for col in 0..row {
                let mut acc = SMatrix::<f64, ADJ_DIM, ADJ_DIM>::zeros();
                for inner in col..row {
                    acc -= hats[row - 1 - inner] * table[inner][col];
                }
                table[row][col] = acc / row as f64;
            }
        }

        table
    }
}

impl RotationalCmtm {
    /// Build a rotational CMTM from an SO(3) element.
    pub fn from_so3(rotation: &So3) -> Self {
        Self {
            element: rotation.clone(),
            derivatives: Vec::new(),
        }
    }

    /// Build a rotational CMTM with higher-order angular derivatives.
    pub fn from_so3_with_derivatives(rotation: &So3, derivatives: Vec<[f64; 3]>) -> Self {
        Self::with_derivatives(rotation.clone(), derivatives)
    }

    /// Apply the base adjoint matrix to an angular velocity vector.
    pub fn apply_omega(&self, omega: [f64; 3]) -> [f64; 3] {
        apply_linear(&self.element.adjoint_matrix(), omega)
    }

    /// Backwards-compatible alias for [`RotationalCmtm::mat_adj`].
    pub fn to_block_matrix(&self, output_order: Option<usize>) -> DMatrix<f64> {
        self.mat_adj(output_order)
    }
}

impl SpatialCmtm {
    /// Build a spatial CMTM from an SE(3) element.
    pub fn from_se3(transform: &Se3) -> Self {
        Self {
            element: transform.clone(),
            derivatives: Vec::new(),
        }
    }

    /// Build a spatial CMTM with higher-order spatial derivatives.
    pub fn from_se3_with_derivatives(transform: &Se3, derivatives: Vec<[f64; 6]>) -> Self {
        Self::with_derivatives(transform.clone(), derivatives)
    }

    /// Apply the base adjoint matrix to a spatial twist.
    pub fn apply_twist(&self, twist: [f64; 6]) -> [f64; 6] {
        apply_linear(&self.element.adjoint_matrix(), twist)
    }

    /// Backwards-compatible alias for [`SpatialCmtm::mat_adj`].
    pub fn to_block_matrix(&self, output_order: Option<usize>) -> DMatrix<f64> {
        self.mat_adj(output_order)
    }
}

impl CmtmElement<3, 3> for So3 {
    fn from_matrix(matrix: [[f64; 3]; 3]) -> Self {
        So3::from_matrix(matrix)
    }

    fn from_adjoint_matrix(matrix: [[f64; 3]; 3]) -> Self {
        So3::from_matrix(matrix)
    }

    fn hat(vector: [f64; 3]) -> [[f64; 3]; 3] {
        So3::hat(vector)
    }

    fn vee(matrix: [[f64; 3]; 3]) -> [f64; 3] {
        So3::vee(matrix)
    }

    fn hat_adj(vector: [f64; 3]) -> [[f64; 3]; 3] {
        So3::hat_adj(vector)
    }

    fn hat_commute_adj(vector: [f64; 3]) -> [[f64; 3]; 3] {
        So3::hat_commute_adj(vector)
    }

    fn vee_adj(matrix: [[f64; 3]; 3]) -> [f64; 3] {
        So3::vee_adj(matrix)
    }
}

impl CmtmElement<4, 6> for Se3 {
    fn from_matrix(matrix: [[f64; 4]; 4]) -> Self {
        Se3::from_matrix(matrix)
    }

    fn from_adjoint_matrix(matrix: [[f64; 6]; 6]) -> Self {
        Se3::from_adjoint_matrix(matrix)
    }

    fn hat(vector: [f64; 6]) -> [[f64; 4]; 4] {
        Se3::hat(vector)
    }

    fn vee(matrix: [[f64; 4]; 4]) -> [f64; 6] {
        Se3::vee(matrix)
    }

    fn hat_adj(vector: [f64; 6]) -> [[f64; 6]; 6] {
        Se3::hat_adj(vector)
    }

    fn hat_commute_adj(vector: [f64; 6]) -> [[f64; 6]; 6] {
        Se3::hat_commute_adj(vector)
    }

    fn vee_adj(matrix: [[f64; 6]; 6]) -> [f64; 6] {
        Se3::vee_adj(matrix)
    }
}

impl<G, const MAT_DIM: usize, const ADJ_DIM: usize> Mul for GenericCmtm<G, MAT_DIM, ADJ_DIM>
where
    G: CmtmElement<MAT_DIM, ADJ_DIM>,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.compose(&rhs)
    }
}

impl<'a, G, const MAT_DIM: usize, const ADJ_DIM: usize> Mul<&'a GenericCmtm<G, MAT_DIM, ADJ_DIM>>
    for GenericCmtm<G, MAT_DIM, ADJ_DIM>
where
    G: CmtmElement<MAT_DIM, ADJ_DIM>,
{
    type Output = GenericCmtm<G, MAT_DIM, ADJ_DIM>;

    fn mul(self, rhs: &'a GenericCmtm<G, MAT_DIM, ADJ_DIM>) -> Self::Output {
        self.compose(rhs)
    }
}

impl<'a, G, const MAT_DIM: usize, const ADJ_DIM: usize> Mul<GenericCmtm<G, MAT_DIM, ADJ_DIM>>
    for &'a GenericCmtm<G, MAT_DIM, ADJ_DIM>
where
    G: CmtmElement<MAT_DIM, ADJ_DIM>,
{
    type Output = GenericCmtm<G, MAT_DIM, ADJ_DIM>;

    fn mul(self, rhs: GenericCmtm<G, MAT_DIM, ADJ_DIM>) -> Self::Output {
        self.compose(&rhs)
    }
}

impl<'a, 'b, G, const MAT_DIM: usize, const ADJ_DIM: usize>
    Mul<&'a GenericCmtm<G, MAT_DIM, ADJ_DIM>> for &'b GenericCmtm<G, MAT_DIM, ADJ_DIM>
where
    G: CmtmElement<MAT_DIM, ADJ_DIM>,
{
    type Output = GenericCmtm<G, MAT_DIM, ADJ_DIM>;

    fn mul(self, rhs: &'a GenericCmtm<G, MAT_DIM, ADJ_DIM>) -> Self::Output {
        self.compose(rhs)
    }
}
