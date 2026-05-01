use nalgebra::{DMatrix, Matrix3, Matrix4, SMatrix};

pub fn matrix3_from_array(matrix: [[f64; 3]; 3]) -> Matrix3<f64> {
    Matrix3::from_row_slice(&[
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[2][0],
        matrix[2][1],
        matrix[2][2],
    ])
}

pub fn matrix4_from_array(matrix: [[f64; 4]; 4]) -> Matrix4<f64> {
    Matrix4::from_row_slice(&[
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
    ])
}

pub fn matrix6_from_array(matrix: [[f64; 6]; 6]) -> SMatrix<f64, 6, 6> {
    SMatrix::<f64, 6, 6>::from_row_slice(&[
        matrix[0][0],
        matrix[0][1],
        matrix[0][2],
        matrix[0][3],
        matrix[0][4],
        matrix[0][5],
        matrix[1][0],
        matrix[1][1],
        matrix[1][2],
        matrix[1][3],
        matrix[1][4],
        matrix[1][5],
        matrix[2][0],
        matrix[2][1],
        matrix[2][2],
        matrix[2][3],
        matrix[2][4],
        matrix[2][5],
        matrix[3][0],
        matrix[3][1],
        matrix[3][2],
        matrix[3][3],
        matrix[3][4],
        matrix[3][5],
        matrix[4][0],
        matrix[4][1],
        matrix[4][2],
        matrix[4][3],
        matrix[4][4],
        matrix[4][5],
        matrix[5][0],
        matrix[5][1],
        matrix[5][2],
        matrix[5][3],
        matrix[5][4],
        matrix[5][5],
    ])
}

pub fn dmatrix_to_vec(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| (0..matrix.ncols()).map(|col| matrix[(row, col)]).collect())
        .collect()
}

pub fn dmatrix_from_vec(matrix: &[Vec<f64>]) -> DMatrix<f64> {
    let rows = matrix.len();
    let cols = matrix.first().map(|row| row.len()).unwrap_or(0);
    DMatrix::<f64>::from_fn(rows, cols, |row, col| matrix[row][col])
}
