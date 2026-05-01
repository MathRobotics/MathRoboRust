use std::path::PathBuf;
use std::process::Command;
use std::sync::OnceLock;

use mathroborust::{RustSe3, RustSo3};
use serde::Deserialize;

#[derive(Deserialize)]
struct ParityPayload {
    so3: So3Parity,
    se3: Se3Parity,
    cmtm_so3: CmtmSo3Parity,
    cmtm_se3: CmtmSe3Parity,
}

#[derive(Deserialize)]
struct So3Parity {
    vector: [f64; 3],
    scale: f64,
    hat: [[f64; 3]; 3],
    vee: [f64; 3],
    exp: [[f64; 3]; 3],
    compose_left: [f64; 3],
    compose_right: [f64; 3],
    compose: [[f64; 3]; 3],
    quaternion: [f64; 4],
    quaternion_matrix: [[f64; 3]; 3],
}

#[derive(Deserialize)]
struct Se3Parity {
    twist: [f64; 6],
    scale: f64,
    hat: [[f64; 4]; 4],
    vee: [f64; 6],
    exp: [[f64; 4]; 4],
    left_rotvec: [f64; 3],
    left_translation: [f64; 3],
    right_rotvec: [f64; 3],
    right_translation: [f64; 3],
    compose: [[f64; 4]; 4],
    inverse: [[f64; 4]; 4],
    adjoint: [[f64; 6]; 6],
    point: [f64; 3],
    apply: [f64; 3],
}

#[derive(Deserialize)]
struct CmtmSo3Parity {
    rotation_vector: [f64; 3],
    derivatives: Vec<[f64; 3]>,
    mat: Vec<Vec<f64>>,
    mat_adj: Vec<Vec<f64>>,
    mat_inv: Vec<Vec<f64>>,
    mat_inv_adj: Vec<Vec<f64>>,
    set_mat_vecs: Vec<[f64; 3]>,
    inverse_mat: Vec<Vec<f64>>,
    compose_right_rotation_vector: [f64; 3],
    compose_right_derivatives: Vec<[f64; 3]>,
    compose_mat: Vec<Vec<f64>>,
    hat: Vec<Vec<f64>>,
    hat_adj: Vec<Vec<f64>>,
    hat_commute_adj: Vec<Vec<f64>>,
    vee: Vec<[f64; 3]>,
    vee_adj: Vec<[f64; 3]>,
    tangent_mat: Vec<Vec<f64>>,
    tangent_mat_cm: Vec<Vec<f64>>,
}

#[derive(Deserialize)]
struct CmtmSe3Parity {
    rotation_vector: [f64; 3],
    translation: [f64; 3],
    derivatives: Vec<[f64; 6]>,
    mat: Vec<Vec<f64>>,
    mat_adj: Vec<Vec<f64>>,
    mat_inv: Vec<Vec<f64>>,
    mat_inv_adj: Vec<Vec<f64>>,
    set_mat_vecs: Vec<[f64; 6]>,
    inverse_mat: Vec<Vec<f64>>,
    compose_right_rotation_vector: [f64; 3],
    compose_right_translation: [f64; 3],
    compose_right_derivatives: Vec<[f64; 6]>,
    compose_mat: Vec<Vec<f64>>,
    hat: Vec<Vec<f64>>,
    hat_adj: Vec<Vec<f64>>,
    hat_commute_adj: Vec<Vec<f64>>,
    vee: Vec<[f64; 6]>,
    vee_adj: Vec<[f64; 6]>,
    tangent_mat: Vec<Vec<f64>>,
    tangent_mat_cm: Vec<Vec<f64>>,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn python_command() -> String {
    std::env::var("PYTHON").unwrap_or_else(|_| "python3".to_string())
}

fn load_parity_payload() -> ParityPayload {
    let script = repo_root().join("tests/python/mathrobo_parity.py");
    let output = Command::new(python_command())
        .arg(script)
        .current_dir(repo_root())
        .output()
        .expect("failed to run MathRobo parity script");

    assert!(
        output.status.success(),
        "MathRobo parity script failed.\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    serde_json::from_slice(&output.stdout).expect("failed to parse MathRobo parity payload")
}

fn parity_payload() -> &'static ParityPayload {
    static PAYLOAD: OnceLock<ParityPayload> = OnceLock::new();
    PAYLOAD.get_or_init(load_parity_payload)
}

fn approx_eq_vec(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "mismatch at index {idx}: expected {e}, got {a}"
        );
    }
}

fn approx_eq_matrix<const R: usize, const C: usize>(
    actual: &[[f64; C]; R],
    expected: &[[f64; C]; R],
    tol: f64,
) {
    for row in 0..R {
        for col in 0..C {
            let a = actual[row][col];
            let e = expected[row][col];
            assert!(
                (a - e).abs() <= tol,
                "mismatch at ({row},{col}): expected {e}, got {a}"
            );
        }
    }
}

fn approx_eq_matrix_dynamic(actual: &[Vec<f64>], expected: &[Vec<f64>], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (row_idx, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(actual_row.len(), expected_row.len());
        for (col_idx, (a, e)) in actual_row.iter().zip(expected_row.iter()).enumerate() {
            assert!(
                (a - e).abs() <= tol,
                "mismatch at ({row_idx},{col_idx}): expected {e}, got {a}"
            );
        }
    }
}

fn approx_eq_vector_list<const DIM: usize>(
    actual: &[[f64; DIM]],
    expected: &[[f64; DIM]],
    tol: f64,
) {
    assert_eq!(actual.len(), expected.len());
    for (idx, (actual_vec, expected_vec)) in actual.iter().zip(expected.iter()).enumerate() {
        for component in 0..DIM {
            let a = actual_vec[component];
            let e = expected_vec[component];
            assert!(
                (a - e).abs() <= tol,
                "mismatch at vector {idx}, component {component}: expected {e}, got {a}"
            );
        }
    }
}

fn matrix6_to_array(matrix: nalgebra::SMatrix<f64, 6, 6>) -> [[f64; 6]; 6] {
    let mut out = [[0.0; 6]; 6];
    for row in 0..6 {
        for col in 0..6 {
            out[row][col] = matrix[(row, col)];
        }
    }
    out
}

fn dmatrix_to_vec(matrix: nalgebra::DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| (0..matrix.ncols()).map(|col| matrix[(row, col)]).collect())
        .collect()
}

#[test]
fn so3_matches_mathrobo_reference() {
    let so3 = &parity_payload().so3;

    approx_eq_matrix(&RustSo3::hat(so3.vector), &so3.hat, 1e-12);
    approx_eq_vec(&RustSo3::vee(so3.hat), &so3.vee, 1e-12);
    approx_eq_matrix(&RustSo3::exp(so3.vector, Some(so3.scale)), &so3.exp, 1e-12);

    let left = RustSo3::from_rotation_vector(so3.compose_left);
    let right = RustSo3::from_rotation_vector(so3.compose_right);
    approx_eq_matrix(&left.compose(&right).to_matrix(), &so3.compose, 1e-12);

    let from_quaternion = RustSo3::from_quaternion(so3.quaternion);
    approx_eq_matrix(&from_quaternion.to_matrix(), &so3.quaternion_matrix, 1e-11);
}

#[test]
fn se3_matches_mathrobo_reference() {
    let se3 = &parity_payload().se3;

    approx_eq_matrix(&RustSe3::hat(se3.twist), &se3.hat, 1e-12);
    approx_eq_vec(&RustSe3::vee(se3.hat), &se3.vee, 1e-12);
    approx_eq_matrix(&RustSe3::exp(se3.twist, Some(se3.scale)), &se3.exp, 1e-12);

    let left = RustSe3::from_parts(
        RustSo3::from_rotation_vector(se3.left_rotvec),
        se3.left_translation,
    );
    let right = RustSe3::from_parts(
        RustSo3::from_rotation_vector(se3.right_rotvec),
        se3.right_translation,
    );

    approx_eq_matrix(&left.compose(&right).to_matrix(), &se3.compose, 1e-12);
    approx_eq_matrix(&left.inverse().to_matrix(), &se3.inverse, 1e-12);
    approx_eq_matrix(&matrix6_to_array(left.adjoint()), &se3.adjoint, 1e-12);
    approx_eq_vec(&left.apply(se3.point), &se3.apply, 1e-12);
}

#[test]
fn cmtm_so3_matches_mathrobo_reference() {
    let cmtm = &parity_payload().cmtm_so3;

    let left = mathroborust::RotationalCmtm::from_so3_with_derivatives(
        &RustSo3::from_rotation_vector(cmtm.rotation_vector),
        cmtm.derivatives.clone(),
    );

    approx_eq_matrix_dynamic(&dmatrix_to_vec(left.mat(None)), &cmtm.mat, 1e-10);
    approx_eq_matrix_dynamic(&dmatrix_to_vec(left.mat_adj(None)), &cmtm.mat_adj, 1e-10);
    approx_eq_matrix_dynamic(&dmatrix_to_vec(left.mat_inv(None)), &cmtm.mat_inv, 1e-10);
    approx_eq_matrix_dynamic(
        &dmatrix_to_vec(left.mat_inv_adj(None)),
        &cmtm.mat_inv_adj,
        1e-10,
    );

    let set_from_mat = mathroborust::RotationalCmtm::set_mat(&left.mat(None));
    approx_eq_vector_list(&set_from_mat.vecs(None), &cmtm.set_mat_vecs, 1e-10);

    approx_eq_matrix_dynamic(
        &dmatrix_to_vec(left.inverse().mat(None)),
        &cmtm.inverse_mat,
        1e-10,
    );

    let right = mathroborust::RotationalCmtm::from_so3_with_derivatives(
        &RustSo3::from_rotation_vector(cmtm.compose_right_rotation_vector),
        cmtm.compose_right_derivatives.clone(),
    );
    approx_eq_matrix_dynamic(
        &dmatrix_to_vec(left.compose(&right).mat(None)),
        &cmtm.compose_mat,
        1e-10,
    );

    approx_eq_matrix_dynamic(
        &dmatrix_to_vec(mathroborust::RotationalCmtm::hat(&cmtm.derivatives)),
        &cmtm.hat,
        1e-10,
    );
    approx_eq_matrix_dynamic(
        &dmatrix_to_vec(mathroborust::RotationalCmtm::hat_adj(&cmtm.derivatives)),
        &cmtm.hat_adj,
        1e-10,
    );
    approx_eq_matrix_dynamic(
        &dmatrix_to_vec(mathroborust::RotationalCmtm::hat_commute_adj(
            &cmtm.derivatives,
        )),
        &cmtm.hat_commute_adj,
        1e-10,
    );
    approx_eq_vector_list(
        &mathroborust::RotationalCmtm::vee(&mathroborust::RotationalCmtm::hat(&cmtm.derivatives)),
        &cmtm.vee,
        1e-10,
    );
    approx_eq_vector_list(
        &mathroborust::RotationalCmtm::vee_adj(&mathroborust::RotationalCmtm::hat_adj(
            &cmtm.derivatives,
        )),
        &cmtm.vee_adj,
        1e-10,
    );
    approx_eq_matrix_dynamic(
        &dmatrix_to_vec(left.tangent_mat(None)),
        &cmtm.tangent_mat,
        1e-10,
    );
    approx_eq_matrix_dynamic(
        &dmatrix_to_vec(left.tangent_mat_cm(None)),
        &cmtm.tangent_mat_cm,
        1e-10,
    );
}

#[test]
fn cmtm_se3_matches_mathrobo_reference() {
    let cmtm = &parity_payload().cmtm_se3;

    let left = mathroborust::RustCmtm::from_se3_with_derivatives(
        &RustSe3::from_parts(
            RustSo3::from_rotation_vector(cmtm.rotation_vector),
            cmtm.translation,
        ),
        cmtm.derivatives.clone(),
    );

    approx_eq_matrix_dynamic(&dmatrix_to_vec(left.mat(None)), &cmtm.mat, 1e-10);
    approx_eq_matrix_dynamic(&dmatrix_to_vec(left.mat_adj(None)), &cmtm.mat_adj, 1e-10);
    approx_eq_matrix_dynamic(&dmatrix_to_vec(left.mat_inv(None)), &cmtm.mat_inv, 1e-10);
    approx_eq_matrix_dynamic(
        &dmatrix_to_vec(left.mat_inv_adj(None)),
        &cmtm.mat_inv_adj,
        1e-10,
    );

    let set_from_mat = mathroborust::RustCmtm::set_mat(&left.mat(None));
    approx_eq_vector_list(&set_from_mat.vecs(None), &cmtm.set_mat_vecs, 1e-10);

    approx_eq_matrix_dynamic(
        &dmatrix_to_vec(left.inverse().mat(None)),
        &cmtm.inverse_mat,
        1e-10,
    );

    let right = mathroborust::RustCmtm::from_se3_with_derivatives(
        &RustSe3::from_parts(
            RustSo3::from_rotation_vector(cmtm.compose_right_rotation_vector),
            cmtm.compose_right_translation,
        ),
        cmtm.compose_right_derivatives.clone(),
    );
    approx_eq_matrix_dynamic(
        &dmatrix_to_vec(left.compose(&right).mat(None)),
        &cmtm.compose_mat,
        1e-10,
    );

    approx_eq_matrix_dynamic(
        &dmatrix_to_vec(mathroborust::RustCmtm::hat(&cmtm.derivatives)),
        &cmtm.hat,
        1e-10,
    );
    approx_eq_matrix_dynamic(
        &dmatrix_to_vec(mathroborust::RustCmtm::hat_adj(&cmtm.derivatives)),
        &cmtm.hat_adj,
        1e-10,
    );
    approx_eq_matrix_dynamic(
        &dmatrix_to_vec(mathroborust::RustCmtm::hat_commute_adj(&cmtm.derivatives)),
        &cmtm.hat_commute_adj,
        1e-10,
    );
    approx_eq_vector_list(
        &mathroborust::RustCmtm::vee(&mathroborust::RustCmtm::hat(&cmtm.derivatives)),
        &cmtm.vee,
        1e-10,
    );
    approx_eq_vector_list(
        &mathroborust::RustCmtm::vee_adj(&mathroborust::RustCmtm::hat_adj(&cmtm.derivatives)),
        &cmtm.vee_adj,
        1e-10,
    );
    approx_eq_matrix_dynamic(
        &dmatrix_to_vec(left.tangent_mat(None)),
        &cmtm.tangent_mat,
        1e-10,
    );
    approx_eq_matrix_dynamic(
        &dmatrix_to_vec(left.tangent_mat_cm(None)),
        &cmtm.tangent_mat_cm,
        1e-10,
    );
}
