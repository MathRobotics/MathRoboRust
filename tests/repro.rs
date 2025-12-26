use std::f64::consts::FRAC_PI_2;

use mathroborust::util::{skew_symmetric, vector3_from_array};
use mathroborust::{RustCmtm, RustSe3, RustSo3};
use nalgebra::{SMatrix, SVector};

fn approx_eq(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < tol, "expected {y}, got {x}");
    }
}

fn approx_eq_matrix(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3], tol: f64) {
    for r in 0..3 {
        for c in 0..3 {
            let x = a[r][c];
            let y = b[r][c];
            assert!((x - y).abs() < tol, "expected {y}, got {x} at ({r},{c})");
        }
    }
}

fn approx_eq_matrix4(a: &[[f64; 4]; 4], b: &[[f64; 4]; 4], tol: f64) {
    for r in 0..4 {
        for c in 0..4 {
            let x = a[r][c];
            let y = b[r][c];
            assert!((x - y).abs() < tol, "expected {y}, got {x} at ({r},{c})");
        }
    }
}

#[test]
fn so3_rotation_matches_expected() {
    let rotation = RustSo3::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
    let rotated = rotation.apply([1.0, 0.0, 0.0]);
    approx_eq(&rotated, &[0.0, 1.0, 0.0], 1e-12);
}

#[test]
fn se3_translation_and_rotation_combined() {
    let rotation = RustSo3::from_axis_angle([0.0, 1.0, 0.0], FRAC_PI_2);
    let transform = RustSe3::from_parts(rotation, [1.0, 2.0, 3.0]);
    let applied = transform.apply([1.0, 0.0, 0.0]);
    approx_eq(&applied, &[1.0, 2.0, 2.0], 1e-12);
}

#[test]
fn cmtm_adjoint_matches_reference() {
    let rotation = RustSo3::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
    let transform = RustSe3::from_parts(rotation, [0.5, 0.25, -0.75]);
    let adjoint = RustCmtm::from_se3(&transform);

    let twist = [0.1, 0.2, 0.3, 1.0, 2.0, 3.0];
    let transformed = adjoint.apply_twist(twist);

    let rotation_matrix = transform.rotation().rotation().matrix();
    let translation = transform.translation();
    let translation_vec = vector3_from_array(translation);
    let skew = skew_symmetric(&translation_vec);

    let mut expected_matrix = SMatrix::<f64, 6, 6>::zeros();
    for r in 0..3 {
        for c in 0..3 {
            expected_matrix[(r, c)] = rotation_matrix[(r, c)];
            expected_matrix[(r + 3, c + 3)] = rotation_matrix[(r, c)];
            expected_matrix[(r + 3, c)] = (skew * rotation_matrix)[(r, c)];
        }
    }

    let expected: [f64; 6] = (expected_matrix * SVector::<f64, 6>::from_row_slice(&twist)).into();

    // Ensure the group-level adjoint uses the same block structure leveraged by CMTM
    let group_adjoint = transform.adjoint();
    for r in 0..6 {
        for c in 0..6 {
            assert!((group_adjoint[(r, c)] - expected_matrix[(r, c)]).abs() < 1e-12);
        }
    }

    approx_eq(&transformed, &expected, 1e-10);
}

#[test]
fn so3_quaternion_roundtrip_matches_matrix() {
    let rotation = RustSo3::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
    let quaternion = rotation.to_quaternion();
    let rebuilt = RustSo3::from_quaternion(quaternion);

    let original_matrix = rotation.to_matrix();
    let rebuilt_matrix = rebuilt.to_matrix();

    approx_eq_matrix(&original_matrix, &rebuilt_matrix, 1e-12);
}

#[test]
fn so3_rotation_vector_log_exp_roundtrip() {
    let vector = [0.2, -0.1, 0.3];
    let rotation = RustSo3::from_rotation_vector(vector);
    let recovered = rotation.to_rotation_vector();
    approx_eq(&recovered, &vector, 1e-12);
}

#[test]
fn hat_and_vee_are_inverses() {
    let vector = [0.25, -0.5, 1.25];
    let hat = RustSo3::hat(vector);
    let recovered = RustSo3::vee(hat);
    approx_eq(&recovered, &vector, 1e-12);
}

#[test]
fn se3_hat_and_vee_are_inverses() {
    let twist = [0.1, -0.2, 0.3, 1.0, -2.0, 3.0];
    let hat = RustSe3::hat(twist);
    let recovered = RustSe3::vee(hat);
    approx_eq(&recovered, &twist, 1e-12);
}

#[test]
fn se3_exp_with_pure_translation_matches_expected() {
    let twist = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0];
    let exp = RustSe3::exp(twist, None);
    let mut expected = [[0.0_f64; 4]; 4];
    expected[0][0] = 1.0;
    expected[1][1] = 1.0;
    expected[2][2] = 1.0;
    expected[3][3] = 1.0;
    expected[0][3] = 1.0;
    expected[1][3] = 2.0;
    expected[2][3] = 3.0;

    approx_eq_matrix4(&exp, &expected, 1e-12);
}

#[test]
fn se3_from_matrix_round_trip() {
    let rotation = RustSo3::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
    let transform = RustSe3::from_parts(rotation.clone(), [0.25, -0.5, 0.75]);
    let matrix = transform.to_matrix();
    let rebuilt = RustSe3::from_matrix(matrix);

    approx_eq_matrix4(&matrix, &rebuilt.to_matrix(), 1e-12);
    approx_eq_matrix(&rotation.to_matrix(), &rebuilt.rotation().to_matrix(), 1e-12);
    approx_eq(&transform.translation(), &rebuilt.translation(), 1e-12);
}
