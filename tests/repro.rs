use std::f64::consts::FRAC_PI_2;

use mathroborust::lie::LieGroup;
use mathroborust::util::{skew_symmetric, vector3_from_array};
use mathroborust::{RotationalCmtm, RustCmtm, RustSe3, RustSo3};
use nalgebra::{DMatrix, SMatrix, SVector};

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
fn cmtm_from_so3_matches_rotation_block() {
    let rotation = RustSo3::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
    let adjoint = RotationalCmtm::from_so3(&rotation);

    let omega = [0.25, 0.5, -1.0];
    let rotated = adjoint.apply_omega(omega);
    let expected = rotation.apply(omega);

    approx_eq(&rotated, &expected, 1e-12);
    approx_eq_matrix(&adjoint.to_matrix(), &rotation.to_matrix(), 1e-12);
}

#[test]
fn cmtm_block_matrix_handles_second_order() {
    let rotation = RustSo3::identity();
    let derivatives = vec![[0.1, -0.2, 0.3]]; // enable second-order output
    let adjoint = RotationalCmtm::from_so3_with_derivatives(&rotation, derivatives);

    let block = adjoint.to_block_matrix(None);

    // Expected 6×6 block matrix with the base rotation on both diagonal blocks
    // and the first derivative hat block in the lower-left corner.
    let mut expected = DMatrix::<f64>::zeros(6, 6);
    let base = rotation.rotation().matrix();
    let hat = skew_symmetric(&vector3_from_array([0.1, -0.2, 0.3]));
    for r in 0..3 {
        for c in 0..3 {
            expected[(r, c)] = base[(r, c)];
            expected[(r + 3, c + 3)] = base[(r, c)];
            expected[(r + 3, c)] = hat[(r, c)];
        }
    }

    for r in 0..6 {
        for c in 0..6 {
            assert!((block[(r, c)] - expected[(r, c)]).abs() < 1e-12);
        }
    }
}

#[test]
fn cmtm_block_matrix_handles_third_order() {
    // Identity rotation simplifies the expected block structure while still
    // exercising the recursive derivative accumulation.
    let rotation = RustSo3::identity();
    let w0 = [0.05, -0.1, 0.2];
    let w1 = [-0.02, 0.04, -0.08];
    let adjoint = RotationalCmtm::from_so3_with_derivatives(&rotation, vec![w0, w1]);

    let block = adjoint.to_block_matrix(None);

    let mut expected = DMatrix::<f64>::zeros(9, 9);
    let base = rotation.rotation().matrix();
    let hat0 = skew_symmetric(&vector3_from_array(w0));
    let hat1 = skew_symmetric(&vector3_from_array(w1));
    let mat2 = (hat1 + hat0 * hat0) / 2.0;

    // Block layout for order-3 CMTM:
    // [ I    0    0 ]
    // [ H0   I    0 ]
    // [ M2   H0   I ]
    for r in 0..3 {
        for c in 0..3 {
            expected[(r, c)] = base[(r, c)];
            expected[(r + 3, c + 3)] = base[(r, c)];
            expected[(r + 6, c + 6)] = base[(r, c)];

            expected[(r + 3, c)] = hat0[(r, c)];
            expected[(r + 6, c + 3)] = hat0[(r, c)];
            expected[(r + 6, c)] = mat2[(r, c)];
        }
    }

    for r in 0..9 {
        for c in 0..9 {
            assert!((block[(r, c)] - expected[(r, c)]).abs() < 1e-12);
        }
    }
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
fn so3_mul_matches_compose() {
    let r1 = RustSo3::from_axis_angle([1.0, 0.0, 0.0], FRAC_PI_2);
    let r2 = RustSo3::from_axis_angle([0.0, 1.0, 0.0], FRAC_PI_2);

    let composed = r1.compose(&r2);
    let multiplied = r1 * r2;

    approx_eq_matrix(&composed.to_matrix(), &multiplied.to_matrix(), 1e-12);
    let vector = [1.0, 2.0, 3.0];
    approx_eq(&composed.apply(vector), &multiplied.apply(vector), 1e-12);
}

#[test]
fn se3_mul_matches_compose() {
    let r1 = RustSo3::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
    let t1 = [0.5, -0.25, 0.75];
    let g1 = RustSe3::from_parts(r1, t1);

    let r2 = RustSo3::from_axis_angle([0.0, 1.0, 0.0], FRAC_PI_2);
    let t2 = [-0.3, 0.6, 0.9];
    let g2 = RustSe3::from_parts(r2, t2);

    let composed = g1.compose(&g2);
    let multiplied = g1 * g2;

    approx_eq_matrix4(&composed.to_matrix(), &multiplied.to_matrix(), 1e-12);
    let point = [0.25, -0.5, 1.0];
    approx_eq(&composed.apply(point), &multiplied.apply(point), 1e-12);
}

#[test]
fn cmtm_mul_matches_compose() {
    let r1 = RustSo3::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
    let r2 = RustSo3::from_axis_angle([0.0, 1.0, 0.0], FRAC_PI_2);

    let c1 = RotationalCmtm::from_so3_with_derivatives(&r1, vec![[0.1, -0.2, 0.3]]);
    let c2 = RotationalCmtm::from_so3_with_derivatives(&r2, vec![[-0.05, 0.15, -0.25]]);

    let composed = c1.compose(&c2);
    let multiplied = c1 * c2;

    let composed_block = composed.to_block_matrix(None);
    let multiplied_block = multiplied.to_block_matrix(None);

    for (x, y) in composed_block.iter().zip(multiplied_block.iter()) {
        assert!((x - y).abs() < 1e-12, "expected {y}, got {x}");
    }
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
    approx_eq_matrix(
        &rotation.to_matrix(),
        &rebuilt.rotation().to_matrix(),
        1e-12,
    );
    approx_eq(&transform.translation(), &rebuilt.translation(), 1e-12);
}
