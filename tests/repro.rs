use std::f64::consts::FRAC_PI_2;

use mathroborust::{RustCmtm, RustSe3, RustSo3, mathrobo};
use nalgebra::{SMatrix, SVector};
use pyo3::prelude::*;

fn approx_eq(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < tol, "expected {y}, got {x}");
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
    approx_eq(&applied, &[1.0 + 3.0, 2.0, 3.0 - 1.0], 1e-12);
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
    let skew = SMatrix::<f64, 3, 3>::new(
        0.0,
        translation[2],
        -translation[1],
        -translation[2],
        0.0,
        translation[0],
        translation[1],
        -translation[0],
        0.0,
    );

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
fn python_interface_produces_same_values() {
    pyo3::prepare_freethreaded_python();

    let rotation = RustSo3::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2);
    let transform = RustSe3::from_parts(rotation, [0.5, -0.25, 1.25]);
    let expected = transform.apply([1.0, 0.0, 0.0]);

    Python::with_gil(|py| {
        let module = PyModule::new(py, "mathrobo").expect("create module");
        mathrobo(py, module).expect("initialize module");
        let se3_class = module.getattr("SE3").expect("get class");
        let instance = se3_class
            .call_method1(
                "from_axis_angle_translation",
                ([0.0_f64, 0.0, 1.0], FRAC_PI_2, [0.5_f64, -0.25, 1.25]),
            )
            .expect("construct SE3");

        let python_result: Vec<f64> = instance
            .call_method1("apply", ([1.0_f64, 0.0, 0.0],))
            .expect("apply")
            .extract()
            .expect("extract result");

        approx_eq(&python_result, &expected, 1e-12);
    });
}
