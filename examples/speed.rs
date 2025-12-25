use std::time::Instant;

use mathroborust::{RustCmtm, RustSe3, RustSo3};

fn time_operation(label: &str, mut f: impl FnMut()) {
    let start = Instant::now();
    f();
    let elapsed = start.elapsed();
    println!("{label}: {:.2?}", elapsed);
}

fn main() {
    let rotation = RustSo3::from_axis_angle([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
    let transform = RustSe3::from_parts(rotation.clone(), [0.25, -0.5, 1.0]);
    let adjoint = RustCmtm::from_se3(&transform);

    let iterations = 200_000;

    time_operation("SO3 apply", || {
        let mut acc = [1.0_f64, 0.0, 0.0];
        for _ in 0..iterations {
            acc = rotation.apply(acc);
        }
        assert!(acc.iter().all(|v| v.is_finite()));
    });

    time_operation("SE3 apply", || {
        let mut acc = [1.0_f64, 0.0, 0.0];
        for _ in 0..iterations {
            acc = transform.apply(acc);
        }
        assert!(acc.iter().all(|v| v.is_finite()));
    });

    time_operation("CMTM apply_twist", || {
        let mut twist = [0.1_f64, 0.2, 0.3, 1.0, 2.0, 3.0];
        for _ in 0..iterations {
            twist = adjoint.apply_twist(twist);
        }
        assert!(twist.iter().all(|v| v.is_finite()));
    });
}
