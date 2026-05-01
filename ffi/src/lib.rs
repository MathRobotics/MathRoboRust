use mathroborust::{Se3, So3};
use mathroborust::lie::matrix_to_array;

fn read_vec3(ptr: *const f64) -> Option<[f64; 3]> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: caller must provide at least 3 contiguous f64 values.
    let slice = unsafe { std::slice::from_raw_parts(ptr, 3) };
    Some([slice[0], slice[1], slice[2]])
}

fn read_vec6(ptr: *const f64) -> Option<[f64; 6]> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: caller must provide at least 6 contiguous f64 values.
    let slice = unsafe { std::slice::from_raw_parts(ptr, 6) };
    Some([slice[0], slice[1], slice[2], slice[3], slice[4], slice[5]])
}

fn write_vec3(ptr: *mut f64, value: [f64; 3]) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: caller must provide at least 3 contiguous f64 values.
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, 3) };
    slice[0] = value[0];
    slice[1] = value[1];
    slice[2] = value[2];
}

fn write_mat4(ptr: *mut f64, value: [[f64; 4]; 4]) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: caller must provide at least 16 contiguous f64 values.
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, 16) };
    let mut idx = 0;
    for r in 0..4 {
        for c in 0..4 {
            slice[idx] = value[r][c];
            idx += 1;
        }
    }
}

fn write_mat6(ptr: *mut f64, value: [[f64; 6]; 6]) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: caller must provide at least 36 contiguous f64 values.
    let slice = unsafe { std::slice::from_raw_parts_mut(ptr, 36) };
    let mut idx = 0;
    for r in 0..6 {
        for c in 0..6 {
            slice[idx] = value[r][c];
            idx += 1;
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mr_so3_new(axis: *const f64, angle: f64) -> *mut So3 {
    let axis = match read_vec3(axis) {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };
    let rotation = So3::from_axis_angle(axis, angle);
    Box::into_raw(Box::new(rotation))
}

#[unsafe(no_mangle)]
pub extern "C" fn mr_so3_free(ptr: *mut So3) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: reclaim ownership from mr_so3_new.
    unsafe { drop(Box::from_raw(ptr)) };
}

#[unsafe(no_mangle)]
pub extern "C" fn mr_so3_apply(ptr: *const So3, vector: *const f64, out: *mut f64) {
    if ptr.is_null() {
        return;
    }
    let vector = match read_vec3(vector) {
        Some(v) => v,
        None => return,
    };

    // SAFETY: ptr comes from mr_so3_new and is valid for reads.
    let rotation = unsafe { &*ptr };
    let result = rotation.apply(vector);
    write_vec3(out, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn mr_se3_new(
    axis: *const f64,
    angle: f64,
    translation: *const f64,
) -> *mut Se3 {
    let axis = match read_vec3(axis) {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };
    let translation = match read_vec3(translation) {
        Some(v) => v,
        None => return std::ptr::null_mut(),
    };
    let transform = Se3::from_axis_angle_translation(axis, angle, translation);
    Box::into_raw(Box::new(transform))
}

#[unsafe(no_mangle)]
pub extern "C" fn mr_se3_free(ptr: *mut Se3) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: reclaim ownership from mr_se3_new.
    unsafe { drop(Box::from_raw(ptr)) };
}

#[unsafe(no_mangle)]
pub extern "C" fn mr_se3_apply(ptr: *const Se3, point: *const f64, out: *mut f64) {
    if ptr.is_null() {
        return;
    }
    let point = match read_vec3(point) {
        Some(v) => v,
        None => return,
    };

    // SAFETY: ptr comes from mr_se3_new and is valid for reads.
    let transform = unsafe { &*ptr };
    let result = transform.apply(point);
    write_vec3(out, result);
}

#[unsafe(no_mangle)]
pub extern "C" fn mr_se3_exp(twist: *const f64, scale: f64, out: *mut f64) {
    let twist = match read_vec6(twist) {
        Some(v) => v,
        None => return,
    };
    let matrix = Se3::exp(twist, Some(scale));
    write_mat4(out, matrix);
}

#[unsafe(no_mangle)]
pub extern "C" fn mr_se3_adjoint(ptr: *const Se3, out: *mut f64) {
    if ptr.is_null() {
        return;
    }
    // SAFETY: ptr comes from mr_se3_new and is valid for reads.
    let transform = unsafe { &*ptr };
    let adjoint = transform.adjoint();
    write_mat6(out, matrix_to_array(&adjoint));
}
