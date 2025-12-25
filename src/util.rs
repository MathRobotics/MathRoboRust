use nalgebra::{Matrix3, Vector3};

pub fn vector3_from_array(values: [f64; 3]) -> Vector3<f64> {
    Vector3::new(values[0], values[1], values[2])
}

pub fn vector3_to_array(vector: &Vector3<f64>) -> [f64; 3] {
    [vector.x, vector.y, vector.z]
}

pub fn skew_symmetric(vector: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -vector.z, vector.y, vector.z, 0.0, -vector.x, -vector.y, vector.x, 0.0,
    )
}
