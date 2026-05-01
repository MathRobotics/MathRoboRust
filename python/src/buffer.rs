use nalgebra::{DMatrix, SMatrix};
use numpy::ndarray::Array2;
use numpy::{PyArray1, PyArray2};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyBufferError, PyValueError};
use pyo3::prelude::*;

pub fn write_vec3(py: Python<'_>, out: PyBuffer<f64>, value: [f64; 3]) -> PyResult<()> {
    let out = out
        .as_mut_slice(py)
        .ok_or_else(|| PyBufferError::new_err("output buffer must be writable and contiguous"))?;
    if out.len() < 3 {
        return Err(PyBufferError::new_err(
            "output buffer must have length >= 3",
        ));
    }

    out[0].set(value[0]);
    out[1].set(value[1]);
    out[2].set(value[2]);
    Ok(())
}

pub fn write_vec4(py: Python<'_>, out: PyBuffer<f64>, value: [f64; 4]) -> PyResult<()> {
    let out = out
        .as_mut_slice(py)
        .ok_or_else(|| PyBufferError::new_err("output buffer must be writable and contiguous"))?;
    if out.len() < 4 {
        return Err(PyBufferError::new_err(
            "output buffer must have length >= 4",
        ));
    }

    out[0].set(value[0]);
    out[1].set(value[1]);
    out[2].set(value[2]);
    out[3].set(value[3]);
    Ok(())
}

pub fn write_vec6(py: Python<'_>, out: PyBuffer<f64>, value: [f64; 6]) -> PyResult<()> {
    let out = out
        .as_mut_slice(py)
        .ok_or_else(|| PyBufferError::new_err("output buffer must be writable and contiguous"))?;
    if out.len() < 6 {
        return Err(PyBufferError::new_err(
            "output buffer must have length >= 6",
        ));
    }

    out[0].set(value[0]);
    out[1].set(value[1]);
    out[2].set(value[2]);
    out[3].set(value[3]);
    out[4].set(value[4]);
    out[5].set(value[5]);
    Ok(())
}

pub fn write_mat3(py: Python<'_>, out: PyBuffer<f64>, value: [[f64; 3]; 3]) -> PyResult<()> {
    let out = out
        .as_mut_slice(py)
        .ok_or_else(|| PyBufferError::new_err("output buffer must be writable and contiguous"))?;
    if out.len() < 9 {
        return Err(PyBufferError::new_err(
            "output buffer must have length >= 9",
        ));
    }

    let mut idx = 0;
    for row in value {
        for cell in row {
            out[idx].set(cell);
            idx += 1;
        }
    }
    Ok(())
}

pub fn write_mat4(py: Python<'_>, out: PyBuffer<f64>, value: [[f64; 4]; 4]) -> PyResult<()> {
    let out = out
        .as_mut_slice(py)
        .ok_or_else(|| PyBufferError::new_err("output buffer must be writable and contiguous"))?;
    if out.len() < 16 {
        return Err(PyBufferError::new_err(
            "output buffer must have length >= 16",
        ));
    }

    let mut idx = 0;
    for row in value {
        for cell in row {
            out[idx].set(cell);
            idx += 1;
        }
    }
    Ok(())
}

pub fn write_mat6(py: Python<'_>, out: PyBuffer<f64>, value: [[f64; 6]; 6]) -> PyResult<()> {
    let out = out
        .as_mut_slice(py)
        .ok_or_else(|| PyBufferError::new_err("output buffer must be writable and contiguous"))?;
    if out.len() < 36 {
        return Err(PyBufferError::new_err(
            "output buffer must have length >= 36",
        ));
    }

    let mut idx = 0;
    for row in value {
        for cell in row {
            out[idx].set(cell);
            idx += 1;
        }
    }
    Ok(())
}

pub fn vec_to_numpy<'py, const N: usize>(
    py: Python<'py>,
    value: [f64; N],
) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice(py, &value)
}

pub fn mat_to_numpy<'py, const R: usize, const C: usize>(
    py: Python<'py>,
    value: [[f64; C]; R],
) -> Bound<'py, PyArray2<f64>> {
    let flat = value.into_iter().flatten().collect::<Vec<_>>();
    let array = Array2::from_shape_vec((R, C), flat).expect("fixed-size matrix shape mismatch");
    PyArray2::from_owned_array(py, array)
}

pub fn s_matrix6_to_array(matrix: &SMatrix<f64, 6, 6>) -> [[f64; 6]; 6] {
    let mut out = [[0.0_f64; 6]; 6];
    for r in 0..6 {
        for c in 0..6 {
            out[r][c] = matrix[(r, c)];
        }
    }
    out
}

pub fn dmatrix_to_nested(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| (0..matrix.ncols()).map(|col| matrix[(row, col)]).collect())
        .collect()
}

pub fn nested_to_dmatrix(values: Vec<Vec<f64>>) -> PyResult<DMatrix<f64>> {
    let rows = values.len();
    let cols = values.first().map_or(0, Vec::len);
    if rows == 0 || cols == 0 {
        return Ok(DMatrix::<f64>::zeros(rows, cols));
    }

    if values.iter().any(|row| row.len() != cols) {
        return Err(PyValueError::new_err(
            "matrix rows must all have the same length",
        ));
    }

    let flat: Vec<f64> = values.into_iter().flatten().collect();
    Ok(DMatrix::<f64>::from_row_slice(rows, cols, &flat))
}

pub fn ensure_numpy(lib: Option<&str>) -> PyResult<()> {
    match lib {
        None | Some("numpy") => Ok(()),
        Some(other) => Err(PyValueError::new_err(format!(
            "unsupported library '{other}'; only 'numpy' is supported"
        ))),
    }
}
