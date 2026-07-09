//! NPY file reader for vector datasets.

use ndarray::Array2;
use ndarray_npy::{ReadNpyExt, WriteNpyExt};
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// Read vectors from NPY file, returning (ids, vectors).
/// If normalize is true, each vector is divided by its L2 norm.
pub fn read_npy_vectors(path: &str, normalize: bool) -> Result<(Vec<i64>, Vec<Vec<f32>>), String> {
    let file = File::open(path).map_err(|e| format!("Failed to open NPY file: {}", e))?;
    let reader = BufReader::new(file);

    let arr: Array2<f32> =
        Array2::read_npy(reader).map_err(|e| format!("Failed to read NPY file: {}", e))?;

    let (rows, _cols) = arr.dim();

    // Convert to Vec<Vec<f32>>
    let mut vectors: Vec<Vec<f32>> = arr.rows().into_iter().map(|row| row.to_vec()).collect();

    if normalize {
        for vec in vectors.iter_mut() {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in vec.iter_mut() {
                    *x /= norm;
                }
            }
        }
    }

    let ids: Vec<i64> = (0..rows as i64).collect();

    Ok((ids, vectors))
}

/// Write vectors to an NPY file as a 2-D `f32` array (row-major), the same
/// layout [`read_npy_vectors`] expects. All rows must have identical length.
/// IDs are implicit row indices, matching the compound-format convention.
pub fn write_npy_vectors(path: &str, vectors: &[Vec<f32>]) -> Result<(), String> {
    let rows = vectors.len();
    let cols = vectors.first().map(|v| v.len()).unwrap_or(0);
    if vectors.iter().any(|v| v.len() != cols) {
        return Err("All vectors must have the same dimension".to_string());
    }

    let flat: Vec<f32> = vectors.iter().flatten().copied().collect();
    let arr = Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| format!("Failed to build array: {}", e))?;

    let file = File::create(path).map_err(|e| format!("Failed to create NPY file: {}", e))?;
    let writer = BufWriter::new(file);
    arr.write_npy(writer)
        .map_err(|e| format!("Failed to write NPY file: {}", e))?;
    Ok(())
}
