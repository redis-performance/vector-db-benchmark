//! NPY file reader for vector datasets.

use ndarray::Array2;
use ndarray_npy::ReadNpyExt;
use std::fs::File;
use std::io::BufReader;

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
