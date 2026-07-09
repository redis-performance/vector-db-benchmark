//! Sparse vector reader.
//!
//! Reads sparse vectors stored as a binary CSR (compressed sparse row) matrix,
//! matching the format used by qdrant/vector-db-benchmark's `sparse_reader.py`:
//!
//! ```text
//! [ n_row: i64 ][ n_col: i64 ][ n_non_zero: i64 ]
//! [ index_pointer: i64 × (n_row + 1) ]
//! [ columns: i32 × n_non_zero ]
//! [ values:  f32 × n_non_zero ]
//! ```
//!
//! Row `i` is the sparse vector with `indices = columns[ip[i]..ip[i+1]]` and the
//! parallel `values` slice.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// A single sparse vector: parallel `indices` (dimension ids) and `values`.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
}

fn read_i64_le(r: &mut impl Read, n: usize) -> Result<Vec<i64>, String> {
    let mut buf = vec![0u8; n * 8];
    r.read_exact(&mut buf).map_err(|e| e.to_string())?;
    Ok(buf
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Parse a CSR file into a list of sparse vectors.
pub fn read_sparse_matrix(path: &str) -> Result<Vec<SparseVector>, String> {
    let file = File::open(Path::new(path)).map_err(|e| format!("open {}: {}", path, e))?;
    let mut r = BufReader::new(file);

    let sizes = read_i64_le(&mut r, 3)?;
    let (n_row, n_col, n_non_zero) = (sizes[0], sizes[1], sizes[2]);
    if n_row < 0 || n_col < 0 || n_non_zero < 0 {
        return Err(format!("invalid CSR header in {}", path));
    }
    let n_row = n_row as usize;
    let n_non_zero = n_non_zero as usize;

    let index_pointer = read_i64_le(&mut r, n_row + 1)?;
    if index_pointer.last().copied().unwrap_or(0) as usize != n_non_zero {
        return Err(format!("CSR index_pointer/nnz mismatch in {}", path));
    }

    let mut col_buf = vec![0u8; n_non_zero * 4];
    r.read_exact(&mut col_buf).map_err(|e| e.to_string())?;
    let columns: Vec<u32> = col_buf
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32)
        .collect();

    let mut val_buf = vec![0u8; n_non_zero * 4];
    r.read_exact(&mut val_buf).map_err(|e| e.to_string())?;
    let values: Vec<f32> = val_buf
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();

    let mut out = Vec::with_capacity(n_row);
    for i in 0..n_row {
        let start = index_pointer[i] as usize;
        let end = index_pointer[i + 1] as usize;
        out.push(SparseVector {
            indices: columns[start..end].to_vec(),
            values: values[start..end].to_vec(),
        });
    }
    Ok(out)
}

/// Write a list of sparse vectors as a CSR file (used by tests / fixtures).
pub fn write_sparse_matrix(path: &str, rows: &[SparseVector]) -> Result<(), String> {
    use std::io::Write;
    let n_row = rows.len() as i64;
    let n_col = rows
        .iter()
        .flat_map(|r| r.indices.iter())
        .map(|&i| i as i64 + 1)
        .max()
        .unwrap_or(0);
    let n_non_zero: i64 = rows.iter().map(|r| r.indices.len() as i64).sum();

    let mut buf = Vec::new();
    for v in [n_row, n_col, n_non_zero] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    let mut ip: i64 = 0;
    buf.extend_from_slice(&ip.to_le_bytes());
    for r in rows {
        ip += r.indices.len() as i64;
        buf.extend_from_slice(&ip.to_le_bytes());
    }
    for r in rows {
        for &c in &r.indices {
            buf.extend_from_slice(&(c as i32).to_le_bytes());
        }
    }
    for r in rows {
        for &val in &r.values {
            buf.extend_from_slice(&val.to_le_bytes());
        }
    }
    let mut f = File::create(Path::new(path)).map_err(|e| e.to_string())?;
    f.write_all(&buf).map_err(|e| e.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_csr_sparse_matrix() {
        let rows = vec![
            SparseVector {
                indices: vec![0, 5, 9],
                values: vec![1.0, 2.5, -3.0],
            },
            SparseVector {
                indices: vec![],
                values: vec![],
            },
            SparseVector {
                indices: vec![3],
                values: vec![0.75],
            },
        ];
        let dir = std::env::temp_dir();
        let path = dir
            .join(format!("vdb_sparse_test_{}.csr", std::process::id()))
            .to_str()
            .unwrap()
            .to_string();
        write_sparse_matrix(&path, &rows).unwrap();
        let read = read_sparse_matrix(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(read, rows);
    }
}
