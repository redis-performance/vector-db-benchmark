#![no_main]
//! Structured / differential round-trip fuzzing of the CSR sparse-matrix codec:
//! `read_sparse_matrix(write_sparse_matrix(x)) == x`.
//!
//! This catches asymmetries a byte-fuzzer can't: a writer/reader offset mismatch,
//! silent value corruption, or dropped rows. Sizes are bounded via `arbitrary`
//! so the writer never chokes on absurd allocations.

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use vector_db_benchmark::readers::{read_sparse_matrix, write_sparse_matrix, SparseVector};

/// A bounded generator for a list of sparse rows. Indices/values are kept small
/// in count (not in magnitude) so the CSR writer stays cheap.
fn gen_rows(u: &mut Unstructured) -> arbitrary::Result<Vec<SparseVector>> {
    let n_rows = u8::arbitrary(u)? as usize % 33; // 0..=32 rows
    let mut rows = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        let nnz = u8::arbitrary(u)? as usize % 33; // 0..=32 nnz per row
        let mut indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        for _ in 0..nnz {
            indices.push(u32::arbitrary(u)?);
            let v = f32::arbitrary(u)?;
            // NaN never equals itself, which would break the round-trip
            // assertion for reasons unrelated to the codec; canonicalize it.
            values.push(if v.is_nan() { 0.0 } else { v });
        }
        rows.push(SparseVector { indices, values });
    }
    Ok(rows)
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let rows = match gen_rows(&mut u) {
        Ok(r) => r,
        Err(_) => return,
    };

    let tmp = match tempfile::NamedTempFile::new() {
        Ok(t) => t,
        Err(_) => return,
    };
    let path = match tmp.path().to_str() {
        Some(p) => p,
        None => return,
    };

    if write_sparse_matrix(path, &rows).is_err() {
        return;
    }
    let read = read_sparse_matrix(path).expect("valid CSR written by us must read back");
    assert_eq!(read, rows, "sparse CSR round-trip mismatch");
});
