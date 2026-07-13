#![no_main]
//! Structured / differential round-trip fuzzing of the NPY codec:
//! `read_npy_vectors(write_npy_vectors(x)).1 == x`.
//!
//! Catches writer/reader layout mismatches and silent f32 corruption that
//! crash-only byte fuzzing cannot. All rows share one dimension (the writer
//! requires it) and NaN is canonicalized (NaN != NaN would spuriously fail).

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use vector_db_benchmark::readers::{read_npy_vectors, write_npy_vectors};

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let rows = u8::arbitrary(&mut u).unwrap_or(0) as usize % 65; // 0..=64
    let dim = u8::arbitrary(&mut u).unwrap_or(0) as usize % 33; // 0..=32

    let mut vectors: Vec<Vec<f32>> = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut row = Vec::with_capacity(dim);
        for _ in 0..dim {
            let v = f32::arbitrary(&mut u).unwrap_or(0.0);
            row.push(if v.is_nan() { 0.0 } else { v });
        }
        vectors.push(row);
    }

    let tmp = match tempfile::NamedTempFile::new() {
        Ok(t) => t,
        Err(_) => return,
    };
    let path = match tmp.path().to_str() {
        Some(p) => p,
        None => return,
    };

    if write_npy_vectors(path, &vectors).is_err() {
        return;
    }
    let (ids, read) = read_npy_vectors(path, false).expect("valid NPY written by us must read back");
    assert_eq!(read, vectors, "NPY round-trip mismatch");
    assert_eq!(ids.len(), vectors.len(), "NPY id count mismatch");
});
