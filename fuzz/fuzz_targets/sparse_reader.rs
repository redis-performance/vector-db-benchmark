#![no_main]
//! Fuzz the CSR sparse-matrix reader.
//!
//! The reader ingests attacker/corruption-controlled bytes and allocates/indexes
//! based on header values (n_row / n_col / nnz / index_pointer). We only care that
//! it never panics/overflows/OOMs: any malformed input must return `Err`.

use libfuzzer_sys::fuzz_target;
use std::io::Write;
use vector_db_benchmark::readers::read_sparse_matrix;

fuzz_target!(|data: &[u8]| {
    let mut tmp = match tempfile::NamedTempFile::new() {
        Ok(t) => t,
        Err(_) => return,
    };
    if tmp.write_all(data).is_err() {
        return;
    }
    if tmp.flush().is_err() {
        return;
    }
    let path = match tmp.path().to_str() {
        Some(p) => p,
        None => return,
    };
    // We only assert it doesn't crash; Err is fine.
    let _ = read_sparse_matrix(path);
});
