#![no_main]
//! Fuzz `read_compound_data` (a DIRECTORY reader: `vectors.npy` + `payloads.jsonl`).
//!
//! We write a *valid* NPY of `rows`x`dim` zeros (so the reader gets past the
//! vector stage) and feed the remaining fuzz bytes as `payloads.jsonl`. This
//! exercises the count-matching glue between vectors and payloads plus the
//! payload JSON parsing in the compound context — the byte-level readers never
//! reach this cross-file path.

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use std::io::Write;
use vector_db_benchmark::readers::{read_compound_data, write_npy_vectors};

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    // Bound the NPY dimensions hard so we never build a huge array.
    let rows = u8::arbitrary(&mut u).unwrap_or(0) as usize % 33; // 0..=32
    let dim = u8::arbitrary(&mut u).unwrap_or(0) as usize % 17; // 0..=16
    let payloads = u.take_rest();

    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(_) => return,
    };

    let vectors: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; rows];
    let npy_path = dir.path().join("vectors.npy");
    if write_npy_vectors(npy_path.to_str().unwrap(), &vectors).is_err() {
        return;
    }

    let payloads_path = dir.path().join("payloads.jsonl");
    {
        let mut f = match std::fs::File::create(&payloads_path) {
            Ok(f) => f,
            Err(_) => return,
        };
        if f.write_all(payloads).is_err() || f.flush().is_err() {
            return;
        }
    }

    let dir_str = match dir.path().to_str() {
        Some(s) => s,
        None => return,
    };
    let _ = read_compound_data(dir_str, false);
    let _ = read_compound_data(dir_str, true);
});
