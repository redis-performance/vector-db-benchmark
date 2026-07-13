#![no_main]
//! Fuzz the JSONL vector reader (line + JSON-array-of-floats parsing).

use libfuzzer_sys::fuzz_target;
use std::io::Write;
use vector_db_benchmark::readers::read_jsonl_vectors;

fuzz_target!(|data: &[u8]| {
    let mut tmp = match tempfile::NamedTempFile::new() {
        Ok(t) => t,
        Err(_) => return,
    };
    if tmp.write_all(data).is_err() || tmp.flush().is_err() {
        return;
    }
    let path = match tmp.path().to_str() {
        Some(p) => p,
        None => return,
    };
    let _ = read_jsonl_vectors(path, false);
    let _ = read_jsonl_vectors(path, true);
});
