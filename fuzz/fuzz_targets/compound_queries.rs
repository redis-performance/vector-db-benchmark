#![no_main]
//! Fuzz the compound-format query reader (`tests.jsonl` -> queries + ground-truth
//! ids + filter conditions). This exercises the query-vector, `closest_ids`, and
//! `conditions` parsing that the byte-level readers never reach.

use libfuzzer_sys::fuzz_target;
use std::io::Write;
use vector_db_benchmark::readers::read_compound_queries;

fuzz_target!(|data: &[u8]| {
    let dir = match tempfile::tempdir() {
        Ok(d) => d,
        Err(_) => return,
    };
    let tests_path = dir.path().join("tests.jsonl");
    {
        let mut f = match std::fs::File::create(&tests_path) {
            Ok(f) => f,
            Err(_) => return,
        };
        if f.write_all(data).is_err() || f.flush().is_err() {
            return;
        }
    }
    let dir_str = match dir.path().to_str() {
        Some(s) => s,
        None => return,
    };
    // Run both with and without normalization (normalization touches the vector
    // math path).
    let _ = read_compound_queries(dir_str, false);
    let _ = read_compound_queries(dir_str, true);
});
