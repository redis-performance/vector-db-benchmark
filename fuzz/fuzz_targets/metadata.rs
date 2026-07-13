#![no_main]
//! Fuzz JSON -> MetadataItem parsing directly from bytes.

use libfuzzer_sys::fuzz_target;
use vector_db_benchmark::readers::parse_metadata_from_json;

fuzz_target!(|data: &[u8]| {
    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(data) {
        let _ = parse_metadata_from_json(v);
    }
});
