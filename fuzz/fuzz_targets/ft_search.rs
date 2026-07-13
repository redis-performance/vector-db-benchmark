#![no_main]
//! Fuzz the FT.SEARCH response parser end-to-end from raw RESP bytes.
//!
//! The `redis` crate's own wire parser (`parse_redis_value`) frames arbitrary
//! bytes into a `redis::Value`; we then feed that into `parse_ft_search_response`
//! exactly as the engine does after `FT.SEARCH`. This fuzzes BOTH the RESP
//! framing and our (id, score) extraction. Neither may panic.

use libfuzzer_sys::fuzz_target;
use vector_db_benchmark::parsers::parse_ft_search_response;

fuzz_target!(|data: &[u8]| {
    // Not every byte string is valid RESP; only exercise the extractor when the
    // wire parser succeeds (that is precisely the surface the engine sees).
    if let Ok(value) = redis::parse_redis_value(data) {
        let _ = parse_ft_search_response(&value);
    }
});
