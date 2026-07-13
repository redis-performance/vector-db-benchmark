#![no_main]
//! Fuzz `parse_info` — parses raw, UNTRUSTED `INFO` text returned by any
//! Redis-wire server (Redis / Valkey / Dragonfly / ElastiCache / MemoryStore)
//! into nested JSON. It must never panic on arbitrary bytes.

use libfuzzer_sys::fuzz_target;
use vector_db_benchmark::parsers::parse_info;

fuzz_target!(|data: &[u8]| {
    // Server replies are bytes; parse_info takes &str. Lossy UTF-8 mirrors how
    // the engine decodes the reply before parsing.
    let s = String::from_utf8_lossy(data);
    let _ = parse_info(&s);
});
