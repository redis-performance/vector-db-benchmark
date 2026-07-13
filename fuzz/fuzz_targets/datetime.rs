#![no_main]
//! Fuzz `datetime_to_epoch_secs` — parses dataset datetime filter values via
//! chrono. Arbitrary strings must return `Some`/`None`, never panic.

use libfuzzer_sys::fuzz_target;
use vector_db_benchmark::parsers::datetime_to_epoch_secs;

fuzz_target!(|data: &[u8]| {
    let s = String::from_utf8_lossy(data);
    let _ = datetime_to_epoch_secs(&s);
});
