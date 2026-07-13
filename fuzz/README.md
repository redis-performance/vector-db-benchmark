# Fuzzing

Coverage-guided fuzzing of the **untrusted dataset parsers** in the
`vector_db_benchmark` library crate, using
[`cargo-fuzz`](https://github.com/rust-fuzz/cargo-fuzz) / libFuzzer.

These parsers ingest attacker- or corruption-controlled bytes and allocate /
index based on header values, so they are the highest-risk surface. The goal is
that **no malformed input ever panics, integer-overflows, or OOMs** — malformed
input must return `Err`.

## Targets

| Target | Parser | File |
|--------|--------|------|
| `sparse_reader` | `read_sparse_matrix` — CSR binary matrix | `src/readers/sparse_reader.rs` |
| `npy_reader` | `read_npy_vectors` — NumPy `.npy` header + f32 data | `src/readers/npy_reader.rs` |
| `jsonl_reader` | `read_jsonl_vectors` — JSONL arrays of floats | `src/readers/jsonl_reader.rs` |
| `compound_payloads` | `read_payloads_jsonl` — payloads.jsonl → metadata | `src/readers/compound_reader.rs` |
| `metadata` | `parse_metadata_from_json` — JSON → `MetadataItem` | `src/readers/metadata.rs` |
| `compound_queries` | `read_compound_queries` — tests.jsonl (query + conditions + ground-truth ids) | `src/readers/compound_reader.rs` |
| `compound_data` | `read_compound_data` — vectors.npy + payloads.jsonl directory | `src/readers/compound_reader.rs` |
| `parse_info` | `parse_info` — raw server `INFO` text → nested JSON | `src/parsers.rs` |
| `datetime` | `datetime_to_epoch_secs` — ISO-8601 filter value → epoch | `src/parsers.rs` |
| `ft_search` | `parse_ft_search_response` — `FT.SEARCH` RESP reply → hits | `src/parsers.rs` |
| `sparse_roundtrip` | `read_sparse_matrix(write_sparse_matrix(x)) == x` | `src/readers/sparse_reader.rs` |
| `npy_roundtrip` | `read_npy_vectors(write_npy_vectors(x)) == x` | `src/readers/npy_reader.rs` |

The `parse_info` / `datetime` / `ft_search` parsers ingest **untrusted server
responses** (any Redis-wire server: Redis / Valkey / Dragonfly / ElastiCache /
MemoryStore). They were extracted from the engine binary into the library crate
(`src/parsers.rs`) so coverage-guided fuzzing can reach them; the binaries call
the exact same implementations. The `ft_search` target frames arbitrary bytes
into a `redis::Value` with the `redis` crate's own wire parser, then runs our
extractor — fuzzing both RESP framing and our (id, score) extraction.

The two `*_roundtrip` targets are **differential**: they `assert_eq!` that a
value survives `write` → `read`, catching silent corruption / offset bugs that
crash-only fuzzing cannot. NaN is canonicalized (NaN != NaN would spuriously
fail) and sizes are bounded via `arbitrary` so the writer never over-allocates.

## Dictionaries

`fuzz/dictionaries/*.dict` seed libFuzzer with format tokens (NPY magic, JSON /
RESP / INFO / CSR / datetime tokens). Each matrix entry in `fuzz.yml` maps a
target to a dictionary, passed as `-dict=...`. Control bytes in a dict must use
`\xHH` escapes (libFuzzer only understands `\\`, `\"`, and `\xHH`).

## Requirements

- Nightly toolchain (libFuzzer needs it): `rustup toolchain install nightly`
- `cargo install cargo-fuzz`
- System deps for the library crate: `libhdf5-dev pkg-config`

## Run locally

```bash
# from the repo root
cargo +nightly fuzz build                      # build all targets
cargo +nightly fuzz run sparse_reader -- -max_total_time=60 -rss_limit_mb=2048
```

Each target writes crash reproducers to `fuzz/artifacts/<target>/`. Reproduce a
crash with:

```bash
cargo +nightly fuzz run sparse_reader fuzz/artifacts/sparse_reader/crash-XXXX
```

## Corpus

A few valid, minimal inputs are committed under `fuzz/corpus/<target>/valid*` so
fuzzing starts from valid structure. Everything else the fuzzer generates locally
is git-ignored; CI persists the accumulated corpus via `actions/cache`.

## Adding a target

1. Add `fuzz/fuzz_targets/<name>.rs` (`#![no_main]` + `fuzz_target!`), writing the
   bytes to a `tempfile::NamedTempFile` and calling the reader (or feeding bytes
   straight to a JSON parser).
2. Register a `[[bin]]` entry in `fuzz/Cargo.toml`.
3. Add the target name to the matrix in `.github/workflows/fuzz.yml`.

## When the fuzzer finds a crash

Harden the reader so the input returns `Err` (checked arithmetic for size math,
validate offsets before slicing, cap allocations by file size), then pin the
exact crashing bytes as a unit test in the reader module (`assert!(... .is_err())`)
so the fix stays fixed in the normal `cargo test` suite.
