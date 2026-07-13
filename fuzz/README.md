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
