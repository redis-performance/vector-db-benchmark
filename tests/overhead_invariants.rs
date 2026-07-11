//! Overhead-invariant guard tests (regression tripwire).
//!
//! These are pure source-scanning tests: they read the engine `.rs` files as
//! TEXT and assert that the measurement-overhead fixes from PRs #108 / #110 /
//! #111 / #113 stay in place. They do NOT need a database — they run in the
//! normal `cargo test` set and fail CI the moment someone reintroduces one of
//! the anti-patterns those PRs removed.
//!
//! Two invariants are locked in:
//!
//!   INV-2  No per-query cross-thread `Mutex<Vec>` push inside a timed worker
//!          loop. Every per-query latency/quality sample must land in a
//!          thread-local buffer that is merged on join — NOT pushed through a
//!          single `Arc<Mutex<Vec<f64>>>` that serializes workers at high
//!          parallelism. (#108 for the Redis-family filter-only/mixed paths,
//!          #110 for turbopuffer.)
//!
//!   INV-3  One stats path. Every search / filter-only / mixed path computes
//!          its latency percentiles through the shared `compute_search_stats`
//!          (linear-interpolation) helper, NOT a hand-rolled nearest-rank
//!          `(len as f64 * q) as usize` index — the biased method that made
//!          `p99 == max` for `N <= 100`. (#108.)
//!
//! Keep the assertions on DISTINCTIVE substrings that will not false-positive on
//! unrelated code (e.g. the legitimate `errors.lock().unwrap().push(e)` in the
//! UPLOAD paths must stay allowed). The percentile-parity behaviour itself is
//! covered by the unit test `engine::tests::filter_mixed_stats_use_linear_percentiles`
//! (p99 == 99.01 on 1..=100); this file guards the SOURCE shape that keeps that
//! true across all engines.

use std::fs;
use std::path::PathBuf;

/// Engine source files that own a timed per-query worker loop whose sample
/// buffers were converted from `Arc<Mutex<Vec>>` to thread-local (#108/#110/#111)
/// and whose percentiles route through `compute_search_stats` (#108).
const ENGINE_FILES: &[&str] = &[
    "redis.rs",
    "valkey.rs",
    "vectorsets.rs",
    "mongodb_engine.rs",
    "turbopuffer.rs",
];

fn engine_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/bin/vector_db_benchmark/engine")
}

fn read_engine(file: &str) -> String {
    let path = engine_dir().join(file);
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {}", path.display(), e))
}

/// INV-2 — no per-query metric buffer is pushed through a cross-thread mutex in
/// a timed loop. We ban the EXACT sample-buffer lock-push idioms that #108/#110
/// removed. Each pattern below named a per-query metric buffer
/// (`search_times`/`precisions`/`recalls`/`mrrs`/`ndcgs`/`update_times`) that
/// used to be an `Arc<Mutex<Vec<f64>>>`; if any reappears, a timed worker loop
/// is once again serializing on a shared lock per query.
///
/// NOTE: this deliberately does NOT ban `.lock().unwrap().push(` in general —
/// the upload/error paths legitimately collect errors under a mutex
/// (`errors.lock().unwrap().push(e)`), which is off the timed hot path.
#[test]
fn inv2_no_per_query_mutex_push_in_timed_loops() {
    // (banned substring, what it used to guard)
    let banned: &[(&str, &str)] = &[
        (
            "search_times.lock().unwrap().push",
            "per-query search latency pushed through a shared Mutex<Vec> (was #108 FIX1)",
        ),
        (
            "update_times.lock().unwrap().push",
            "per-op update latency pushed through a shared Mutex<Vec> (mixed path, #108 FIX1)",
        ),
        (
            "precisions.lock().unwrap().push",
            "per-query precision pushed through a shared Mutex<Vec> (mixed path, #108 FIX1)",
        ),
        (
            "recalls.lock().unwrap().push",
            "per-query recall pushed through a shared Mutex<Vec> (mixed path, #108 FIX1)",
        ),
        (
            "mrrs.lock().unwrap().push",
            "per-query MRR pushed through a shared Mutex<Vec> (mixed path, #108 FIX1)",
        ),
        (
            "ndcgs.lock().unwrap().push",
            "per-query NDCG pushed through a shared Mutex<Vec> (mixed path, #108 FIX1)",
        ),
        (
            "times.lock().unwrap().push",
            "per-query latency pushed through a shared Mutex<Vec> in a timed loop",
        ),
    ];

    let mut violations = Vec::new();
    for &file in ENGINE_FILES {
        let src = read_engine(file);
        for &(pat, why) in banned {
            let count = src.matches(pat).count();
            if count != 0 {
                violations.push(format!(
                    "  {} contains `{}` ({}x) — {}",
                    file, pat, count, why
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "INV-2 VIOLATED: a per-query metric buffer is pushed through a cross-thread \
         Mutex<Vec> inside a timed worker loop. Accumulate into a THREAD-LOCAL Vec \
         and merge on join instead (see redis.rs::search). Offenders:\n{}",
        violations.join("\n")
    );
}

/// INV-3a — every timed engine still routes its stats through the single shared
/// `compute_search_stats` helper. If an engine stops referencing it, it has
/// almost certainly grown a private (likely nearest-rank) percentile path again.
#[test]
fn inv3_all_engines_use_shared_compute_search_stats() {
    let mut missing = Vec::new();
    for &file in ENGINE_FILES {
        let src = read_engine(file);
        if !src.contains("compute_search_stats") {
            missing.push(file);
        }
    }
    assert!(
        missing.is_empty(),
        "INV-3 VIOLATED: these engines no longer reference `compute_search_stats`, so \
         their search/filter-only/mixed stats are no longer on the shared \
         linear-percentile footing: {:?}",
        missing
    );
}

/// INV-3b — the hand-rolled nearest-rank percentile idiom `(len as f64 * q) as
/// usize` (which made `p99 == max` for `N <= 100`) must not reappear in any
/// engine file. The three multiplier substrings below are distinctive of that
/// specific indexing pattern and do not occur in correct code (which calls
/// `percentile_linear` / `compute_search_stats`).
#[test]
fn inv3_no_nearest_rank_percentile_indexing() {
    let banned: &[&str] = &[
        "as f64 * 0.50) as usize",
        "as f64 * 0.95) as usize",
        "as f64 * 0.99) as usize",
    ];

    let mut violations = Vec::new();
    // Scan every engine source, not just the five — the biased idiom is wrong
    // anywhere it appears.
    for entry in fs::read_dir(engine_dir()).expect("read engine dir") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let src = fs::read_to_string(&path).expect("read engine source");
        for &pat in banned {
            if src.contains(pat) {
                violations.push(format!(
                    "  {} contains `{}`",
                    path.file_name().unwrap().to_string_lossy(),
                    pat
                ));
            }
        }
    }

    assert!(
        violations.is_empty(),
        "INV-3 VIOLATED: a hand-rolled nearest-rank percentile index reappeared \
         (`(len as f64 * q) as usize`). This biases p99 upward and makes p99 == max \
         for N <= 100 — route through `percentile_linear` / `compute_search_stats` \
         instead. Offenders:\n{}",
        violations.join("\n")
    );
}
