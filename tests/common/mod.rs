//! Shared helpers for `match_any` filter integration tests.
//!
//! Builds a small compound-format dataset (`vectors.npy` + `payloads.jsonl` +
//! `tests.jsonl`) whose queries each carry a `match_any` condition on a keyword
//! field, drives the real benchmark binary against a running engine, and reads
//! back the reported recall.
//!
//! Ground truth is brute-forced over ONLY the documents that satisfy the
//! filter, so a high recall proves the engine actually applied `match_any`
//! (it returned the OR-set's nearest neighbours, not the whole corpus's). An
//! engine that ignores the filter, or matches the wrong set, scores low recall.

#![allow(dead_code)]

use std::fs;
use std::path::{Path, PathBuf};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use vector_db_benchmark::readers::write_npy_vectors;

/// Keyword values assigned round-robin to documents by `id % 4`.
///
/// The last value is intentionally MULTI-WORD ("dark blue"): keyword matching
/// must be whole-value/exact, so `match_any ["red","blue"]` must NOT select a
/// "dark blue" doc. This makes the recall test sensitive to an engine that
/// tokenizes keyword fields (e.g. a regression to Weaviate's default `word`
/// tokenization, under which `Equal "blue"` would wrongly match "dark blue").
const COLORS: [&str; 4] = ["red", "green", "blue", "dark blue"];
/// The `match_any` set every query filters on (COLORS indices 0 and 2).
pub const MATCH_ANY_COLORS: [&str; 2] = ["red", "blue"];

const N_DOCS: usize = 400;
const N_QUERIES: usize = 10;
const TOP: usize = 10;

pub struct MatchAnyProject {
    /// Temp project root (leaked; lives for the process). Passed as cwd.
    pub root: PathBuf,
    pub dataset_name: String,
    pub top: usize,
    /// Number of documents satisfying the filter (sanity bound: >> top).
    pub matching_docs: usize,
}

fn color_for(id: usize) -> &'static str {
    COLORS[id % COLORS.len()]
}

fn matches_filter(id: usize) -> bool {
    MATCH_ANY_COLORS.contains(&color_for(id))
}

/// Build a full temp project (datasets + config + results dir) for a
/// `match_any` benchmark and return its root. `engine_configs_json` is the
/// verbatim contents of `experiments/configurations/test.json` (a JSON array
/// of engine configs). `dim` is the vector dimensionality.
pub fn write_match_any_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> MatchAnyProject {
    // Deterministic data/queries so ground truth is reproducible across engines.
    let mut rng = StdRng::seed_from_u64(0xA11CE);
    let gen_vec =
        |rng: &mut StdRng| -> Vec<f32> { (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect() };
    let vectors: Vec<Vec<f32>> = (0..N_DOCS).map(|_| gen_vec(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..N_QUERIES).map(|_| gen_vec(&mut rng)).collect();

    let l2 = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
            .sum()
    };
    // Nearest neighbours computed over the FILTERED corpus only.
    let filtered_gt = |q: &[f32]| -> Vec<i64> {
        let mut scored: Vec<(i64, f64)> = (0..N_DOCS)
            .filter(|id| matches_filter(*id))
            .map(|id| (id as i64, l2(q, &vectors[id])))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.iter().take(TOP).map(|(id, _)| *id).collect()
    };

    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp); // keep the dir alive for the subprocess

    let ds_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&ds_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();

    // Data vectors -> vectors.npy (implicit ids 0..N_DOCS).
    write_npy_vectors(ds_dir.join("vectors.npy").to_str().unwrap(), &vectors).unwrap();

    // Per-document metadata -> payloads.jsonl (keyword `color`, int `size`).
    let payloads: String = (0..N_DOCS)
        .map(|id| {
            serde_json::json!({ "color": color_for(id), "size": (id % 3) as i64 + 1 }).to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds_dir.join("payloads.jsonl"), payloads).unwrap();

    // Queries + match_any condition + filtered ground truth -> tests.jsonl.
    let any_vals: Vec<serde_json::Value> = MATCH_ANY_COLORS
        .iter()
        .map(|c| serde_json::json!(c))
        .collect();
    let tests: String = queries
        .iter()
        .map(|q| {
            serde_json::json!({
                "query": q.iter().map(|x| *x as f64).collect::<Vec<_>>(),
                "conditions": { "and": [ { "color": { "match": { "any": any_vals } } } ] },
                "closest_ids": filtered_gt(q),
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds_dir.join("tests.jsonl"), tests).unwrap();

    let datasets_json = serde_json::json!([{
        "name": dataset_name,
        "type": "tar",
        "path": format!("{}/", dataset_name),
        "distance": "l2",
        "vector_size": dim,
        "vector_count": N_DOCS,
        "schema": { "color": "keyword", "size": "int" },
    }]);
    fs::write(
        root.join("datasets/datasets.json"),
        serde_json::to_string_pretty(&datasets_json).unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("experiments/configurations/test.json"),
        engine_configs_json,
    )
    .unwrap();

    MatchAnyProject {
        root,
        dataset_name: dataset_name.to_string(),
        top: TOP,
        matching_docs: (0..N_DOCS).filter(|id| matches_filter(*id)).count(),
    }
}

// ── Generic filter-datatype fixtures (bool / uuid / full-text / datetime) ──
//
// Mirrors `write_match_any_project` but is parameterised by the filter under
// test. Each builds a compound (`tar`) dataset (`vectors.npy` +
// `payloads.jsonl` + `tests.jsonl`) whose queries carry a fixed `conditions`
// filter, with ground truth brute-forced over ONLY the documents that satisfy
// the filter. A high recall therefore proves the engine actually applied the
// filter (returned the filtered nearest neighbours, not the whole corpus's).

/// A built filter-benchmark project (same shape as `MatchAnyProject`).
pub struct FilterProject {
    pub root: PathBuf,
    pub dataset_name: String,
    pub top: usize,
    /// Number of documents satisfying the filter (sanity bound: >> top).
    pub matching_docs: usize,
}

/// Core builder: `schema` is the datasets.json `schema` object, `payload_for`
/// returns the per-document payload object, `condition` is the (shared) filter
/// JSON attached to every query, and `matches` decides whether a document id
/// satisfies the filter (used to brute-force the filtered ground truth).
fn write_filter_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
    schema: serde_json::Value,
    payload_for: impl Fn(usize) -> serde_json::Value,
    condition: serde_json::Value,
    matches: impl Fn(usize) -> bool,
) -> FilterProject {
    // Single shared filter → every query gets the same condition/predicate.
    write_filter_project_multi(
        dataset_name,
        engine_configs_json,
        dim,
        N_QUERIES,
        schema,
        payload_for,
        move |_q| condition.clone(),
        move |_q, id| matches(id),
    )
}

/// Generalised core builder that allows EACH query to carry its own `condition`
/// and its own `matches` predicate (used for multi-tenancy, where every query
/// is scoped to a different tenant). `condition_for(q)` is the filter JSON for
/// query `q`; `matches_for(q, id)` decides whether document `id` satisfies query
/// `q`'s filter (used to brute-force that query's tenant-local ground truth).
///
/// `matching_docs` is reported as the MINIMUM per-query match count, so the
/// caller's `matching_docs >= top` sanity check bounds the smallest tenant.
#[allow(clippy::too_many_arguments)]
fn write_filter_project_multi(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
    n_queries: usize,
    schema: serde_json::Value,
    payload_for: impl Fn(usize) -> serde_json::Value,
    condition_for: impl Fn(usize) -> serde_json::Value,
    matches_for: impl Fn(usize, usize) -> bool,
) -> FilterProject {
    let mut rng = StdRng::seed_from_u64(0xF117E);
    let gen_vec =
        |rng: &mut StdRng| -> Vec<f32> { (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect() };
    let vectors: Vec<Vec<f32>> = (0..N_DOCS).map(|_| gen_vec(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..n_queries).map(|_| gen_vec(&mut rng)).collect();

    let l2 = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
            .sum()
    };
    // Nearest neighbours for query `q`, computed over ONLY the docs that satisfy
    // query `q`'s filter (its tenant/subset).
    let filtered_gt = |q_idx: usize, q: &[f32]| -> Vec<i64> {
        let mut scored: Vec<(i64, f64)> = (0..N_DOCS)
            .filter(|id| matches_for(q_idx, *id))
            .map(|id| (id as i64, l2(q, &vectors[id])))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.iter().take(TOP).map(|(id, _)| *id).collect()
    };

    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp);

    let ds_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&ds_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();

    write_npy_vectors(ds_dir.join("vectors.npy").to_str().unwrap(), &vectors).unwrap();

    let payloads: String = (0..N_DOCS)
        .map(|id| payload_for(id).to_string())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds_dir.join("payloads.jsonl"), payloads).unwrap();

    let tests: String = queries
        .iter()
        .enumerate()
        .map(|(q_idx, q)| {
            serde_json::json!({
                "query": q.iter().map(|x| *x as f64).collect::<Vec<_>>(),
                "conditions": condition_for(q_idx),
                "closest_ids": filtered_gt(q_idx, q),
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds_dir.join("tests.jsonl"), tests).unwrap();

    let datasets_json = serde_json::json!([{
        "name": dataset_name,
        "type": "tar",
        "path": format!("{}/", dataset_name),
        "distance": "l2",
        "vector_size": dim,
        "vector_count": N_DOCS,
        "schema": schema,
    }]);
    fs::write(
        root.join("datasets/datasets.json"),
        serde_json::to_string_pretty(&datasets_json).unwrap(),
    )
    .unwrap();
    fs::write(
        root.join("experiments/configurations/test.json"),
        engine_configs_json,
    )
    .unwrap();

    // Smallest per-query match count (bounds the smallest tenant).
    let matching_docs = (0..n_queries)
        .map(|q_idx| (0..N_DOCS).filter(|id| matches_for(q_idx, *id)).count())
        .min()
        .unwrap_or(0);

    FilterProject {
        root,
        dataset_name: dataset_name.to_string(),
        top: TOP,
        matching_docs,
    }
}

/// bool filter: field `flag` (schema type `bool`), value `id % 2 == 0`. The
/// query filters `flag == true`, so half the corpus matches.
pub fn write_bool_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    write_filter_project(
        dataset_name,
        engine_configs_json,
        dim,
        serde_json::json!({ "flag": "bool" }),
        |id| serde_json::json!({ "flag": id % 2 == 0 }),
        serde_json::json!({ "and": [ { "flag": { "match": { "value": true } } } ] }),
        |id| id % 2 == 0,
    )
}

/// UUID values assigned round-robin by `id % 4`. The query filters on the first
/// UUID (exact keyword/TAG match), so a quarter of the corpus matches.
pub const UUIDS: [&str; 4] = [
    "550e8400-e29b-41d4-a716-446655440000",
    "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    "6ba7b811-9dad-11d1-80b4-00c04fd430c8",
    "6ba7b812-9dad-11d1-80b4-00c04fd430c8",
];

/// uuid filter: field `uid` (schema type `uuid`), exact match on `UUIDS[0]`.
pub fn write_uuid_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    write_filter_project(
        dataset_name,
        engine_configs_json,
        dim,
        serde_json::json!({ "uid": "uuid" }),
        |id| serde_json::json!({ "uid": UUIDS[id % UUIDS.len()] }),
        serde_json::json!({ "and": [ { "uid": { "match": { "value": UUIDS[0] } } } ] }),
        |id| id % UUIDS.len() == 0,
    )
}

/// full-text filter: field `body` (schema type `text`). Even docs contain the
/// term "quick"; odd docs do not. The query is a single-term full-text match on
/// "quick" (works on Redis TEXT and on Valkey's degraded tokenised-TAG path).
pub fn write_fulltext_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    write_filter_project(
        dataset_name,
        engine_configs_json,
        dim,
        serde_json::json!({ "body": "text" }),
        |id| {
            let body = if id % 2 == 0 {
                "the quick brown fox"
            } else {
                "lazy dog sleeps here"
            };
            serde_json::json!({ "body": body })
        },
        serde_json::json!({ "and": [ { "body": { "match": { "text": "quick" } } } ] }),
        |id| id % 2 == 0,
    )
}

/// datetime filter: field `ts` (schema type `datetime`), one ISO-8601 timestamp
/// per doc spaced one day apart from 2021-01-01. The query is an ISO range
/// `[day 100, day 300)`, selecting ids 100..=299 (200 docs).
pub fn write_datetime_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    use chrono::{Duration, TimeZone, Utc};
    let base = Utc.timestamp_opt(1_609_459_200, 0).unwrap(); // 2021-01-01T00:00:00Z
    let iso_for = move |day: i64| (base + Duration::days(day)).to_rfc3339();
    let gte = iso_for(100);
    let lt = iso_for(300);
    write_filter_project(
        dataset_name,
        engine_configs_json,
        dim,
        serde_json::json!({ "ts": "datetime" }),
        move |id| serde_json::json!({ "ts": iso_for(id as i64) }),
        serde_json::json!({ "and": [ { "ts": { "range": { "gte": gte, "lt": lt } } } ] }),
        |id| (100..300).contains(&id),
    )
}

// ── Multi-tenancy fixture ───────────────────────────────────────────────────
//
// Mirrors upstream qdrant/vector-db-benchmark's `random-768-*-tenants` scenario:
// MANY tenants share ONE index; every search is scoped to a single tenant via a
// keyword-equality filter on a `tenant` field, and recall is measured against
// the nearest neighbours WITHIN that tenant only. It reuses the existing
// keyword-TAG filter path (no new engine code) — the ONLY difference from the
// other filter fixtures is that each query targets a DIFFERENT tenant.
//
// Because the ground truth is tenant-local, recall is also a leakage detector:
// any cross-tenant document an engine wrongly returns is absent from that
// query's ground truth, so it cannot count toward recall AND it displaces a
// correct tenant-local neighbour — a leaking engine therefore scores low recall.

/// Number of tenants sharing the single index. With `N_DOCS` docs assigned
/// round-robin, each tenant owns `N_DOCS / N_TENANTS` docs (400 / 25 = 16 > TOP).
pub const N_TENANTS: usize = 25;

/// Tenant label for document / query index `k` (round-robin by `k % N_TENANTS`).
pub fn tenant_for(k: usize) -> String {
    format!("tenant_{}", k % N_TENANTS)
}

/// multi-tenancy filter: field `tenant` (schema type `keyword`), one tenant per
/// doc round-robin. Query `q` is scoped to tenant `q % N_TENANTS` via an exact
/// keyword match, with ground truth brute-forced over ONLY that tenant's docs.
pub fn write_tenant_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> FilterProject {
    // One query per tenant (N_TENANTS queries) so EVERY tenant label — including
    // the two-digit ones — is exercised as a query scope, not just as documents.
    write_filter_project_multi(
        dataset_name,
        engine_configs_json,
        dim,
        N_TENANTS,
        serde_json::json!({ "tenant": "keyword" }),
        |id| serde_json::json!({ "tenant": tenant_for(id) }),
        |q| {
            serde_json::json!({
                "and": [ { "tenant": { "match": { "value": tenant_for(q) } } } ]
            })
        },
        |q, id| id % N_TENANTS == q % N_TENANTS,
    )
}

/// Path to the compiled binary under test. Cargo exports
/// `CARGO_BIN_EXE_vector-db-benchmark` to integration tests automatically.
pub fn binary_path() -> PathBuf {
    if let Some(p) = std::env::var_os("CARGO_BIN_EXE_vector-db-benchmark") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/release/vector-db-benchmark")
}

/// Run the benchmark binary once for `engine`/`dataset`, with extra env vars
/// (engine host/port overrides). Returns whether it exited successfully;
/// prints stdout/stderr on failure.
pub fn run_binary(
    root: &Path,
    engine: &str,
    dataset: &str,
    host: &str,
    envs: &[(&str, &str)],
) -> bool {
    let mut cmd = std::process::Command::new(binary_path());
    cmd.args([
        "--engines",
        engine,
        "--datasets",
        dataset,
        "--host",
        host,
        "--skip-if-exists",
        "false",
    ])
    .current_dir(root);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run vector-db-benchmark");
    if !out.status.success() {
        eprintln!(
            "stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    out.status.success()
}

/// Read `results.mean_recall` from the engine's search result JSON.
pub fn read_recall(root: &Path, engine: &str) -> f64 {
    let pattern = format!("{}-*-search-*.json", engine);
    let dir = root.join("results");
    let path = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            glob::Pattern::new(&pattern)
                .unwrap()
                .matches(&p.file_name().unwrap().to_string_lossy())
        })
        .unwrap_or_else(|| panic!("no search result for {}", engine));
    let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    v["results"]["mean_recall"].as_f64().unwrap()
}

/// Read the per-query `results.recalls` array from the engine's search result
/// JSON. Each entry is one query's recall vs its (tenant-local) ground truth, so
/// asserting a floor on EVERY entry catches a single tenant that leaked or was
/// mis-scoped — stronger than only checking the mean.
pub fn read_recalls(root: &Path, engine: &str) -> Vec<f64> {
    let pattern = format!("{}-*-search-*.json", engine);
    let dir = root.join("results");
    let path = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            glob::Pattern::new(&pattern)
                .unwrap()
                .matches(&p.file_name().unwrap().to_string_lossy())
        })
        .unwrap_or_else(|| panic!("no search result for {}", engine));
    let v: serde_json::Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    v["results"]["recalls"]
        .as_array()
        .expect("recalls array")
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
