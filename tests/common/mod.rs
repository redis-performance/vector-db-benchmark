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
use vector_db_benchmark::readers::{write_npy_vectors, write_sparse_matrix, SparseVector};

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

/// Distinct `int` `size` values assigned round-robin to documents by `id % 5`
/// (values 1..=5). The `match_any` int filter selects a STRICT SUBSET of these
/// (see `MATCH_ANY_SIZES`), so an engine that ignores the filter — or that
/// compares the int filter against string-typed storage (the HIGH bug this
/// test guards) — returns whole-corpus nearest neighbours and scores low recall.
fn size_for(id: usize) -> i64 {
    (id % 5) as i64 + 1
}

/// The int `match_any` set every query filters on: `size IN {1, 2}` — a strict
/// subset of the 5 possible sizes (~40% of docs match).
pub const MATCH_ANY_SIZES: [i64; 2] = [1, 2];

fn matches_int_filter(id: usize) -> bool {
    MATCH_ANY_SIZES.contains(&size_for(id))
}

/// Like `write_match_any_project`, but attaches the `match_any` filter to the
/// INT `size` field (`{size: {match: {any: [1, 2]}}}` → Mongo `{size:{$in:[…]}}`).
/// Ground truth is brute-forced over ONLY documents whose `size` is in the
/// IN-set, so high recall proves the engine applied a NUMERIC `$in` that matched
/// natively-stored integers. A filter-ignoring engine — or one that emits an
/// integer `$in` against string-stored sizes — scores low recall.
pub fn write_match_any_int_project(
    dataset_name: &str,
    engine_configs_json: &str,
    dim: usize,
) -> MatchAnyProject {
    // Deterministic data/queries so ground truth is reproducible across engines.
    let mut rng = StdRng::seed_from_u64(0x5133_u64);
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
    // Nearest neighbours computed over the size-FILTERED corpus only.
    let filtered_gt = |q: &[f32]| -> Vec<i64> {
        let mut scored: Vec<(i64, f64)> = (0..N_DOCS)
            .filter(|id| matches_int_filter(*id))
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

    write_npy_vectors(ds_dir.join("vectors.npy").to_str().unwrap(), &vectors).unwrap();

    // Per-document metadata: keyword `color` + int `size`.
    let payloads: String = (0..N_DOCS)
        .map(|id| serde_json::json!({ "color": color_for(id), "size": size_for(id) }).to_string())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds_dir.join("payloads.jsonl"), payloads).unwrap();

    // Queries + int match_any condition + filtered ground truth -> tests.jsonl.
    let any_vals: Vec<serde_json::Value> = MATCH_ANY_SIZES
        .iter()
        .map(|s| serde_json::json!(s))
        .collect();
    let tests: String = queries
        .iter()
        .map(|q| {
            serde_json::json!({
                "query": q.iter().map(|x| *x as f64).collect::<Vec<_>>(),
                "conditions": { "and": [ { "size": { "match": { "any": any_vals } } } ] },
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
        matching_docs: (0..N_DOCS).filter(|id| matches_int_filter(*id)).count(),
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

// ── Sparse-vector fixture ───────────────────────────────────────────────────
//
// Builds a small sparse (`type: "sparse"`) dataset: `data.csr` + `queries.csr`
// + `neighbours.jsonl`. Ground truth is brute-forced by sparse DOT PRODUCT and
// sorted DESCENDING (sparse similarity is MIPS — larger dot = more similar), so
// a high recall proves the engine ran a real sparse-index search. Sorting the
// wrong way (ascending, as if it were an L2 distance) would pick the least
// similar docs and silently zero out recall — hence the explicit `b.cmp a`.

/// A built sparse-benchmark project.
pub struct SparseProject {
    pub root: PathBuf,
    pub dataset_name: String,
    pub top: usize,
}

/// Build a temp project with a deterministic random sparse dataset and its
/// dot-product (descending) ground truth. `engine_configs_json` is the verbatim
/// `experiments/configurations/test.json`.
pub fn write_sparse_project(dataset_name: &str, engine_configs_json: &str) -> SparseProject {
    const DIM: usize = 300;
    const NNZ: usize = 10;
    const N: usize = 150;
    const Q: usize = 10;
    const TOP: usize = 10;

    // Fixed seed → reproducible data/queries/ground-truth across engines & runs.
    let mut rng = StdRng::seed_from_u64(0x5A5A_5EED);
    let mut make = |count: usize| -> Vec<SparseVector> {
        (0..count)
            .map(|_| {
                let mut idx: Vec<u32> = Vec::with_capacity(NNZ);
                while idx.len() < NNZ {
                    let c = rng.gen_range(0..DIM as u32);
                    if !idx.contains(&c) {
                        idx.push(c);
                    }
                }
                idx.sort_unstable();
                let values: Vec<f32> = (0..NNZ).map(|_| rng.gen_range(0.1..1.0)).collect();
                SparseVector {
                    indices: idx,
                    values,
                }
            })
            .collect()
    };
    let data = make(N);
    let queries = make(Q);

    // Brute-force sparse dot product; sort DESCENDING (MIPS).
    let dot = |a: &SparseVector, b: &SparseVector| -> f64 {
        let mut s = 0.0f64;
        for (i, &ai) in a.indices.iter().enumerate() {
            if let Some(j) = b.indices.iter().position(|&bi| bi == ai) {
                s += a.values[i] as f64 * b.values[j] as f64;
            }
        }
        s
    };
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|qv| {
            let mut scored: Vec<(i64, f64)> = data
                .iter()
                .enumerate()
                .map(|(i, d)| (i as i64, dot(qv, d)))
                .collect();
            // DESCENDING by dot product (b vs a). Do NOT flip this.
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            scored.iter().take(TOP).map(|(id, _)| *id).collect()
        })
        .collect();

    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp);

    let ds_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&ds_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();
    write_sparse_matrix(ds_dir.join("data.csr").to_str().unwrap(), &data).unwrap();
    write_sparse_matrix(ds_dir.join("queries.csr").to_str().unwrap(), &queries).unwrap();
    write_neighbours(&ds_dir, &neighbors);

    let datasets_json = serde_json::json!([{
        "name": dataset_name, "type": "sparse", "path": dataset_name,
        "distance": "dot", "vector_size": DIM, "vector_count": N,
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

    SparseProject {
        root,
        dataset_name: dataset_name.to_string(),
        top: TOP,
    }
}

// ── Hybrid (dense + sparse) fixture ─────────────────────────────────────────
//
// Builds a `type: "hybrid"` dataset: dense `vectors.npy`/`queries.npy` (L2) +
// sparse `data.csr`/`queries.csr` (dot/MIPS) + a SHARED `neighbours.jsonl`.
//
// GROUND-TRUTH / RECALL-FLOOR CHOICE — the ground truth R is recoverable ONLY
// via fusion; NEITHER modality alone reaches the floor.
//
// We deliberately do NOT brute-force the exact RRF order (its constant `k` is a
// server detail). Instead we PLANT, per query, K ground-truth docs split into
// two halves and two rings of single-modality distractors:
//   * R_dense (K/2 docs): dense ranks 0..K/2 (nearest by L2), but only MODERATE
//     sparse dot → in the sparse ranking they land at ranks K..3K/2 (below both
//     R_sparse and the sparse distractors).
//   * R_sparse (K/2 docs): sparse ranks 0..K/2 (highest dot), but only MODERATE
//     dense distance → in the dense ranking they land at ranks K..3K/2.
//   * D_d (K/2 dense-only distractors): dense ranks K/2..K (just past R_dense),
//     ~zero sparse dot → absent from the meaningful sparse list.
//   * D_s (K/2 sparse-only distractors): sparse ranks K/2..K (just below
//     R_sparse), dense-far → absent from the meaningful dense list.
//
// Consequence:
//   * dense-only top-K  = R_dense + D_d  → recall(R) ≈ 0.5
//   * sparse-only top-K = R_sparse + D_s → recall(R) ≈ 0.5
//   * fused (RRF) top-K = R (all K)      → recall(R) ≈ 1.0
// Under RRF every R doc appears in BOTH prefetches (its "off" modality ranks it
// at K..3K/2, still inside the prefetch depth), so it collects TWO 1/(k+rank)
// terms — and one of them has rank < K/2. Every distractor appears in only ONE
// prefetch with rank ≥ K/2, so its best (only meaningful) term is ≤ 1/(k+K/2) <
// 1/(k+K/2−1). Thus every R doc outscores every distractor for ANY k ≥ 0, and
// the fused top-K is exactly R. We assert a 0.9 FLOOR on the fused recall to
// absorb ANN slack, and the companion `*-dense` view (registered below) drives
// the SAME data through a plain dense search as a NEGATIVE CONTROL that MUST
// score < 0.6 — proving the dataset genuinely requires fusion. An inverted
// sparse orientation (ascending, as if L2), a dropped sparse prefetch, or a
// broken `Fusion::Rrf` all collapse the fused result toward one modality and
// fail the floor.

/// A built hybrid-benchmark project. `dataset_name` is the `type:"hybrid"`
/// dataset; `dense_dataset_name` is a dense-only (`type:"jsonl"`) VIEW over the
/// SAME dense vectors + SAME ground truth, used as a negative control.
pub struct HybridProject {
    pub root: PathBuf,
    pub dataset_name: String,
    pub dense_dataset_name: String,
    pub top: usize,
}

/// Build a temp project with a deterministic planted hybrid dataset whose fused
/// (RRF) top-K ground truth is recoverable ONLY by combining both modalities.
pub fn write_hybrid_project(dataset_name: &str, engine_configs_json: &str) -> HybridProject {
    const K: usize = 8; // top-k / ground-truth-set size (must be even)
    const HALF: usize = K / 2; // per-half / per-distractor-ring size
    const Q: usize = 6; // queries (and dense centre axes)
    const DENSE_DIM: usize = 16; // >= Q centre dims + HALF distractor bump dims
    const PER_Q: usize = 4 * HALF; // R_dense + R_sparse + D_d + D_s per query
    const FILLER: usize = 24;
    const N: usize = Q * PER_Q + FILLER; // = 120
    const BIG: f32 = 100.0; // centre magnitude → regions ~141 apart, origin ~100

    // Sparse layout: query q owns index block F_q = [q*HALF .. q*HALF+HALF);
    // dense-only distractors / filler use a disjoint "junk" block J (dot 0).
    const F_TOTAL: usize = Q * HALF;

    let mut rng = StdRng::seed_from_u64(0xB19_1DEA);
    let tiny = |rng: &mut StdRng| -> f32 { rng.gen_range(-0.01f32..0.01) };

    // Doc-id block layout for query q: base = q*PER_Q, then four HALF-sized rings.
    let base = |q: usize| q * PER_Q;
    let r_dense_id = |q: usize, j: usize| base(q) + j; //            [base,       base+HALF)
    let r_sparse_id = |q: usize, j: usize| base(q) + HALF + j; //    [base+HALF,  base+2HALF)
    let d_d_id = |q: usize, j: usize| base(q) + 2 * HALF + j; //     [base+2HALF, base+3HALF)
    let d_s_id = |q: usize, j: usize| base(q) + 3 * HALF + j; //     [base+3HALF, base+4HALF)
    let filler_start = Q * PER_Q;

    // Dense centre for query q: BIG on axis q, else 0.
    let centre = |q: usize| -> Vec<f32> {
        let mut v = vec![0.0f32; DENSE_DIM];
        v[q] = BIG;
        v
    };
    // A dense doc = centre + `mag` along a distractor axis (dims Q..DENSE_DIM),
    // so its L2 distance from the query (= centre) is exactly `mag`.
    let offset_from = |c: &[f32], mag: f32, j: usize| -> Vec<f32> {
        let mut v = c.to_vec();
        v[Q + (j % (DENSE_DIM - Q))] += mag;
        v
    };

    let junk: Vec<u32> = (F_TOTAL..F_TOTAL + HALF).map(|i| i as u32).collect();

    let mut dense: Vec<Vec<f32>> = vec![vec![0.0f32; DENSE_DIM]; N];
    let mut sparse: Vec<SparseVector> = vec![
        SparseVector {
            indices: vec![],
            values: vec![]
        };
        N
    ];

    for q in 0..Q {
        let c = centre(q);
        let f_q: Vec<u32> = (q * HALF..q * HALF + HALF).map(|i| i as u32).collect();
        // Per-doc tiny increments break ties so tiers stay crisply ordered.
        let sp = |indices: &[u32], val: f32, j: usize| SparseVector {
            indices: indices.to_vec(),
            values: indices.iter().map(|_| val + 0.001 * j as f32).collect(),
        };
        for j in 0..HALF {
            // R_dense: dense dist 1.0 (ranks 0..HALF); sparse dot ~ HALF*1 (low).
            dense[r_dense_id(q, j)] = offset_from(&c, 1.0, j);
            sparse[r_dense_id(q, j)] = sp(&f_q, 1.0, j);

            // D_d: dense dist 2.0 (ranks HALF..K); sparse = junk → dot 0.
            dense[d_d_id(q, j)] = offset_from(&c, 2.0, HALF + j);
            sparse[d_d_id(q, j)] = sp(&junk, 1.0, j);

            // R_sparse: dense dist 3.0 (ranks K..3K/2); sparse dot ~ HALF*3 (top).
            dense[r_sparse_id(q, j)] = offset_from(&c, 3.0, 2 * HALF + j);
            sparse[r_sparse_id(q, j)] = sp(&f_q, 3.0, j);

            // D_s: dense ≈ origin (dist ~BIG, absent from dense top); sparse dot ~
            // HALF*2 (ranks HALF..K, between R_sparse and R_dense).
            let mut ds_v = vec![0.0f32; DENSE_DIM];
            for x in ds_v.iter_mut() {
                *x += tiny(&mut rng);
            }
            dense[d_s_id(q, j)] = ds_v;
            sparse[d_s_id(q, j)] = sp(&f_q, 2.0, j);
        }
    }
    // Filler: dense ≈ origin, sparse in junk block (dot 0 with every query).
    for id in filler_start..N {
        let mut fv = vec![0.0f32; DENSE_DIM];
        for x in fv.iter_mut() {
            *x += tiny(&mut rng);
        }
        dense[id] = fv;
        sparse[id] = SparseVector {
            indices: junk.clone(),
            values: vec![1.0f32; HALF],
        };
    }

    // Queries: dense = centre_q, sparse = ones on F_q. Ground truth R_q =
    // R_dense ∪ R_sparse (the full K planted docs).
    let mut dense_q: Vec<Vec<f32>> = Vec::with_capacity(Q);
    let mut sparse_q: Vec<SparseVector> = Vec::with_capacity(Q);
    let mut neighbours: Vec<Vec<i64>> = Vec::with_capacity(Q);
    for q in 0..Q {
        dense_q.push(centre(q));
        sparse_q.push(SparseVector {
            indices: (q * HALF..q * HALF + HALF).map(|i| i as u32).collect(),
            values: vec![1.0f32; HALF],
        });
        let mut gt: Vec<i64> = (0..HALF).map(|j| r_dense_id(q, j) as i64).collect();
        gt.extend((0..HALF).map(|j| r_sparse_id(q, j) as i64));
        neighbours.push(gt);
    }

    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp);

    let ds_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&ds_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();

    write_npy_vectors(ds_dir.join("vectors.npy").to_str().unwrap(), &dense).unwrap();
    write_npy_vectors(ds_dir.join("queries.npy").to_str().unwrap(), &dense_q).unwrap();
    write_sparse_matrix(ds_dir.join("data.csr").to_str().unwrap(), &sparse).unwrap();
    write_sparse_matrix(ds_dir.join("queries.csr").to_str().unwrap(), &sparse_q).unwrap();
    write_neighbours(&ds_dir, &neighbours);

    // Dense-only VIEW (negative control): same dense vectors + same ground truth
    // as a plain jsonl dataset, so an ordinary dense search can be run on it.
    let dense_dataset_name = format!("{dataset_name}-dense");
    let dv_dir = root.join("datasets").join(&dense_dataset_name);
    fs::create_dir_all(&dv_dir).unwrap();
    write_jsonl_vectors(&dv_dir.join("vectors.jsonl"), &dense);
    write_jsonl_vectors(&dv_dir.join("queries.jsonl"), &dense_q);
    write_neighbours(&dv_dir, &neighbours);

    let datasets_json = serde_json::json!([
        {
            "name": dataset_name, "type": "hybrid", "path": dataset_name,
            "distance": "l2", "vector_size": DENSE_DIM, "vector_count": N,
        },
        {
            "name": dense_dataset_name, "type": "jsonl", "path": format!("{dense_dataset_name}/"),
            "distance": "l2", "vector_size": DENSE_DIM, "vector_count": N,
        },
    ]);
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

    HybridProject {
        root,
        dataset_name: dataset_name.to_string(),
        dense_dataset_name,
        top: K,
    }
}

/// Write `vectors` as a `.jsonl` file (one JSON float-array per line), the
/// layout the `type:"jsonl"` reader expects.
fn write_jsonl_vectors(path: &Path, vectors: &[Vec<f32>]) {
    let body = vectors
        .iter()
        .map(|v| serde_json::to_string(&v.iter().map(|x| *x as f64).collect::<Vec<_>>()).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(path, body).unwrap();
}

/// Write `neighbours.jsonl` (one JSON id-array per line) into `ds_dir`.
fn write_neighbours(ds_dir: &Path, neighbours: &[Vec<i64>]) {
    let nb = neighbours
        .iter()
        .map(|nn| serde_json::to_string(nn).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds_dir.join("neighbours.jsonl"), nb).unwrap();
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
