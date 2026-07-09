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
