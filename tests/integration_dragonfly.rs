//! Integration tests for the Dragonfly engine (Dragonfly Search FT.* KNN).
//!
//! Requires Dragonfly running on port 6385.
//! Start with: docker compose -f tests/docker-compose.test.yml up -d dragonfly
//! Run with:   DRAGONFLY_PORT=6385 cargo test --test integration_dragonfly -- --test-threads=1
//!
//! Scope: pure vector KNN only (the engine implements no filters). The fixtures
//! therefore use whole-corpus COSINE ground truth so recall reflects the vector
//! index quality alone, matching the `DISTANCE_METRIC COSINE` the engine builds.

use std::fs;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use redis::Connection;

mod common;

/// (vectors, queries, cosine-ground-truth neighbours) produced by [`make_data`].
type KnnData = (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<i64>>);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_port() -> u16 {
    std::env::var("DRAGONFLY_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6385)
}

const TEST_HOST: &str = "127.0.0.1";

fn get_test_connection() -> Connection {
    let url = format!("redis://{}:{}/", TEST_HOST, test_port());
    let client = redis::Client::open(url.as_str()).expect("Failed to create Dragonfly client");
    client
        .get_connection()
        .expect("Failed to connect to Dragonfly. Is dragonfly running on the test port?")
}

fn wait_for_dragonfly() {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let url = format!("redis://{}:{}/", TEST_HOST, test_port());
        if let Ok(client) = redis::Client::open(url.as_str()) {
            if let Ok(mut conn) = client.get_connection() {
                let pong: Result<String, _> = redis::cmd("PING").query(&mut conn);
                if pong.is_ok() {
                    return;
                }
            }
        }
        if Instant::now() > deadline {
            panic!("Dragonfly not available on port {} after 30s", test_port());
        }
        thread::sleep(Duration::from_millis(200));
    }
}

fn flush_db(conn: &mut Connection) {
    // Drop all FT indexes first, then flush leftover keys.
    if let Ok(indexes) = redis::cmd("FT._LIST").query::<Vec<String>>(conn) {
        for idx_name in indexes {
            let _ = redis::cmd("FT.DROPINDEX").arg(&idx_name).query::<()>(conn);
        }
    }
    let _: () = redis::cmd("FLUSHALL").query(conn).unwrap();
}

/// Cosine distance `1 - cos_sim` (scale-invariant; matches DISTANCE_METRIC COSINE
/// ranking whether or not the vectors are normalized).
fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    1.0 - dot / (na.sqrt() * nb.sqrt())
}

fn brute_force_cosine_neighbors(query: &[f32], vectors: &[Vec<f32>], top: usize) -> Vec<i64> {
    let mut dists: Vec<(i64, f64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i as i64, cosine_distance(query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(top).map(|(id, _)| *id).collect()
}

/// Write a self-contained pure-KNN project (vectors.jsonl + queries.jsonl +
/// neighbours.jsonl, cosine distance) and return its root. Ground truth is
/// whole-corpus cosine NN — no filters — so recall measures index quality only.
#[allow(clippy::too_many_arguments)]
fn create_knn_project(
    dataset_name: &str,
    engine_configs_json: &str,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    neighbors: &[Vec<i64>],
    dim: usize,
) -> PathBuf {
    let tmp = tempfile::tempdir().expect("Failed to create temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp); // keep alive for the subprocess

    let dataset_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&dataset_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();

    let mut vecs_content = String::new();
    for v in vectors {
        let line: Vec<f64> = v.iter().map(|x| *x as f64).collect();
        vecs_content.push_str(&serde_json::to_string(&line).unwrap());
        vecs_content.push('\n');
    }
    fs::write(dataset_dir.join("vectors.jsonl"), &vecs_content).unwrap();

    let mut queries_content = String::new();
    for q in queries {
        let line: Vec<f64> = q.iter().map(|x| *x as f64).collect();
        queries_content.push_str(&serde_json::to_string(&line).unwrap());
        queries_content.push('\n');
    }
    fs::write(dataset_dir.join("queries.jsonl"), &queries_content).unwrap();

    let mut neighbors_content = String::new();
    for n in neighbors {
        neighbors_content.push_str(&serde_json::to_string(n).unwrap());
        neighbors_content.push('\n');
    }
    fs::write(dataset_dir.join("neighbours.jsonl"), &neighbors_content).unwrap();

    let datasets_json = serde_json::json!([{
        "name": dataset_name,
        "type": "jsonl",
        "path": format!("{}/", dataset_name),
        "distance": "cosine",
        "vector_size": dim,
        "vector_count": vectors.len(),
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

    root
}

/// Deterministic vectors/queries + cosine ground truth for a KNN run.
fn make_data(n_docs: usize, n_queries: usize, dim: usize, top: usize) -> KnnData {
    let mut rng = StdRng::seed_from_u64(0xD6A6);
    let gen =
        |rng: &mut StdRng| -> Vec<f32> { (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect() };
    let vectors: Vec<Vec<f32>> = (0..n_docs).map(|_| gen(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..n_queries).map(|_| gen(&mut rng)).collect();
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_cosine_neighbors(q, &vectors, top))
        .collect();
    (vectors, queries, neighbors)
}

fn run_knn_recall_test(engine_name: &str, dataset_name: &str, parallel: u64) -> f64 {
    wait_for_dragonfly();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let n_docs = 2000;
    let n_queries = 100;
    let top = 10;

    let (vectors, queries, neighbors) = make_data(n_docs, n_queries, dim, top);

    let engine_config = serde_json::json!([{
        "name": engine_name,
        "engine": "dragonfly",
        "connection_params": {},
        "collection_params": {
            "hnsw_config": { "M": 16, "EF_CONSTRUCTION": 128 }
        },
        "search_params": [
            { "parallel": parallel, "top": top, "search_params": { "ef": 256 } }
        ],
        "upload_params": { "parallel": if parallel > 1 { parallel } else { 1 }, "batch_size": 64 }
    }]);

    let root = create_knn_project(
        dataset_name,
        &serde_json::to_string_pretty(&engine_config).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        dim,
    );

    let port = test_port().to_string();
    let ok = common::run_binary(
        &root,
        engine_name,
        dataset_name,
        TEST_HOST,
        &[("DRAGONFLY_PORT", port.as_str())],
    );
    assert!(ok, "benchmark binary run failed for {}", engine_name);

    let recall = common::read_recall(&root, engine_name);
    println!(
        "dragonfly KNN recall (parallel={}) = {:.3}",
        parallel, recall
    );
    recall
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// End-to-end KNN recall via the binary at parallel=1. Asserts recall >= 0.9
/// against whole-corpus cosine ground truth (proves upload + FT.SEARCH KNN work).
#[test]
fn test_dragonfly_knn_recall() {
    let recall = run_knn_recall_test("dragonfly-knn-p1", "dragonfly-knn-p1", 1);
    assert!(recall >= 0.9, "dragonfly KNN recall {:.3} < 0.9", recall);
}

/// Same KNN recall check at parallel=4 — exercises the multi-worker search path
/// (per-thread sample buffers merged on join) and concurrent connections.
#[test]
fn test_dragonfly_knn_recall_parallel() {
    let recall = run_knn_recall_test("dragonfly-knn-p4", "dragonfly-knn-p4", 4);
    assert!(
        recall >= 0.9,
        "dragonfly KNN recall (parallel=4) {:.3} < 0.9",
        recall
    );
}

/// Direct FT.CREATE + HSET + FT.SEARCH sanity check: the query vector must be
/// its own nearest neighbour. Guards the FLOAT32 encoding + KNN wiring without
/// going through the binary.
#[test]
fn test_dragonfly_knn_self_neighbor() {
    wait_for_dragonfly();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let count = 200;
    let (vectors, _q, _n) = make_data(count, 1, dim, 1);

    let _: () = redis::cmd("FT.CREATE")
        .arg("idx")
        .arg("ON")
        .arg("HASH")
        .arg("PREFIX")
        .arg("1")
        .arg("")
        .arg("SCHEMA")
        .arg("vector")
        .arg("VECTOR")
        .arg("HNSW")
        .arg(10)
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dim)
        .arg("DISTANCE_METRIC")
        .arg("COSINE")
        .arg("M")
        .arg(16)
        .arg("EF_CONSTRUCTION")
        .arg(128)
        .query(&mut conn)
        .expect("FT.CREATE");

    for (i, v) in vectors.iter().enumerate() {
        let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        let _: () = redis::cmd("HSET")
            .arg(i.to_string())
            .arg("vector")
            .arg(&bytes[..])
            .query(&mut conn)
            .expect("HSET");
    }

    let query_bytes: Vec<u8> = vectors[0].iter().flat_map(|f| f.to_le_bytes()).collect();
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx")
        .arg("*=>[KNN 5 @vector $vec_param EF_RUNTIME $EF AS vector_score]")
        .arg("SORTBY")
        .arg("vector_score")
        .arg("ASC")
        .arg("LIMIT")
        .arg(0)
        .arg(5)
        .arg("PARAMS")
        .arg(4)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("EF")
        .arg("64")
        .arg("DIALECT")
        .arg(2)
        .query(&mut conn)
        .expect("FT.SEARCH");

    assert!(!response.is_empty(), "expected search results");
    let top_id = match &response[1] {
        redis::Value::BulkString(s) => String::from_utf8_lossy(s).to_string(),
        redis::Value::SimpleString(s) => s.clone(),
        other => panic!("unexpected id value: {:?}", other),
    };
    assert_eq!(
        top_id, "0",
        "query vector should be its own nearest neighbor"
    );

    let _ = redis::cmd("FT.DROPINDEX").arg("idx").query::<()>(&mut conn);
}
