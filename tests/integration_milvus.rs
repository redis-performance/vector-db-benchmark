//! Integration tests for the Milvus engine.
//!
//! Requires Milvus 2.5.6 running on port 19531 (standalone mode with etcd + minio).
//! Start with: docker compose -f tests/docker-compose.test.yml up -d milvus --wait
//! Run with:   MILVUS_PORT=19531 cargo test --test integration_milvus --release -- --test-threads=1

use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const MILVUS_PORT: u16 = 19531;
const MILVUS_HOST: &str = "127.0.0.1";
const TEST_COLLECTION: &str = "bench_test";

fn milvus_base_url() -> String {
    let port: u16 = std::env::var("MILVUS_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MILVUS_PORT);
    format!("http://{}:{}", MILVUS_HOST, port)
}

fn http_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client")
}

fn wait_for_milvus() {
    let client = http_client();
    let url = format!("{}/v2/vectordb/collections/list", milvus_base_url());
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if let Ok(resp) = client.post(&url).json(&serde_json::json!({})).send() {
            if resp.status().is_success() {
                let body: serde_json::Value = resp.json().unwrap_or_default();
                if body.get("code").and_then(|c| c.as_i64()).unwrap_or(-1) == 0 {
                    return;
                }
            }
        }
        if Instant::now() > deadline {
            panic!("Milvus not available on port {} after 120s", MILVUS_PORT);
        }
        thread::sleep(Duration::from_millis(1000));
    }
}

fn drop_test_collection() {
    let client = http_client();
    let url = format!("{}/v2/vectordb/collections/drop", milvus_base_url());
    let _ = client
        .post(&url)
        .json(&serde_json::json!({"collectionName": TEST_COLLECTION}))
        .send();
}

fn generate_test_vectors(count: usize, dim: usize) -> (Vec<i64>, Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    let ids: Vec<i64> = (0..count as i64).collect();
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    (ids, vectors)
}

fn create_collection(dim: usize, metric_type: &str) {
    let client = http_client();
    let body = serde_json::json!({
        "collectionName": TEST_COLLECTION,
        "schema": {
            "fields": [
                {
                    "fieldName": "id",
                    "dataType": "Int64",
                    "isPrimary": true,
                },
                {
                    "fieldName": "vector",
                    "dataType": "FloatVector",
                    "elementTypeParams": {
                        "dim": dim.to_string(),
                    }
                }
            ],
            "enableDynamicField": false,
        }
    });

    let url = format!("{}/v2/vectordb/collections/create", milvus_base_url());
    let resp = client.post(&url).json(&body).send().unwrap();
    assert!(
        resp.status().is_success(),
        "Failed to create collection: {}",
        resp.text().unwrap_or_default()
    );

    // Create index
    let index_body = serde_json::json!({
        "collectionName": TEST_COLLECTION,
        "indexParams": [{
            "fieldName": "vector",
            "indexName": "vector_index",
            "metricType": metric_type,
            "indexType": "HNSW",
            "params": {
                "M": 16,
                "efConstruction": 200,
            }
        }]
    });
    let url = format!("{}/v2/vectordb/indexes/create", milvus_base_url());
    let resp = client.post(&url).json(&index_body).send().unwrap();
    assert!(resp.status().is_success());

    // Load collection
    let load_body = serde_json::json!({"collectionName": TEST_COLLECTION});
    let url = format!("{}/v2/vectordb/collections/load", milvus_base_url());
    let _ = client.post(&url).json(&load_body).send();

    // Wait for load state
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let url = format!(
            "{}/v2/vectordb/collections/get_load_state",
            milvus_base_url()
        );
        if let Ok(resp) = client
            .post(&url)
            .json(&serde_json::json!({"collectionName": TEST_COLLECTION}))
            .send()
        {
            let body: serde_json::Value = resp.json().unwrap_or_default();
            if let Some(data) = body.get("data") {
                let state = data.get("loadState").and_then(|s| s.as_str()).unwrap_or("");
                if state == "LoadStateLoaded" {
                    break;
                }
            }
        }
        if Instant::now() > deadline {
            break;
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn insert_vectors(ids: &[i64], vectors: &[Vec<f32>]) {
    let client = http_client();

    let data: Vec<serde_json::Value> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(&id, vec)| {
            serde_json::json!({
                "id": id,
                "vector": vec,
            })
        })
        .collect();

    let body = serde_json::json!({
        "collectionName": TEST_COLLECTION,
        "data": data,
    });

    let url = format!("{}/v2/vectordb/entities/insert", milvus_base_url());
    let resp = client.post(&url).json(&body).send().unwrap();
    assert!(
        resp.status().is_success(),
        "Insert failed: {}",
        resp.text().unwrap_or_default()
    );
    let resp_body: serde_json::Value = resp.json().unwrap();
    let code = resp_body.get("code").and_then(|c| c.as_i64()).unwrap_or(-1);
    assert_eq!(code, 0, "Insert error: {:?}", resp_body);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_milvus_collection_crud() {
    wait_for_milvus();
    drop_test_collection();

    let client = http_client();

    // Create
    create_collection(4, "L2");

    // Verify exists
    let url = format!("{}/v2/vectordb/collections/has", milvus_base_url());
    let resp = client
        .post(&url)
        .json(&serde_json::json!({"collectionName": TEST_COLLECTION}))
        .send()
        .unwrap();
    let body: serde_json::Value = resp.json().unwrap();
    assert_eq!(
        body.get("data")
            .and_then(|d| d.get("has"))
            .and_then(|h| h.as_bool()),
        Some(true)
    );

    // Drop
    drop_test_collection();

    // Verify gone
    let resp = client
        .post(&url)
        .json(&serde_json::json!({"collectionName": TEST_COLLECTION}))
        .send()
        .unwrap();
    let body: serde_json::Value = resp.json().unwrap();
    assert_eq!(
        body.get("data")
            .and_then(|d| d.get("has"))
            .and_then(|h| h.as_bool()),
        Some(false)
    );
}

#[test]
fn test_milvus_insert_and_search() {
    wait_for_milvus();
    drop_test_collection();
    create_collection(4, "L2");

    // Insert known vectors
    let ids = vec![0i64, 1, 2, 3, 4];
    let vectors: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.9, 0.1, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    insert_vectors(&ids, &vectors);

    // Wait for flush
    thread::sleep(Duration::from_secs(1));

    // Search for [1, 0, 0, 0]
    let client = http_client();
    let search_body = serde_json::json!({
        "collectionName": TEST_COLLECTION,
        "data": [[1.0, 0.0, 0.0, 0.0]],
        "limit": 3,
        "outputFields": ["id"],
        "annsField": "vector",
    });
    let url = format!("{}/v2/vectordb/entities/search", milvus_base_url());
    let resp = client.post(&url).json(&search_body).send().unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();

    let code = body.get("code").and_then(|c| c.as_i64()).unwrap_or(-1);
    assert_eq!(code, 0, "Search error: {:?}", body);

    let data = body.get("data").and_then(|d| d.as_array()).unwrap();
    assert!(!data.is_empty(), "Expected search results");

    // First result should be id=0 (L2 distance = 0)
    let first_id = data[0].get("id").and_then(|v| v.as_i64()).unwrap();
    assert_eq!(first_id, 0, "First result should be exact match");

    drop_test_collection();
}

#[test]
fn test_milvus_precision_l2() {
    wait_for_milvus();
    drop_test_collection();

    let dim = 8;
    let n = 200;
    let k = 10;

    create_collection(dim, "L2");

    let (ids, vectors) = generate_test_vectors(n, dim);

    // Insert in batches to avoid too large payloads
    for chunk_start in (0..n).step_by(100) {
        let chunk_end = (chunk_start + 100).min(n);
        insert_vectors(
            &ids[chunk_start..chunk_end],
            &vectors[chunk_start..chunk_end],
        );
    }

    // Wait for flush
    thread::sleep(Duration::from_secs(2));

    // Compute brute-force ground truth
    let query = &vectors[0];
    let mut distances: Vec<(i64, f64)> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(&id, v)| {
            let dist: f64 = query
                .iter()
                .zip(v.iter())
                .map(|(a, b)| ((*a as f64) - (*b as f64)).powi(2))
                .sum();
            (id, dist)
        })
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let ground_truth: std::collections::HashSet<i64> =
        distances.iter().take(k).map(|(id, _)| *id).collect();

    // Search
    let client = http_client();
    let search_body = serde_json::json!({
        "collectionName": TEST_COLLECTION,
        "data": [query],
        "limit": k,
        "outputFields": ["id"],
        "annsField": "vector",
        "searchParams": {
            "params": {"ef": 256},
        }
    });
    let url = format!("{}/v2/vectordb/entities/search", milvus_base_url());
    let resp = client.post(&url).json(&search_body).send().unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();
    assert_eq!(body.get("code").and_then(|c| c.as_i64()), Some(0));

    let data = body.get("data").and_then(|d| d.as_array()).unwrap();
    let found: std::collections::HashSet<i64> = data
        .iter()
        .filter_map(|item| item.get("id").and_then(|v| v.as_i64()))
        .collect();

    let overlap = ground_truth.intersection(&found).count();
    let precision = overlap as f64 / k as f64;
    println!(
        "Milvus L2 precision@{}: {:.2} ({}/{})",
        k, precision, overlap, k
    );
    assert!(
        precision >= 0.8,
        "Expected precision >= 0.80, got {:.2}",
        precision
    );

    drop_test_collection();
}

#[test]
fn test_milvus_full_cycle() {
    wait_for_milvus();
    drop_test_collection();

    let dim = 4;

    // Create + index + load
    create_collection(dim, "L2");

    // Insert
    let (ids, vectors) = generate_test_vectors(20, dim);
    insert_vectors(&ids, &vectors);
    thread::sleep(Duration::from_secs(1));

    // Search
    let client = http_client();
    let search_body = serde_json::json!({
        "collectionName": TEST_COLLECTION,
        "data": [vectors[0]],
        "limit": 5,
        "outputFields": ["id"],
        "annsField": "vector",
    });
    let url = format!("{}/v2/vectordb/entities/search", milvus_base_url());
    let resp = client.post(&url).json(&search_body).send().unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();
    assert_eq!(body.get("code").and_then(|c| c.as_i64()), Some(0));
    let data = body.get("data").and_then(|d| d.as_array()).unwrap();
    assert_eq!(data.len(), 5);

    // Delete collection
    drop_test_collection();

    // Verify gone
    let url = format!("{}/v2/vectordb/collections/has", milvus_base_url());
    let resp = client
        .post(&url)
        .json(&serde_json::json!({"collectionName": TEST_COLLECTION}))
        .send()
        .unwrap();
    let body: serde_json::Value = resp.json().unwrap();
    assert_eq!(
        body.get("data")
            .and_then(|d| d.get("has"))
            .and_then(|h| h.as_bool()),
        Some(false)
    );
}

/// End-to-end `match_any`: filter a keyword field to an OR-set and assert the
/// engine returns the filtered nearest neighbours (recall vs ground truth
/// brute-forced over only the matching docs). Proves the `in [...]` expr arm.
#[test]
fn test_binary_milvus_match_any() {
    wait_for_milvus();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "milvus-ma", "engine": "milvus",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100, "index_params": {"M": 16, "efConstruction": 200}}
    }]);
    let proj = common::write_match_any_project(
        "match-any-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
    );
    assert!(
        proj.matching_docs >= proj.top,
        "fixture must have >= top matching docs (got {})",
        proj.matching_docs
    );

    assert!(
        common::run_binary(
            &proj.root,
            "milvus-ma",
            "match-any-test",
            "127.0.0.1",
            &[
                ("MILVUS_PORT", "19531"),
                ("MILVUS_COLLECTION_NAME", "bench_matchany"),
            ],
        ),
        "milvus match_any run failed"
    );

    let recall = common::read_recall(&proj.root, "milvus-ma");
    println!("milvus match_any recall={:.3}", recall);
    assert!(recall >= 0.9, "milvus match_any recall {:.3} < 0.9", recall);
}

/// Bool-field equality filter end-to-end. Regression: `"bool"` hit the schema
/// `_ => continue` arm so no column was created, while the filter emitted a
/// native `flag == true` against the missing column. Now `bool` -> native Bool
/// column (upload converts the reader's "true"/"false" string to a JSON bool).
#[test]
fn test_binary_milvus_bool() {
    wait_for_milvus();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "milvus-bool", "engine": "milvus",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100, "index_params": {"M": 16, "efConstruction": 200}}
    }]);
    let proj =
        common::write_bool_project("bool-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "milvus-bool",
            "bool-test",
            "127.0.0.1",
            &[
                ("MILVUS_PORT", "19531"),
                ("MILVUS_COLLECTION_NAME", "bench_bool")
            ],
        ),
        "milvus bool run failed"
    );
    let recall = common::read_recall(&proj.root, "milvus-bool");
    println!("milvus bool recall={:.3}", recall);
    assert!(recall >= 0.9, "milvus bool recall {:.3} < 0.9", recall);
}

/// Datetime range filter end-to-end. Regression: `"datetime"` was dropped from
/// the schema and the range builder inlined the quoted ISO string. Now
/// `datetime` -> Int64 epoch column; upload and the range filter both convert
/// ISO-8601 to epoch seconds, so the `[day 100, day 300)` window is selected.
#[test]
fn test_binary_milvus_datetime() {
    wait_for_milvus();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "milvus-dt", "engine": "milvus",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100, "index_params": {"M": 16, "efConstruction": 200}}
    }]);
    let proj =
        common::write_datetime_project("dt-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "milvus-dt",
            "dt-test",
            "127.0.0.1",
            &[
                ("MILVUS_PORT", "19531"),
                ("MILVUS_COLLECTION_NAME", "bench_dt")
            ],
        ),
        "milvus datetime run failed"
    );
    let recall = common::read_recall(&proj.root, "milvus-dt");
    println!("milvus datetime recall={:.3}", recall);
    assert!(recall >= 0.9, "milvus datetime recall {:.3} < 0.9", recall);
}

/// Full-text filter end-to-end. Regression: a `{match:{text}}` clause was dropped
/// (the match arm required `value`/`any`), so the search ran UNFILTERED. Now the
/// `text` VarChar column is created with enable_analyzer/enable_match and the
/// filter uses `TEXT_MATCH(body, 'quick')`, selecting docs containing the token.
#[test]
fn test_binary_milvus_fulltext() {
    wait_for_milvus();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "milvus-ft", "engine": "milvus",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100, "index_params": {"M": 16, "efConstruction": 200}}
    }]);
    let proj =
        common::write_fulltext_project("ft-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "milvus-ft",
            "ft-test",
            "127.0.0.1",
            &[
                ("MILVUS_PORT", "19531"),
                ("MILVUS_COLLECTION_NAME", "bench_ft")
            ],
        ),
        "milvus fulltext run failed"
    );
    let recall = common::read_recall(&proj.root, "milvus-ft");
    println!("milvus fulltext recall={:.3}", recall);
    assert!(recall >= 0.9, "milvus fulltext recall {:.3} < 0.9", recall);
}

/// End-to-end `match_any` on a MULTI-VALUED keyword field (`labels`, #88).
/// Milvus stores it as an Array(VarChar) and filters with
/// `array_contains_any`; before the fix it was a comma-joined VarChar tested
/// with whole-string `in`, which cannot match a single element (recall ~0).
#[test]
fn test_binary_milvus_match_any_labels() {
    wait_for_milvus();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "milvus-mal", "engine": "milvus",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100, "index_params": {"M": 16, "efConstruction": 200}}
    }]);
    let proj = common::write_match_any_labels_project(
        "match-any-labels-test",
        &serde_json::to_string(&configs).unwrap(),
        dim,
        common::GtMetric::L2,
    );
    assert!(
        proj.matching_docs >= proj.top,
        "fixture must have >= top matching docs (got {})",
        proj.matching_docs
    );

    assert!(
        common::run_binary(
            &proj.root,
            "milvus-mal",
            "match-any-labels-test",
            "127.0.0.1",
            &[
                ("MILVUS_PORT", "19531"),
                ("MILVUS_COLLECTION_NAME", "bench_matchany_labels"),
            ],
        ),
        "milvus match_any labels run failed"
    );

    let recall = common::read_recall(&proj.root, "milvus-mal");
    println!("milvus match_any labels recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "milvus multi-valued labels match_any recall {:.3} < 0.9",
        recall
    );
}
