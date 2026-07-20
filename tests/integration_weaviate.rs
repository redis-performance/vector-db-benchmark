//! Integration tests for the Weaviate engine.
//!
//! Requires Weaviate 1.28.9 running on port 8081.
//! Start with: docker compose -f tests/docker-compose.test.yml up -d weaviate --wait
//! Run with:   WEAVIATE_HTTP_PORT=8081 cargo test --test integration_weaviate --release -- --test-threads=1

use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const WEAVIATE_PORT: u16 = 8081;
const WEAVIATE_HOST: &str = "127.0.0.1";
const TEST_CLASS: &str = "BenchTest";

fn weaviate_base_url() -> String {
    let port: u16 = std::env::var("WEAVIATE_HTTP_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(WEAVIATE_PORT);
    format!("http://{}:{}", WEAVIATE_HOST, port)
}

fn http_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client")
}

fn wait_for_weaviate() {
    let client = http_client();
    let url = format!("{}/v1/.well-known/ready", weaviate_base_url());
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Ok(resp) = client.get(&url).send() {
            if resp.status().is_success() {
                return;
            }
        }
        if Instant::now() > deadline {
            panic!("Weaviate not available on port {} after 60s", WEAVIATE_PORT);
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn delete_test_class() {
    let client = http_client();
    let url = format!("{}/v1/schema/{}", weaviate_base_url(), TEST_CLASS);
    let _ = client.delete(&url).send();
}

fn id_from_int(id: i64) -> String {
    uuid::Uuid::from_u128(id as u128).to_string()
}

fn generate_test_vectors(count: usize, dim: usize) -> (Vec<i64>, Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    let ids: Vec<i64> = (0..count as i64).collect();
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    (ids, vectors)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_weaviate_class_management() {
    wait_for_weaviate();
    delete_test_class();

    let client = http_client();

    // Create class with HNSW config
    let class_body = serde_json::json!({
        "class": TEST_CLASS,
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "efConstruction": 128,
            "maxConnections": 16,
            "distance": "l2-squared",
        },
        "properties": [
            {"name": "idx", "dataType": ["int"]},
        ]
    });

    let resp = client
        .post(format!("{}/v1/schema", weaviate_base_url()))
        .json(&class_body)
        .send()
        .expect("Failed to create class");
    assert!(
        resp.status().is_success(),
        "Failed to create class: {}",
        resp.text().unwrap_or_default()
    );

    // Verify class exists
    let resp = client
        .get(format!("{}/v1/schema/{}", weaviate_base_url(), TEST_CLASS))
        .send()
        .unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();
    assert_eq!(body["class"], TEST_CLASS);

    // Delete class
    let resp = client
        .delete(format!("{}/v1/schema/{}", weaviate_base_url(), TEST_CLASS))
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Verify deleted
    let resp = client
        .get(format!("{}/v1/schema/{}", weaviate_base_url(), TEST_CLASS))
        .send()
        .unwrap();
    assert_eq!(resp.status().as_u16(), 404);
}

#[test]
fn test_weaviate_batch_upload() {
    wait_for_weaviate();
    delete_test_class();

    let client = http_client();
    let dim = 4;

    // Create class
    let class_body = serde_json::json!({
        "class": TEST_CLASS,
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {"distance": "l2-squared"},
    });
    let resp = client
        .post(format!("{}/v1/schema", weaviate_base_url()))
        .json(&class_body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Batch upload
    let (ids, vectors) = generate_test_vectors(30, dim);
    let objects: Vec<serde_json::Value> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(&id, vec)| {
            serde_json::json!({
                "class": TEST_CLASS,
                "id": id_from_int(id),
                "vector": vec,
            })
        })
        .collect();

    let batch_body = serde_json::json!({"objects": objects});
    let resp = client
        .post(format!("{}/v1/batch/objects", weaviate_base_url()))
        .json(&batch_body)
        .send()
        .expect("Failed batch upload");
    assert!(
        resp.status().is_success(),
        "Batch upload failed: {}",
        resp.text().unwrap_or_default()
    );

    // Verify count via GraphQL aggregate
    let gql_body = serde_json::json!({
        "query": format!("{{ Aggregate {{ {} {{ meta {{ count }} }} }} }}", TEST_CLASS),
    });
    let resp = client
        .post(format!("{}/v1/graphql", weaviate_base_url()))
        .json(&gql_body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();
    let count = body["data"]["Aggregate"][TEST_CLASS][0]["meta"]["count"]
        .as_i64()
        .unwrap_or(0);
    assert_eq!(count, 30);

    delete_test_class();
}

#[test]
fn test_weaviate_near_vector_search() {
    wait_for_weaviate();
    delete_test_class();

    let client = http_client();
    let _dim = 4;

    // Create class with cosine distance
    let class_body = serde_json::json!({
        "class": TEST_CLASS,
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {"distance": "cosine"},
    });
    let resp = client
        .post(format!("{}/v1/schema", weaviate_base_url()))
        .json(&class_body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Upload known vectors
    let vectors: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.9, 0.1, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    let objects: Vec<serde_json::Value> = vectors
        .iter()
        .enumerate()
        .map(|(i, vec)| {
            serde_json::json!({
                "class": TEST_CLASS,
                "id": id_from_int(i as i64),
                "vector": vec,
            })
        })
        .collect();

    let batch_body = serde_json::json!({"objects": objects});
    let resp = client
        .post(format!("{}/v1/batch/objects", weaviate_base_url()))
        .json(&batch_body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Search for vector closest to [1, 0, 0, 0]
    let gql = format!(
        r#"{{ Get {{ {class}(nearVector: {{vector: [1.0, 0.0, 0.0, 0.0]}}, limit: 3) {{ _additional {{ id distance }} }} }} }}"#,
        class = TEST_CLASS
    );
    let resp = client
        .post(format!("{}/v1/graphql", weaviate_base_url()))
        .json(&serde_json::json!({"query": gql}))
        .send()
        .unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();

    let results = body["data"]["Get"][TEST_CLASS].as_array().unwrap();
    assert!(!results.is_empty(), "Expected search results");

    // First result should be id=0 (exact match, distance=0)
    let first_id = results[0]["_additional"]["id"].as_str().unwrap();
    let expected_id = id_from_int(0);
    assert_eq!(first_id, expected_id, "First result should be exact match");

    delete_test_class();
}

#[test]
fn test_weaviate_precision() {
    wait_for_weaviate();
    delete_test_class();

    let client = http_client();
    let dim = 8;
    let n = 200;
    let k = 10;

    // Create class
    let class_body = serde_json::json!({
        "class": TEST_CLASS,
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {
            "distance": "l2-squared",
            "efConstruction": 200,
            "maxConnections": 32,
            "ef": 256,
        },
    });
    let resp = client
        .post(format!("{}/v1/schema", weaviate_base_url()))
        .json(&class_body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Upload random vectors
    let (ids, vectors) = generate_test_vectors(n, dim);
    let objects: Vec<serde_json::Value> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(&id, vec)| {
            serde_json::json!({
                "class": TEST_CLASS,
                "id": id_from_int(id),
                "vector": vec,
            })
        })
        .collect();

    let batch_body = serde_json::json!({"objects": objects});
    let resp = client
        .post(format!("{}/v1/batch/objects", weaviate_base_url()))
        .json(&batch_body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Compute ground truth: brute-force L2 distances for query = vectors[0]
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

    // Near vector search via GraphQL
    let vec_str: Vec<String> = query.iter().map(|f| format!("{}", f)).collect();
    let gql = format!(
        r#"{{ Get {{ {class}(nearVector: {{vector: [{vec}]}}, limit: {k}) {{ _additional {{ id }} }} }} }}"#,
        class = TEST_CLASS,
        vec = vec_str.join(", "),
        k = k
    );
    let resp = client
        .post(format!("{}/v1/graphql", weaviate_base_url()))
        .json(&serde_json::json!({"query": gql}))
        .send()
        .unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();

    let results = body["data"]["Get"][TEST_CLASS].as_array().unwrap();

    let found: std::collections::HashSet<i64> = results
        .iter()
        .filter_map(|r| {
            let id_str = r["_additional"]["id"].as_str()?;
            let uuid = uuid::Uuid::parse_str(id_str).ok()?;
            Some(uuid.as_u128() as i64)
        })
        .collect();

    let overlap = ground_truth.intersection(&found).count();
    let precision = overlap as f64 / k as f64;
    println!(
        "Weaviate L2 precision@{}: {:.2} ({}/{})",
        k, precision, overlap, k
    );
    assert!(
        precision >= 0.8,
        "Expected precision >= 0.80, got {:.2}",
        precision
    );

    delete_test_class();
}

#[test]
fn test_weaviate_full_cycle() {
    wait_for_weaviate();
    delete_test_class();

    let client = http_client();
    let dim = 4;

    // Create
    let class_body = serde_json::json!({
        "class": TEST_CLASS,
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {"distance": "l2-squared"},
    });
    let resp = client
        .post(format!("{}/v1/schema", weaviate_base_url()))
        .json(&class_body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Upload
    let (ids, vectors) = generate_test_vectors(20, dim);
    let objects: Vec<serde_json::Value> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(&id, vec)| {
            serde_json::json!({
                "class": TEST_CLASS,
                "id": id_from_int(id),
                "vector": vec,
            })
        })
        .collect();
    let resp = client
        .post(format!("{}/v1/batch/objects", weaviate_base_url()))
        .json(&serde_json::json!({"objects": objects}))
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Count
    let gql = format!(
        "{{ Aggregate {{ {} {{ meta {{ count }} }} }} }}",
        TEST_CLASS
    );
    let resp = client
        .post(format!("{}/v1/graphql", weaviate_base_url()))
        .json(&serde_json::json!({"query": gql}))
        .send()
        .unwrap();
    let body: serde_json::Value = resp.json().unwrap();
    let count = body["data"]["Aggregate"][TEST_CLASS][0]["meta"]["count"]
        .as_i64()
        .unwrap_or(0);
    assert_eq!(count, 20);

    // Search
    let gql = format!(
        r#"{{ Get {{ {}(nearVector: {{vector: {:?}}}, limit: 5) {{ _additional {{ id }} }} }} }}"#,
        TEST_CLASS, vectors[0]
    );
    let resp = client
        .post(format!("{}/v1/graphql", weaviate_base_url()))
        .json(&serde_json::json!({"query": gql}))
        .send()
        .unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();
    let results = body["data"]["Get"][TEST_CLASS].as_array().unwrap();
    assert_eq!(results.len(), 5);

    // Delete
    delete_test_class();
    let resp = client
        .get(format!("{}/v1/schema/{}", weaviate_base_url(), TEST_CLASS))
        .send()
        .unwrap();
    assert_eq!(resp.status().as_u16(), 404);
}

/// Run a filtered `match_any` benchmark once, over the given transport, and
/// return `(transport_log_line, mean_recall)`. `use_graphql` forces the GraphQL
/// fallback via `WEAVIATE_USE_GRAPHQL`; otherwise the run uses gRPC with the
/// filter translated into the gRPC `Filters` message. A fresh project (its own
/// results dir) and class name per call keep the two runs independent.
fn run_weaviate_match_any(use_graphql: bool) -> (String, f64) {
    run_weaviate_match_any_impl(use_graphql, false)
}

/// Shared runner. `labels=true` filters on a MULTI-VALUED keyword field
/// (`labels`, a `text[]` array property) instead of scalar `color`, exercising
/// per-element `ContainsAny`/`Equal` semantics over the array (issue #88).
fn run_weaviate_match_any_impl(use_graphql: bool, labels: bool) -> (String, f64) {
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "weaviate-ma", "engine": "weaviate",
        "connection_params": {},
        "search_params": [{"parallel": 1, "vectorIndexConfig": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let ds_name = match (labels, use_graphql) {
        (false, false) => "match-any-grpc",
        (false, true) => "match-any-graphql",
        (true, false) => "match-any-labels-grpc",
        (true, true) => "match-any-labels-graphql",
    };
    let cfg_json = serde_json::to_string(&configs).unwrap();
    let proj = if labels {
        common::write_match_any_labels_project(ds_name, &cfg_json, dim, common::GtMetric::L2)
    } else {
        common::write_match_any_project(ds_name, &cfg_json, dim)
    };
    assert!(
        proj.matching_docs >= proj.top,
        "fixture must have >= top matching docs (got {})",
        proj.matching_docs
    );

    let port = std::env::var("WEAVIATE_HTTP_PORT").unwrap_or_else(|_| WEAVIATE_PORT.to_string());
    let class_name = match (labels, use_graphql) {
        (false, false) => "BenchMatchanyGrpc",
        (false, true) => "BenchMatchanyGql",
        (true, false) => "BenchMatchanyLabelsGrpc",
        (true, true) => "BenchMatchanyLabelsGql",
    };
    // Run directly (not common::run_binary) so we always surface the engine's
    // stdout/stderr — the filtered search's per-query errors print to stderr and
    // would otherwise be hidden on a zero-exit run that produced no results.
    let mut cmd = std::process::Command::new(common::binary_path());
    cmd.args([
        "--engines",
        "weaviate-ma",
        "--datasets",
        ds_name,
        "--host",
        WEAVIATE_HOST,
        "--skip-if-exists",
        "false",
    ])
    .env("WEAVIATE_HTTP_PORT", &port)
    .env("WEAVIATE_CLASS_NAME", class_name)
    .current_dir(&proj.root);
    if use_graphql {
        cmd.env("WEAVIATE_USE_GRAPHQL", "1");
    }
    let out = cmd.output().expect("run vector-db-benchmark");
    let stdout = String::from_utf8_lossy(&out.stdout);
    println!(
        "weaviate ({}) stdout:\n{}\nweaviate stderr:\n{}",
        if use_graphql { "graphql" } else { "grpc" },
        stdout,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "weaviate match_any run failed");

    let transport = stdout
        .lines()
        .find(|l| l.contains("search transport:"))
        .unwrap_or("<no transport line>")
        .trim()
        .to_string();
    let recall = common::read_recall(&proj.root, "weaviate-ma");
    (transport, recall)
}

/// End-to-end `match_any` across BOTH transports: filter a keyword field to an
/// OR-set and assert the engine returns the filtered nearest neighbours (recall
/// vs ground truth brute-forced over only the matching docs). Runs once over
/// gRPC (filter translated into the gRPC `Filters` message) and once over the
/// GraphQL fallback, then asserts BOTH clear 0.9 recall AND that the two are
/// identical — proving the gRPC `Filters` translation matches GraphQL `where`
/// semantics exactly (#119).
#[test]
fn test_binary_weaviate_match_any() {
    wait_for_weaviate();

    let (grpc_transport, grpc_recall) = run_weaviate_match_any(false);
    let (gql_transport, gql_recall) = run_weaviate_match_any(true);

    println!(
        "weaviate match_any [{}] recall={:.4}",
        grpc_transport, grpc_recall
    );
    println!(
        "weaviate match_any [{}] recall={:.4}",
        gql_transport, gql_recall
    );

    assert!(
        grpc_transport.contains("grpc"),
        "default run should use gRPC, got: {}",
        grpc_transport
    );
    assert!(
        gql_transport.contains("graphql"),
        "WEAVIATE_USE_GRAPHQL run should use GraphQL, got: {}",
        gql_transport
    );

    assert!(
        grpc_recall >= 0.9,
        "weaviate match_any gRPC recall {:.4} < 0.9",
        grpc_recall
    );
    assert!(
        gql_recall >= 0.9,
        "weaviate match_any GraphQL recall {:.4} < 0.9",
        gql_recall
    );
    // The translated gRPC Filters must select the same result set as the GraphQL
    // where; identical mean recall across transports is the proof.
    assert!(
        (grpc_recall - gql_recall).abs() < 1e-6,
        "gRPC vs GraphQL recall differ: {:.6} vs {:.6}",
        grpc_recall,
        gql_recall
    );
}

/// Run a filter project on Weaviate over the default gRPC transport, returning
/// recall. Runs the binary directly so per-query stderr is surfaced.
fn run_weaviate_filter(root: &std::path::Path, ds_name: &str, class_name: &str) -> f64 {
    let port = std::env::var("WEAVIATE_HTTP_PORT").unwrap_or_else(|_| WEAVIATE_PORT.to_string());
    let out = std::process::Command::new(common::binary_path())
        .args([
            "--engines",
            "weaviate-f",
            "--datasets",
            ds_name,
            "--host",
            WEAVIATE_HOST,
            "--skip-if-exists",
            "false",
        ])
        .env("WEAVIATE_HTTP_PORT", &port)
        .env("WEAVIATE_CLASS_NAME", class_name)
        .current_dir(root)
        .output()
        .expect("run vector-db-benchmark");
    println!(
        "weaviate {} stdout:\n{}\nstderr:\n{}",
        ds_name,
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(out.status.success(), "weaviate {} run failed", ds_name);
    common::read_recall(root, "weaviate-f")
}

fn weaviate_filter_config() -> String {
    serde_json::json!([{
        "name": "weaviate-f", "engine": "weaviate",
        "connection_params": {},
        "search_params": [{"parallel": 1, "vectorIndexConfig": {"ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }])
    .to_string()
}

/// Bool-field equality filter end-to-end. Regression: the `bool` schema type was
/// dropped (no property created), so the filter matched nothing. Now `bool` ->
/// `boolean` property, the upload converts the reader's "true"/"false" string to
/// a native bool, and the gRPC `valueBoolean` filter selects the even ids.
#[test]
fn test_binary_weaviate_bool() {
    wait_for_weaviate();
    let proj = common::write_bool_project("bool-test", &weaviate_filter_config(), 8);
    assert!(proj.matching_docs >= proj.top);
    let recall = run_weaviate_filter(&proj.root, "bool-test", "BenchBool");
    println!("weaviate bool recall={:.3}", recall);
    assert!(recall >= 0.9, "weaviate bool recall {:.3} < 0.9", recall);
}

/// Datetime range filter end-to-end. Regression: `datetime` was dropped and the
/// range builder only emitted valueInt/valueNumber. Now `datetime` -> `date`
/// property and the ISO bound is sent as `valueText` (Weaviate compares dates
/// via RFC3339 text), selecting the [day 100, day 300) window.
#[test]
fn test_binary_weaviate_datetime() {
    wait_for_weaviate();
    let proj = common::write_datetime_project("dt-test", &weaviate_filter_config(), 8);
    assert!(proj.matching_docs >= proj.top);
    let recall = run_weaviate_filter(&proj.root, "dt-test", "BenchDt");
    println!("weaviate datetime recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "weaviate datetime recall {:.3} < 0.9",
        recall
    );
}

/// End-to-end `match_any` on a MULTI-VALUED keyword field (`labels`, #88).
/// `labels` is declared as a `text[]` array property and each doc stores an
/// array; the OR-of-`Equal` filter matches per element (Weaviate's `Equal` on
/// an array property is contains). Before the fix `labels` was a scalar `text`
/// property, which rejected the array insert / could not match one element.
/// Runs both transports and asserts each clears 0.9 and they agree.
#[test]
fn test_binary_weaviate_match_any_labels() {
    wait_for_weaviate();

    let (grpc_transport, grpc_recall) = run_weaviate_match_any_impl(false, true);
    let (_gql_transport, gql_recall) = run_weaviate_match_any_impl(true, true);

    println!(
        "weaviate labels match_any [{}] grpc_recall={:.4} gql_recall={:.4}",
        grpc_transport, grpc_recall, gql_recall
    );

    assert!(
        grpc_recall >= 0.9,
        "weaviate labels match_any gRPC recall {:.4} < 0.9",
        grpc_recall
    );
    assert!(
        gql_recall >= 0.9,
        "weaviate labels match_any GraphQL recall {:.4} < 0.9",
        gql_recall
    );
    assert!(
        (grpc_recall - gql_recall).abs() < 1e-6,
        "gRPC vs GraphQL labels recall differ: {:.6} vs {:.6}",
        grpc_recall,
        gql_recall
    );
}
