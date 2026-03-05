//! Integration tests for the Weaviate engine.
//!
//! Requires Weaviate 1.28.9 running on port 8081.
//! Start with: docker compose -f tests/docker-compose.test.yml up -d weaviate --wait
//! Run with:   WEAVIATE_HTTP_PORT=8081 cargo test --test integration_weaviate --release -- --test-threads=1

use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;

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
        .post(&format!("{}/v1/schema", weaviate_base_url()))
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
        .get(&format!("{}/v1/schema/{}", weaviate_base_url(), TEST_CLASS))
        .send()
        .unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();
    assert_eq!(body["class"], TEST_CLASS);

    // Delete class
    let resp = client
        .delete(&format!("{}/v1/schema/{}", weaviate_base_url(), TEST_CLASS))
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Verify deleted
    let resp = client
        .get(&format!("{}/v1/schema/{}", weaviate_base_url(), TEST_CLASS))
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
        .post(&format!("{}/v1/schema", weaviate_base_url()))
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
        .post(&format!("{}/v1/batch/objects", weaviate_base_url()))
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
        .post(&format!("{}/v1/graphql", weaviate_base_url()))
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
    let dim = 4;

    // Create class with cosine distance
    let class_body = serde_json::json!({
        "class": TEST_CLASS,
        "vectorizer": "none",
        "vectorIndexType": "hnsw",
        "vectorIndexConfig": {"distance": "cosine"},
    });
    let resp = client
        .post(&format!("{}/v1/schema", weaviate_base_url()))
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
        .post(&format!("{}/v1/batch/objects", weaviate_base_url()))
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
        .post(&format!("{}/v1/graphql", weaviate_base_url()))
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
        .post(&format!("{}/v1/schema", weaviate_base_url()))
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
        .post(&format!("{}/v1/batch/objects", weaviate_base_url()))
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
        .post(&format!("{}/v1/graphql", weaviate_base_url()))
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
        .post(&format!("{}/v1/schema", weaviate_base_url()))
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
        .post(&format!("{}/v1/batch/objects", weaviate_base_url()))
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
        .post(&format!("{}/v1/graphql", weaviate_base_url()))
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
        .post(&format!("{}/v1/graphql", weaviate_base_url()))
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
        .get(&format!("{}/v1/schema/{}", weaviate_base_url(), TEST_CLASS))
        .send()
        .unwrap();
    assert_eq!(resp.status().as_u16(), 404);
}
