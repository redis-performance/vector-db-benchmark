//! Integration tests for the OpenSearch engine.
//!
//! Requires OpenSearch 2.19.2 running on port 9202 (security disabled).
//! Start with: docker compose -f tests/docker-compose.test.yml up -d opensearch --wait
//! Run with:   OPENSEARCH_PORT=9202 cargo test --test integration_opensearch --release -- --test-threads=1

use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const OS_PORT: u16 = 9202;
const OS_HOST: &str = "127.0.0.1";
const OS_INDEX: &str = "bench_test";

fn os_base_url() -> String {
    let port: u16 = std::env::var("OPENSEARCH_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(OS_PORT);
    format!("http://{}:{}", OS_HOST, port)
}

fn os_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .danger_accept_invalid_certs(true)
        .build()
        .expect("Failed to create HTTP client")
}

fn wait_for_opensearch() {
    let client = os_client();
    let url = format!("{}/_cluster/health", os_base_url());
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Ok(resp) = client.get(&url).send() {
            if resp.status().is_success() {
                return;
            }
        }
        if Instant::now() > deadline {
            panic!("OpenSearch not available on port {} after 60s", OS_PORT);
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn delete_test_index() {
    let client = os_client();
    let url = format!("{}/{}", os_base_url(), OS_INDEX);
    let _ = client.delete(&url).send();
}

fn generate_test_vectors(count: usize, dim: usize) -> (Vec<i64>, Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    let ids: Vec<i64> = (0..count as i64).collect();
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    (ids, vectors)
}

fn id_to_uuid_hex(id: i64) -> String {
    uuid::Uuid::from_u128(id as u128).as_simple().to_string()
}

fn get_index_doc_count() -> usize {
    let client = os_client();
    // Refresh first to make sure all docs are searchable
    let _ = client
        .post(&format!("{}/{}/_refresh", os_base_url(), OS_INDEX))
        .send();
    let url = format!("{}/{}/_count", os_base_url(), OS_INDEX);
    let resp = client.get(&url).send().expect("Failed to get doc count");
    let body: serde_json::Value = resp.json().expect("Failed to parse count response");
    body.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as usize
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_opensearch_create_knn_index() {
    wait_for_opensearch();
    delete_test_index();

    let client = os_client();
    let body = serde_json::json!({
        "settings": {
            "index": {
                "knn": true,
            }
        },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": 4,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "l2",
                        "parameters": {
                            "m": 16,
                            "ef_construction": 100,
                        }
                    }
                }
            }
        }
    });

    let url = format!("{}/{}", os_base_url(), OS_INDEX);
    let resp = client
        .put(&url)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .expect("Failed to create index");
    assert!(
        resp.status().is_success(),
        "Failed to create index: {}",
        resp.text().unwrap_or_default()
    );

    // Verify the index exists
    let resp = client
        .get(&format!("{}/{}", os_base_url(), OS_INDEX))
        .send()
        .expect("Failed to get index");
    assert!(resp.status().is_success());

    delete_test_index();
}

#[test]
fn test_opensearch_bulk_upload() {
    wait_for_opensearch();
    delete_test_index();

    let client = os_client();
    // Create index
    let body = serde_json::json!({
        "settings": {"index": {"knn": true}},
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": 4,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "l2",
                    }
                }
            }
        }
    });
    let resp = client
        .put(&format!("{}/{}", os_base_url(), OS_INDEX))
        .json(&body)
        .send()
        .expect("Failed to create index");
    assert!(resp.status().is_success());

    // Upload vectors
    let (ids, vectors) = generate_test_vectors(50, 4);
    let mut ndjson = String::new();
    for i in 0..ids.len() {
        let uuid_hex = id_to_uuid_hex(ids[i]);
        ndjson.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n", uuid_hex));
        let doc = serde_json::json!({"vector": vectors[i]});
        ndjson.push_str(&serde_json::to_string(&doc).unwrap());
        ndjson.push('\n');
    }

    let resp = client
        .post(&format!("{}/{}/_bulk", os_base_url(), OS_INDEX))
        .header("Content-Type", "application/x-ndjson")
        .body(ndjson)
        .send()
        .expect("Failed to bulk upload");
    assert!(resp.status().is_success());
    let resp_body: serde_json::Value = resp.json().unwrap();
    assert!(!resp_body["errors"].as_bool().unwrap_or(true));

    // Verify count
    assert_eq!(get_index_doc_count(), 50);

    delete_test_index();
}

#[test]
fn test_opensearch_knn_search() {
    wait_for_opensearch();
    delete_test_index();

    let client = os_client();
    // Create index with cosine similarity
    let body = serde_json::json!({
        "settings": {"index": {"knn": true}},
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": 4,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "cosinesimil",
                    }
                }
            }
        }
    });
    let resp = client
        .put(&format!("{}/{}", os_base_url(), OS_INDEX))
        .json(&body)
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

    let mut ndjson = String::new();
    for (i, vec) in vectors.iter().enumerate() {
        let uuid_hex = id_to_uuid_hex(i as i64);
        ndjson.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n", uuid_hex));
        let doc = serde_json::json!({"vector": vec});
        ndjson.push_str(&serde_json::to_string(&doc).unwrap());
        ndjson.push('\n');
    }

    let resp = client
        .post(&format!("{}/{}/_bulk", os_base_url(), OS_INDEX))
        .header("Content-Type", "application/x-ndjson")
        .body(ndjson)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Refresh
    let _ = client
        .post(&format!("{}/{}/_refresh", os_base_url(), OS_INDEX))
        .send();

    // Search for vector closest to [1, 0, 0, 0]
    let search_body = serde_json::json!({
        "query": {
            "knn": {
                "vector": {
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "k": 3,
                }
            }
        },
        "size": 3,
    });

    let resp = client
        .post(&format!("{}/{}/_search", os_base_url(), OS_INDEX))
        .json(&search_body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();

    let hits = body["hits"]["hits"].as_array().unwrap();
    assert!(!hits.is_empty(), "Expected search results");

    // First result should be id=0 (exact match)
    let first_id = hits[0]["_id"].as_str().unwrap();
    let expected_id = id_to_uuid_hex(0);
    assert_eq!(
        first_id, expected_id,
        "First result should be the exact match vector"
    );

    delete_test_index();
}

#[test]
fn test_opensearch_precision_l2() {
    wait_for_opensearch();
    delete_test_index();

    let client = os_client();
    let dim = 8;
    let n = 200;
    let k = 10;

    // Create index
    let body = serde_json::json!({
        "settings": {"index": {"knn": true}},
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "l2",
                        "parameters": {"m": 32, "ef_construction": 200}
                    }
                }
            }
        }
    });
    let resp = client
        .put(&format!("{}/{}", os_base_url(), OS_INDEX))
        .json(&body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Upload random vectors
    let (ids, vectors) = generate_test_vectors(n, dim);
    let mut ndjson = String::new();
    for i in 0..n {
        let uuid_hex = id_to_uuid_hex(ids[i]);
        ndjson.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n", uuid_hex));
        let doc = serde_json::json!({"vector": vectors[i]});
        ndjson.push_str(&serde_json::to_string(&doc).unwrap());
        ndjson.push('\n');
    }
    let resp = client
        .post(&format!("{}/{}/_bulk", os_base_url(), OS_INDEX))
        .header("Content-Type", "application/x-ndjson")
        .body(ndjson)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Refresh
    let _ = client
        .post(&format!("{}/{}/_refresh", os_base_url(), OS_INDEX))
        .send();

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

    // KNN search
    let search_body = serde_json::json!({
        "query": {
            "knn": {
                "vector": {
                    "vector": query,
                    "k": k,
                }
            }
        },
        "size": k,
    });
    let resp = client
        .post(&format!("{}/{}/_search", os_base_url(), OS_INDEX))
        .json(&search_body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();
    let hits = body["hits"]["hits"].as_array().unwrap();

    let found: std::collections::HashSet<i64> = hits
        .iter()
        .filter_map(|h| {
            let hex = h["_id"].as_str()?;
            let uuid = uuid::Uuid::parse_str(hex).ok()?;
            Some(uuid.as_u128() as i64)
        })
        .collect();

    let overlap = ground_truth.intersection(&found).count();
    let precision = overlap as f64 / k as f64;
    println!(
        "OpenSearch L2 precision@{}: {:.2} ({}/{})",
        k, precision, overlap, k
    );
    assert!(
        precision >= 0.8,
        "Expected precision >= 0.80, got {:.2}",
        precision
    );

    delete_test_index();
}

#[test]
fn test_opensearch_full_cycle() {
    wait_for_opensearch();
    delete_test_index();

    let client = os_client();
    let dim = 4;

    // Create
    let body = serde_json::json!({
        "settings": {"index": {"knn": true}},
        "mappings": {
            "properties": {
                "vector": {
                    "type": "knn_vector",
                    "dimension": dim,
                    "method": {
                        "name": "hnsw",
                        "engine": "lucene",
                        "space_type": "l2",
                    }
                }
            }
        }
    });
    let resp = client
        .put(&format!("{}/{}", os_base_url(), OS_INDEX))
        .json(&body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());

    // Upload
    let (ids, vectors) = generate_test_vectors(20, dim);
    let mut ndjson = String::new();
    for i in 0..ids.len() {
        let uuid_hex = id_to_uuid_hex(ids[i]);
        ndjson.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n", uuid_hex));
        let doc = serde_json::json!({"vector": vectors[i]});
        ndjson.push_str(&serde_json::to_string(&doc).unwrap());
        ndjson.push('\n');
    }
    let resp = client
        .post(&format!("{}/{}/_bulk", os_base_url(), OS_INDEX))
        .header("Content-Type", "application/x-ndjson")
        .body(ndjson)
        .send()
        .unwrap();
    assert!(resp.status().is_success());
    assert_eq!(get_index_doc_count(), 20);

    // Search
    let _ = client
        .post(&format!("{}/{}/_refresh", os_base_url(), OS_INDEX))
        .send();
    let search_body = serde_json::json!({
        "query": {
            "knn": {
                "vector": {
                    "vector": vectors[0],
                    "k": 5,
                }
            }
        },
        "size": 5,
    });
    let resp = client
        .post(&format!("{}/{}/_search", os_base_url(), OS_INDEX))
        .json(&search_body)
        .send()
        .unwrap();
    assert!(resp.status().is_success());
    let body: serde_json::Value = resp.json().unwrap();
    let hits = body["hits"]["hits"].as_array().unwrap();
    assert_eq!(hits.len(), 5);

    // Delete
    delete_test_index();
    let resp = client
        .get(&format!("{}/{}", os_base_url(), OS_INDEX))
        .send()
        .unwrap();
    assert_eq!(resp.status().as_u16(), 404);
}
