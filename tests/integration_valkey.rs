//! Integration tests for Valkey engine (Valkey Search FT.* commands).
//!
//! Requires valkey/valkey-bundle running on port 6380.
//! Start with: docker compose -f tests/docker-compose.test.yml up -d valkey
//! Run with:   VALKEY_PORT=6380 cargo test --test integration_valkey -- --test-threads=1

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;
use redis::Connection;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn test_port() -> u16 {
    std::env::var("VALKEY_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6380)
}

const TEST_HOST: &str = "127.0.0.1";

fn get_test_connection() -> Connection {
    let url = format!("redis://{}:{}/", TEST_HOST, test_port());
    let client = redis::Client::open(url.as_str()).expect("Failed to create Valkey client");
    client
        .get_connection()
        .expect("Failed to connect to Valkey. Is valkey-bundle running?")
}

fn wait_for_valkey() {
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
            panic!("Valkey not available on port {} after 30s", test_port());
        }
        thread::sleep(Duration::from_millis(200));
    }
}

fn flush_db(conn: &mut Connection) {
    // Drop all FT indexes first (Valkey Search does not support DD flag)
    if let Ok(indexes) = redis::cmd("FT._LIST").query::<Vec<String>>(conn) {
        for idx_name in indexes {
            let _ = redis::cmd("FT.DROPINDEX")
                .arg(&idx_name)
                .query::<()>(conn);
        }
    }
    let _: () = redis::cmd("FLUSHALL").query(conn).unwrap();
}

fn generate_test_vectors(count: usize, dim: usize) -> (Vec<i64>, Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    let ids: Vec<i64> = (0..count as i64).collect();
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    (ids, vectors)
}

fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn brute_force_neighbors(query: &[f32], vectors: &[Vec<f32>], top: usize) -> Vec<i64> {
    let mut dists: Vec<(i64, f64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d: f64 = query
                .iter()
                .zip(v.iter())
                .map(|(a, b)| (*a as f64 - *b as f64).powi(2))
                .sum();
            (i as i64, d)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(top).map(|(id, _)| *id).collect()
}

fn parse_ft_search_ids(response: &[redis::Value]) -> Vec<i64> {
    let mut ids = Vec::new();
    let mut i = 1;
    while i < response.len() {
        let id_str = match &response[i] {
            redis::Value::BulkString(data) => Some(String::from_utf8_lossy(data).to_string()),
            redis::Value::SimpleString(s) => Some(s.clone()),
            _ => None,
        };
        if let Some(s) = id_str {
            if let Ok(id) = s.parse::<i64>() {
                ids.push(id);
            }
        }
        i += 2;
    }
    ids
}

fn extract_ft_info_value(info: &redis::Value, key: &str) -> Option<i64> {
    fn value_to_string(v: &redis::Value) -> Option<String> {
        match v {
            redis::Value::BulkString(s) => Some(String::from_utf8_lossy(s).to_string()),
            redis::Value::SimpleString(s) => Some(s.clone()),
            _ => None,
        }
    }
    fn value_to_i64(v: &redis::Value) -> Option<i64> {
        match v {
            redis::Value::Int(n) => Some(*n),
            redis::Value::BulkString(s) => String::from_utf8_lossy(s).parse::<i64>().ok(),
            redis::Value::SimpleString(s) => s.parse::<i64>().ok(),
            redis::Value::Double(f) => Some(*f as i64),
            _ => None,
        }
    }
    match info {
        redis::Value::Map(pairs) => {
            for (k, v) in pairs {
                if value_to_string(k).as_deref() == Some(key) {
                    return value_to_i64(v);
                }
            }
            None
        }
        redis::Value::Array(items) => {
            for i in 0..items.len().saturating_sub(1) {
                if value_to_string(&items[i]).as_deref() == Some(key) {
                    return value_to_i64(&items[i + 1]);
                }
            }
            None
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Valkey (Valkey Search FT.*) Integration Tests
// ---------------------------------------------------------------------------

#[test]
fn test_valkey_upload_and_retrieve() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 4;
    let (ids, vectors) = generate_test_vectors(10, dim);

    // Create index
    redis::cmd("FT.CREATE")
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
        .arg("6")
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dim)
        .arg("DISTANCE_METRIC")
        .arg("COSINE")
        .query::<()>(&mut conn)
        .expect("FT.CREATE failed");

    // Upload vectors via HSET
    for i in 0..ids.len() {
        let key = ids[i].to_string();
        let vec_bytes = vec_to_bytes(&vectors[i]);
        redis::cmd("HSET")
            .arg(&key)
            .arg("vector")
            .arg(&vec_bytes[..])
            .query::<()>(&mut conn)
            .expect("HSET failed");
    }

    // Verify vectors stored
    for i in 0..ids.len() {
        let key = ids[i].to_string();
        let exists: bool = redis::cmd("EXISTS").arg(&key).query(&mut conn).unwrap();
        assert!(exists, "Key {} should exist", key);
    }

    // Cleanup
    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_valkey_knn_search() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let count = 100;
    let top = 10;
    let (ids, vectors) = generate_test_vectors(count, dim);

    redis::cmd("FT.CREATE")
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
        .arg("6")
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dim)
        .arg("DISTANCE_METRIC")
        .arg("L2")
        .query::<()>(&mut conn)
        .expect("FT.CREATE failed");

    // Upload via pipeline (batched)
    for chunk_start in (0..ids.len()).step_by(50) {
        let chunk_end = (chunk_start + 50).min(ids.len());
        let mut pipe = redis::pipe();
        for i in chunk_start..chunk_end {
            let key = ids[i].to_string();
            let vec_bytes = vec_to_bytes(&vectors[i]);
            pipe.cmd("HSET").arg(key).arg("vector").arg(vec_bytes).ignore();
        }
        pipe.query::<()>(&mut conn).expect("Pipeline HSET failed");
    }

    thread::sleep(Duration::from_millis(500));

    let query_vec = &vectors[0];
    let query_bytes = vec_to_bytes(query_vec);

    // Valkey Search: DIALECT 2 only, no EF_RUNTIME, no SORTBY on computed fields
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx")
        .arg("*=>[KNN $K @vector $vec_param AS vector_score]")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(2)
        .arg("PARAMS")
        .arg(4)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .query(&mut conn)
        .expect("FT.SEARCH failed");

    assert!(!response.is_empty(), "Should get search results");

    if let redis::Value::BulkString(data) = &response[1] {
        let top_id: i64 = String::from_utf8_lossy(data).parse().unwrap();
        assert_eq!(top_id, 0, "Query vector should be its own nearest neighbor");
    }

    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_valkey_knn_precision() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 16;
    let count = 200;
    let top = 5;
    let (ids, vectors) = generate_test_vectors(count, dim);

    redis::cmd("FT.CREATE")
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
        .arg("6")
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dim)
        .arg("DISTANCE_METRIC")
        .arg("L2")
        .query::<()>(&mut conn)
        .expect("FT.CREATE failed");

    for chunk_start in (0..ids.len()).step_by(50) {
        let chunk_end = (chunk_start + 50).min(ids.len());
        let mut pipe = redis::pipe();
        for i in chunk_start..chunk_end {
            let key = ids[i].to_string();
            let vec_bytes = vec_to_bytes(&vectors[i]);
            pipe.cmd("HSET").arg(key).arg("vector").arg(vec_bytes).ignore();
        }
        pipe.query::<()>(&mut conn).expect("Pipeline HSET failed");
    }

    thread::sleep(Duration::from_millis(500));

    let query_idx = 42;
    let query_vec = &vectors[query_idx];
    let expected = brute_force_neighbors(query_vec, &vectors, top);

    let query_bytes = vec_to_bytes(query_vec);
    // Valkey Search: DIALECT 2 only, no EF_RUNTIME
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx")
        .arg("*=>[KNN $K @vector $vec_param AS vector_score]")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(2)
        .arg("PARAMS")
        .arg(4)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .query(&mut conn)
        .expect("FT.SEARCH failed");

    let mut result_ids: Vec<i64> = Vec::new();
    let mut i = 1;
    while i < response.len() {
        if let redis::Value::BulkString(data) = &response[i] {
            if let Ok(id) = String::from_utf8_lossy(data).parse::<i64>() {
                result_ids.push(id);
            }
        }
        i += 2;
    }

    let expected_set: std::collections::HashSet<i64> = expected.into_iter().collect();
    let found_set: std::collections::HashSet<i64> = result_ids.iter().copied().collect();
    let hits = expected_set.intersection(&found_set).count();
    let precision = hits as f64 / top as f64;

    assert!(
        precision >= 0.8,
        "Precision should be >= 0.8, got {} (expected {:?}, found {:?})",
        precision,
        expected_set,
        found_set
    );

    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_valkey_filtered_knn_search() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let count = 20;
    let top = 5;
    let (ids, vectors) = generate_test_vectors(count, dim);

    // Create index with TAG + NUMERIC metadata fields
    // Note: Valkey Search does not support SORTABLE keyword
    redis::cmd("FT.CREATE")
        .arg("idx_filter")
        .arg("ON")
        .arg("HASH")
        .arg("PREFIX")
        .arg("1")
        .arg("")
        .arg("SCHEMA")
        .arg("vector")
        .arg("VECTOR")
        .arg("HNSW")
        .arg("6")
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dim)
        .arg("DISTANCE_METRIC")
        .arg("L2")
        .arg("category")
        .arg("TAG")
        .arg("SEPARATOR")
        .arg(";")
        .arg("price")
        .arg("NUMERIC")
        .query::<()>(&mut conn)
        .expect("FT.CREATE with metadata failed");

    // Upload: even IDs = "electronics", odd = "clothing"; price = 10 * id
    {
        let mut pipe = redis::pipe();
        for i in 0..ids.len() {
            let key = ids[i].to_string();
            let vec_bytes = vec_to_bytes(&vectors[i]);
            let category = if i % 2 == 0 {
                "electronics"
            } else {
                "clothing"
            };
            let price = (i * 10).to_string();
            pipe.cmd("HSET")
                .arg(key)
                .arg("vector")
                .arg(vec_bytes)
                .arg("category")
                .arg(category)
                .arg("price")
                .arg(price)
                .ignore();
        }
        pipe.query::<()>(&mut conn)
            .expect("Pipeline HSET with metadata failed");
    }

    thread::sleep(Duration::from_millis(500));

    let query_vec = &vectors[0];
    let query_bytes = vec_to_bytes(query_vec);

    // TAG filter (electronics only) - Valkey Search compatible
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx_filter")
        .arg("@category:{electronics}=>[KNN $K @vector $vec_param AS vector_score]")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(2)
        .arg("PARAMS")
        .arg(4)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .query(&mut conn)
        .expect("FT.SEARCH with TAG filter failed");

    let tag_ids = parse_ft_search_ids(&response);
    assert!(!tag_ids.is_empty(), "TAG filter should return results");
    for id in &tag_ids {
        assert_eq!(
            *id % 2,
            0,
            "TAG filter for electronics should only return even IDs, got {}",
            id
        );
    }

    // NUMERIC range filter (price < 100) - Valkey Search compatible
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx_filter")
        .arg("@price:[-inf (100]=>[KNN $K @vector $vec_param AS vector_score]")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(2)
        .arg("PARAMS")
        .arg(4)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .query(&mut conn)
        .expect("FT.SEARCH with NUMERIC filter failed");

    let num_ids = parse_ft_search_ids(&response);
    assert!(
        !num_ids.is_empty(),
        "NUMERIC range filter should return results"
    );
    for id in &num_ids {
        assert!(
            *id < 10,
            "NUMERIC filter price<100 should only return ids<10, got {}",
            id
        );
    }

    // Cleanup
    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx_filter")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_valkey_ft_info() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 4;
    let count = 50;

    redis::cmd("FT.CREATE")
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
        .arg("6")
        .arg("TYPE")
        .arg("FLOAT32")
        .arg("DIM")
        .arg(dim)
        .arg("DISTANCE_METRIC")
        .arg("L2")
        .query::<()>(&mut conn)
        .expect("FT.CREATE failed");

    let (ids, vectors) = generate_test_vectors(count, dim);
    {
        let mut pipe = redis::pipe();
        for i in 0..ids.len() {
            let key = ids[i].to_string();
            let vec_bytes = vec_to_bytes(&vectors[i]);
            pipe.cmd("HSET").arg(key).arg("vector").arg(vec_bytes).ignore();
        }
        pipe.query::<()>(&mut conn).expect("Pipeline HSET failed");
    }

    thread::sleep(Duration::from_millis(500));

    let info: redis::Value = redis::cmd("FT.INFO")
        .arg("idx")
        .query(&mut conn)
        .expect("FT.INFO failed");

    let num_docs = extract_ft_info_value(&info, "num_docs");
    assert!(
        num_docs.is_some(),
        "FT.INFO should contain num_docs field"
    );
    assert_eq!(
        num_docs.unwrap(),
        count as i64,
        "FT.INFO num_docs should match uploaded count"
    );

    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx")
        .query(&mut conn)
        .unwrap();
}

/// Find the release binary path (same pattern as Redis integration tests)
fn binary_path() -> PathBuf {
    let mut path = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    path.push("vector-db-benchmark");
    if path.exists() {
        return path;
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/release/vector-db-benchmark")
}

/// Create a temporary project directory with inline dataset and engine config
fn create_test_project(
    dataset_name: &str,
    engine_configs_json: &str,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    neighbors: &[Vec<i64>],
    distance: &str,
    dim: usize,
) -> PathBuf {
    let tmp = tempfile::tempdir().expect("Failed to create temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp);

    let dataset_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&dataset_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();

    // vectors.jsonl
    let mut vecs_content = String::new();
    for v in vectors {
        let line: Vec<f64> = v.iter().map(|x| *x as f64).collect();
        vecs_content.push_str(&serde_json::to_string(&line).unwrap());
        vecs_content.push('\n');
    }
    fs::write(dataset_dir.join("vectors.jsonl"), &vecs_content).unwrap();

    // queries.jsonl
    let mut queries_content = String::new();
    for q in queries {
        let line: Vec<f64> = q.iter().map(|x| *x as f64).collect();
        queries_content.push_str(&serde_json::to_string(&line).unwrap());
        queries_content.push('\n');
    }
    fs::write(dataset_dir.join("queries.jsonl"), &queries_content).unwrap();

    // neighbours.jsonl
    let mut neighbors_content = String::new();
    for n in neighbors {
        neighbors_content.push_str(&serde_json::to_string(n).unwrap());
        neighbors_content.push('\n');
    }
    fs::write(dataset_dir.join("neighbours.jsonl"), &neighbors_content).unwrap();

    // datasets.json
    let datasets_json = serde_json::json!([{
        "name": dataset_name,
        "type": "jsonl",
        "path": format!("{}/", dataset_name),
        "distance": distance,
        "vector_size": dim,
        "vector_count": vectors.len(),
    }]);
    fs::write(
        root.join("datasets/datasets.json"),
        serde_json::to_string_pretty(&datasets_json).unwrap(),
    )
    .unwrap();

    // Engine configs
    fs::write(
        root.join("experiments/configurations/test.json"),
        engine_configs_json,
    )
    .unwrap();

    root
}

#[test]
fn test_valkey_full_cycle_via_binary() {
    wait_for_valkey();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 10;
    let count = 50;
    let top = 5;
    let mut rng = rand::thread_rng();

    // Generate vectors & queries
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect();
    let queries: Vec<Vec<f32>> = (0..10)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect())
        .collect();
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_neighbors(q, &vectors, top))
        .collect();

    let engine_config = serde_json::json!([{
        "name": "test-valkey-l2",
        "engine": "valkey",
        "connection_params": {},
        "collection_params": {
            "hnsw_config": { "M": 16, "EF_CONSTRUCTION": 128 }
        },
        "search_params": [
            { "parallel": 1, "search_params": { "ef": 128 } }
        ],
        "upload_params": { "parallel": 1 }
    }]);

    let project_root = create_test_project(
        "test-l2",
        &serde_json::to_string_pretty(&engine_config).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        "l2",
        dim,
    );

    let bin = binary_path();
    assert!(
        bin.exists(),
        "Binary not found at {:?}. Run `cargo build --release` first.",
        bin
    );

    let output = Command::new(&bin)
        .args([
            "--engines",
            "test-valkey-l2",
            "--datasets",
            "test-l2",
            "--host",
            TEST_HOST,
        ])
        .env("VALKEY_PORT", test_port().to_string())
        .current_dir(&project_root)
        .output()
        .expect("Failed to run benchmark binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "Benchmark should complete successfully. Exit code: {:?}\nstdout: {}\nstderr: {}",
        output.status.code(),
        stdout,
        stderr
    );

    // Verify output contains expected phases
    assert!(
        stdout.contains("Configure") || stdout.contains("configure"),
        "Output should mention Configure phase"
    );
    assert!(
        stdout.contains("Upload") || stdout.contains("upload"),
        "Output should mention Upload phase"
    );
    assert!(
        stdout.contains("Search") || stdout.contains("search") || stdout.contains("QPS"),
        "Output should mention Search phase or QPS"
    );
}
