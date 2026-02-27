//! Integration tests for Redis and VectorSets engines.
//!
//! Requires redis:8.6.0 running on port 6399.
//! Start with: docker compose -f tests/docker-compose.test.yml up -d
//! Run with:   cargo test --test integration_redis -- --test-threads=1

use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;
use redis::Connection;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const TEST_PORT: u16 = 6399;
const TEST_HOST: &str = "127.0.0.1";

fn get_test_connection() -> Connection {
    let url = format!("redis://{}:{}/", TEST_HOST, TEST_PORT);
    let client = redis::Client::open(url.as_str()).expect("Failed to create Redis client");
    client
        .get_connection()
        .expect("Failed to connect to Redis. Is redis:8.6.0 running on port 6399?")
}

fn wait_for_redis() {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let url = format!("redis://{}:{}/", TEST_HOST, TEST_PORT);
        if let Ok(client) = redis::Client::open(url.as_str()) {
            if let Ok(mut conn) = client.get_connection() {
                let pong: Result<String, _> = redis::cmd("PING").query(&mut conn);
                if pong.is_ok() {
                    return;
                }
            }
        }
        if Instant::now() > deadline {
            panic!("Redis not available on port {} after 10s", TEST_PORT);
        }
        thread::sleep(Duration::from_millis(200));
    }
}

fn flush_db(conn: &mut Connection) {
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

/// Compute brute-force nearest neighbors (by L2 distance) for a query against a set of vectors.
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

// ---------------------------------------------------------------------------
// Redis (RediSearch FT.*) Integration Tests
// ---------------------------------------------------------------------------

#[test]
fn test_redis_upload_and_retrieve() {
    wait_for_redis();
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
        .arg("DD")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_redis_knn_search() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let count = 100;
    let top = 10;
    let (ids, vectors) = generate_test_vectors(count, dim);

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
        .arg("L2")
        .query::<()>(&mut conn)
        .expect("FT.CREATE failed");

    // Upload via pipeline
    {
        let mut pipe = redis::pipe();
        for i in 0..ids.len() {
            let key = ids[i].to_string();
            let vec_bytes = vec_to_bytes(&vectors[i]);
            let mut cmd = redis::cmd("HSET");
            cmd.arg(key).arg("vector").arg(&vec_bytes[..]);
            pipe.add_command(cmd);
        }
        pipe.query::<()>(&mut conn).expect("Pipeline HSET failed");
    }

    // Wait for indexing
    thread::sleep(Duration::from_millis(500));

    // Search using first vector as query
    let query_vec = &vectors[0];
    let query_bytes = vec_to_bytes(query_vec);

    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx")
        .arg("*=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]")
        .arg("SORTBY")
        .arg("vector_score")
        .arg("ASC")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(4)
        .arg("PARAMS")
        .arg(6)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .arg("EF")
        .arg(128)
        .query(&mut conn)
        .expect("FT.SEARCH failed");

    // First element is total count
    assert!(!response.is_empty(), "Should get search results");

    // The query vector itself should be the top-1 result (distance ~0)
    if let redis::Value::BulkString(data) = &response[1] {
        let top_id: i64 = String::from_utf8_lossy(data).parse().unwrap();
        assert_eq!(top_id, 0, "Query vector should be its own nearest neighbor");
    }

    // Cleanup
    let _: () = redis::cmd("FT.DROPINDEX")
        .arg("idx")
        .arg("DD")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_redis_knn_precision() {
    wait_for_redis();
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

    {
        let mut pipe = redis::pipe();
        for i in 0..ids.len() {
            let key = ids[i].to_string();
            let vec_bytes = vec_to_bytes(&vectors[i]);
            let mut cmd = redis::cmd("HSET");
            cmd.arg(key).arg("vector").arg(&vec_bytes[..]);
            pipe.add_command(cmd);
        }
        pipe.query::<()>(&mut conn).expect("Pipeline HSET failed");
    }

    thread::sleep(Duration::from_millis(500));

    // Pick a query vector from the dataset
    let query_idx = 42;
    let query_vec = &vectors[query_idx];
    let expected = brute_force_neighbors(query_vec, &vectors, top);

    let query_bytes = vec_to_bytes(query_vec);
    let response: Vec<redis::Value> = redis::cmd("FT.SEARCH")
        .arg("idx")
        .arg("*=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]")
        .arg("SORTBY")
        .arg("vector_score")
        .arg("ASC")
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(4)
        .arg("PARAMS")
        .arg(6)
        .arg("vec_param")
        .arg(&query_bytes[..])
        .arg("K")
        .arg(top)
        .arg("EF")
        .arg(256)
        .query(&mut conn)
        .expect("FT.SEARCH failed");

    // Parse result IDs
    let mut result_ids: Vec<i64> = Vec::new();
    let mut i = 1;
    while i < response.len() {
        if let redis::Value::BulkString(data) = &response[i] {
            if let Ok(id) = String::from_utf8_lossy(data).parse::<i64>() {
                result_ids.push(id);
            }
        }
        i += 2; // skip field values
    }

    // Compute precision
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
        .arg("DD")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_redis_metadata_upload() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let vec_bytes = vec_to_bytes(&[1.0f32, 2.0, 3.0, 4.0]);

    // Upload a hash with vector + metadata fields
    redis::cmd("HSET")
        .arg("meta_test_0")
        .arg("vector")
        .arg(&vec_bytes[..])
        .arg("category")
        .arg("electronics")
        .arg("tags")
        .arg("sale;new")
        .query::<()>(&mut conn)
        .expect("HSET with metadata failed");

    // Verify metadata stored
    let category: String = redis::cmd("HGET")
        .arg("meta_test_0")
        .arg("category")
        .query(&mut conn)
        .unwrap();
    assert_eq!(category, "electronics");

    let tags: String = redis::cmd("HGET")
        .arg("meta_test_0")
        .arg("tags")
        .query(&mut conn)
        .unwrap();
    assert_eq!(tags, "sale;new");
}

// ---------------------------------------------------------------------------
// VectorSets (VADD/VSIM) Integration Tests
// ---------------------------------------------------------------------------

#[test]
fn test_vectorsets_vadd_and_vsim() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let vectors: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.9, 0.1, 0.0, 0.0],
    ];

    // Upload vectors using VADD
    for (i, vec) in vectors.iter().enumerate() {
        let vec_bytes = vec_to_bytes(vec);
        redis::cmd("VADD")
            .arg("vset_basic")
            .arg("FP32")
            .arg(&vec_bytes[..])
            .arg(i.to_string())
            .arg("NOQUANT")
            .arg("CAS")
            .query::<()>(&mut conn)
            .unwrap_or_else(|e| panic!("VADD failed for vector {}: {}", i, e));
    }

    // Search for a vector close to [1,0,0,0]
    let query = vec![1.0f32, 0.0, 0.0, 0.0];
    let query_bytes = vec_to_bytes(&query);

    let response: Vec<redis::Value> = redis::cmd("VSIM")
        .arg("vset_basic")
        .arg("FP32")
        .arg(&query_bytes[..])
        .arg("WITHSCORES")
        .arg("COUNT")
        .arg(3)
        .query(&mut conn)
        .expect("VSIM failed");

    // Should get results back
    assert!(
        response.len() >= 2,
        "Expected at least one result pair, got {:?}",
        response
    );

    // Parse first result
    let top_id = match &response[0] {
        redis::Value::BulkString(data) => String::from_utf8_lossy(data).parse::<i64>().unwrap(),
        redis::Value::Int(n) => *n,
        _ => panic!("Unexpected response type: {:?}", response[0]),
    };

    // ID 0 ([1,0,0,0]) should be most similar to query [1,0,0,0]
    assert_eq!(top_id, 0, "Vector 0 should be the closest match");

    // Cleanup
    let _: () = redis::cmd("DEL")
        .arg("vset_basic")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_vectorsets_score_conversion() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    // Upload two orthogonal unit vectors
    let v0 = vec![1.0f32, 0.0];
    let v1 = vec![0.0f32, 1.0];

    for (i, vec) in [&v0, &v1].iter().enumerate() {
        let vec_bytes = vec_to_bytes(vec);
        redis::cmd("VADD")
            .arg("vset_scores")
            .arg("FP32")
            .arg(&vec_bytes[..])
            .arg(i.to_string())
            .arg("NOQUANT")
            .arg("CAS")
            .query::<()>(&mut conn)
            .unwrap();
    }

    // Search with v0 as query
    let query_bytes = vec_to_bytes(&v0);
    let response: Vec<redis::Value> = redis::cmd("VSIM")
        .arg("vset_scores")
        .arg("FP32")
        .arg(&query_bytes[..])
        .arg("WITHSCORES")
        .arg("COUNT")
        .arg(2)
        .query(&mut conn)
        .expect("VSIM failed");

    // Parse scores: VectorSets returns similarity (1 = identical, 0 = orthogonal)
    assert!(response.len() >= 4, "Expected 2 result pairs");

    let score0 = match &response[1] {
        redis::Value::BulkString(data) => String::from_utf8_lossy(data).parse::<f64>().unwrap(),
        redis::Value::Double(f) => *f,
        _ => panic!("Unexpected score type"),
    };
    let score1 = match &response[3] {
        redis::Value::BulkString(data) => String::from_utf8_lossy(data).parse::<f64>().unwrap(),
        redis::Value::Double(f) => *f,
        _ => panic!("Unexpected score type"),
    };

    // After 1-score conversion: identical vector -> 0, orthogonal -> ~1
    let dist0 = 1.0 - score0;
    let dist1 = 1.0 - score1;

    assert!(
        dist0 < 0.01,
        "Self-similarity distance should be ~0, got {}",
        dist0
    );
    assert!(
        dist1 > 0.4,
        "Orthogonal distance should be large, got {}",
        dist1
    );

    let _: () = redis::cmd("DEL")
        .arg("vset_scores")
        .query(&mut conn)
        .unwrap();
}

#[test]
fn test_vectorsets_pipeline_batch() {
    wait_for_redis();
    let mut conn = get_test_connection();
    flush_db(&mut conn);

    let dim = 8;
    let count = 50;
    let (ids, vectors) = generate_test_vectors(count, dim);

    // Upload via pipeline (matching engine implementation)
    let mut pipe = redis::pipe();
    for i in 0..ids.len() {
        let vec_bytes = vec_to_bytes(&vectors[i]);
        let mut cmd = redis::cmd("VADD");
        cmd.arg("vset_pipe")
            .arg("FP32")
            .arg(&vec_bytes[..])
            .arg(ids[i].to_string())
            .arg("NOQUANT")
            .arg("M")
            .arg(16)
            .arg("EF")
            .arg(200)
            .arg("CAS");
        pipe.add_command(cmd);
    }
    pipe.query::<()>(&mut conn).expect("Pipeline VADD failed");

    // Verify we can search and get results
    let query_bytes = vec_to_bytes(&vectors[0]);
    let response: Vec<redis::Value> = redis::cmd("VSIM")
        .arg("vset_pipe")
        .arg("FP32")
        .arg(&query_bytes[..])
        .arg("WITHSCORES")
        .arg("COUNT")
        .arg(5)
        .arg("EF")
        .arg(64)
        .query(&mut conn)
        .expect("VSIM failed");

    assert!(
        response.len() >= 2,
        "Should get at least 1 result from VSIM"
    );

    // The query vector (id=0) should appear in results
    let top_id = match &response[0] {
        redis::Value::BulkString(data) => String::from_utf8_lossy(data).parse::<i64>().unwrap(),
        redis::Value::Int(n) => *n,
        _ => panic!("Unexpected type"),
    };
    assert_eq!(top_id, 0, "Query vector should be its own nearest neighbor");

    let _: () = redis::cmd("DEL").arg("vset_pipe").query(&mut conn).unwrap();
}
