//! Integration tests for the Qdrant engine.
//!
//! Requires Qdrant running on gRPC port 6335 and REST port 6334.
//! Start with: docker compose -f tests/docker-compose.test.yml up -d qdrant
//! Run with:   QDRANT_GRPC_PORT=6335 cargo test --test integration_qdrant -- --test-threads=1

use rand::Rng;
use std::thread;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const QDRANT_REST_PORT: u16 = 6334;
const QDRANT_GRPC_PORT: u16 = 6335;
const QDRANT_HOST: &str = "127.0.0.1";
const COLLECTION: &str = "bench_test";

fn rest_url() -> String {
    format!("http://{}:{}", QDRANT_HOST, QDRANT_REST_PORT)
}

fn grpc_url() -> String {
    format!("http://{}:{}", QDRANT_HOST, QDRANT_GRPC_PORT)
}

fn rest_client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .expect("Failed to create HTTP client")
}

fn wait_for_qdrant() {
    let client = rest_client();
    let url = format!("{}/collections", rest_url());
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if let Ok(resp) = client.get(&url).send() {
            if resp.status().is_success() {
                return;
            }
        }
        if Instant::now() > deadline {
            panic!(
                "Qdrant not available on port {} after 60s",
                QDRANT_REST_PORT
            );
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn delete_collection() {
    let client = rest_client();
    let _ = client
        .delete(&format!("{}/collections/{}", rest_url(), COLLECTION))
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

fn brute_force_neighbors_l2(query: &[f32], vectors: &[Vec<f32>], top: usize) -> Vec<i64> {
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

fn create_grpc_client() -> (tokio::runtime::Runtime, qdrant_client::Qdrant) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    // `from_url(...).build()` is synchronous and returns a Result, not a future.
    let client = qdrant_client::Qdrant::from_url(&grpc_url())
        .build()
        .unwrap();
    (rt, client)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_qdrant_collection_crud() {
    wait_for_qdrant();
    delete_collection();

    let (rt, client) = create_grpc_client();

    // Create collection
    use qdrant_client::qdrant::{
        vectors_config::Config, CreateCollectionBuilder, Distance, VectorParamsBuilder,
        VectorsConfig,
    };

    rt.block_on(
        client.create_collection(CreateCollectionBuilder::new(COLLECTION).vectors_config(
            VectorsConfig {
                config: Some(Config::Params(
                    VectorParamsBuilder::new(4, Distance::Euclid).build(),
                )),
            },
        )),
    )
    .expect("Failed to create collection");

    // Verify exists
    let info = rt
        .block_on(client.collection_info(COLLECTION))
        .expect("Failed to get collection info");
    assert!(info.result.is_some(), "Collection should exist");

    // Delete
    rt.block_on(
        client.delete_collection(qdrant_client::qdrant::DeleteCollectionBuilder::new(
            COLLECTION,
        )),
    )
    .expect("Failed to delete collection");
}

#[test]
fn test_qdrant_upsert_and_search() {
    wait_for_qdrant();
    delete_collection();

    let (rt, client) = create_grpc_client();

    use qdrant_client::qdrant::{
        vectors_config::Config, CreateCollectionBuilder, Distance, PointStruct,
        SearchPointsBuilder, VectorParamsBuilder, VectorsConfig,
    };

    // Create collection
    rt.block_on(
        client.create_collection(CreateCollectionBuilder::new(COLLECTION).vectors_config(
            VectorsConfig {
                config: Some(Config::Params(
                    VectorParamsBuilder::new(4, Distance::Euclid).build(),
                )),
            },
        )),
    )
    .unwrap();

    // Upsert points
    let (ids, vectors) = generate_test_vectors(50, 4);
    let points: Vec<PointStruct> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(id, vec)| PointStruct::new(*id as u64, vec.clone(), qdrant_client::Payload::new()))
        .collect();

    rt.block_on(client.upsert_points(
        qdrant_client::qdrant::UpsertPointsBuilder::new(COLLECTION, points).wait(true),
    ))
    .expect("Failed to upsert points");

    // Search
    let results = rt
        .block_on(client.search_points(
            SearchPointsBuilder::new(COLLECTION, vectors[0].clone(), 5).with_payload(false),
        ))
        .expect("Failed to search");

    assert!(!results.result.is_empty(), "Search should return results");

    // First result should be the query vector itself (id=0)
    if let Some(first) = results.result.first() {
        if let Some(id) = &first.id {
            if let Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) =
                &id.point_id_options
            {
                assert_eq!(*n, 0, "Query vector should be its own nearest neighbor");
            }
        }
    }

    delete_collection();
}

#[test]
fn test_qdrant_precision() {
    wait_for_qdrant();
    delete_collection();

    let (rt, client) = create_grpc_client();

    use qdrant_client::qdrant::{
        vectors_config::Config, CreateCollectionBuilder, Distance, PointStruct,
        SearchPointsBuilder, VectorParamsBuilder, VectorsConfig,
    };

    let dim = 8;
    let count = 100;
    let top = 10;

    rt.block_on(
        client.create_collection(CreateCollectionBuilder::new(COLLECTION).vectors_config(
            VectorsConfig {
                config: Some(Config::Params(
                    VectorParamsBuilder::new(dim as u64, Distance::Euclid).build(),
                )),
            },
        )),
    )
    .unwrap();

    let (ids, vectors) = generate_test_vectors(count, dim);
    let points: Vec<PointStruct> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(id, vec)| PointStruct::new(*id as u64, vec.clone(), qdrant_client::Payload::new()))
        .collect();

    rt.block_on(client.upsert_points(
        qdrant_client::qdrant::UpsertPointsBuilder::new(COLLECTION, points).wait(true),
    ))
    .unwrap();

    // Wait for indexing
    thread::sleep(Duration::from_secs(2));

    let query_idx = 42;
    let expected = brute_force_neighbors_l2(&vectors[query_idx], &vectors, top);

    let results = rt
        .block_on(
            client.search_points(
                SearchPointsBuilder::new(COLLECTION, vectors[query_idx].clone(), top as u64)
                    .with_payload(false),
            ),
        )
        .unwrap();

    let found: std::collections::HashSet<i64> = results
        .result
        .iter()
        .filter_map(|p| {
            if let Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) =
                &p.id.as_ref().and_then(|id| id.point_id_options.as_ref())
            {
                Some(*n as i64)
            } else {
                None
            }
        })
        .collect();

    let expected_set: std::collections::HashSet<i64> = expected.into_iter().collect();
    let hits = expected_set.intersection(&found).count();
    let precision = hits as f64 / top as f64;

    assert!(
        precision >= 0.9,
        "Precision should be >= 0.9 for small dataset, got {}",
        precision
    );

    delete_collection();
}

#[test]
fn test_qdrant_payload_filter() {
    wait_for_qdrant();
    delete_collection();

    let (rt, client) = create_grpc_client();

    use qdrant_client::qdrant::{
        vectors_config::Config, Condition, CreateCollectionBuilder, Distance, FieldType, Filter,
        PointStruct, SearchPointsBuilder, VectorParamsBuilder, VectorsConfig,
    };

    rt.block_on(
        client.create_collection(CreateCollectionBuilder::new(COLLECTION).vectors_config(
            VectorsConfig {
                config: Some(Config::Params(
                    VectorParamsBuilder::new(4, Distance::Euclid).build(),
                )),
            },
        )),
    )
    .unwrap();

    // Create field index
    rt.block_on(client.create_field_index(
        qdrant_client::qdrant::CreateFieldIndexCollectionBuilder::new(
            COLLECTION,
            "category",
            FieldType::Keyword,
        ),
    ))
    .unwrap();

    // Upsert with payload
    let (ids, vectors) = generate_test_vectors(20, 4);
    let points: Vec<PointStruct> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(id, vec)| {
            let mut payload = qdrant_client::Payload::new();
            payload.insert("category", if *id % 2 == 0 { "A" } else { "B" });
            PointStruct::new(*id as u64, vec.clone(), payload)
        })
        .collect();

    rt.block_on(client.upsert_points(
        qdrant_client::qdrant::UpsertPointsBuilder::new(COLLECTION, points).wait(true),
    ))
    .unwrap();

    // Search with filter: only category "A"
    let filter = Filter {
        must: vec![Condition::matches("category", "A".to_string())],
        ..Default::default()
    };

    let results = rt
        .block_on(
            client.search_points(
                SearchPointsBuilder::new(COLLECTION, vectors[0].clone(), 10)
                    .filter(filter)
                    .with_payload(true),
            ),
        )
        .unwrap();

    assert!(
        !results.result.is_empty(),
        "Filtered search should return results"
    );

    // Verify all results have even IDs (category A)
    for p in &results.result {
        if let Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(n)) =
            &p.id.as_ref().and_then(|id| id.point_id_options.as_ref())
        {
            assert!(
                *n % 2 == 0,
                "Filtered search should only return category A (even IDs), got id={}",
                n
            );
        }
    }

    delete_collection();
}

// ---------------------------------------------------------------------------
// Binary-level coverage: run the real engine end-to-end via the CLI.
// Covers the query_points migration and prefetch (search_params.prefetch).
// ---------------------------------------------------------------------------

fn binary_path() -> std::path::PathBuf {
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
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/release/vector-db-benchmark")
}

/// Write a temp project (datasets + configs + results) and return its root.
fn write_dense_project(
    dataset_name: &str,
    engine_configs_json: &str,
    vectors: &[Vec<f32>],
    queries: &[Vec<f32>],
    neighbors: &[Vec<i64>],
    dim: usize,
) -> std::path::PathBuf {
    use std::fs;
    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp);

    let dataset_dir = root.join("datasets").join(dataset_name);
    fs::create_dir_all(&dataset_dir).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();

    let jsonl = |rows: &[Vec<f32>]| -> String {
        rows.iter()
            .map(|v| {
                serde_json::to_string(&v.iter().map(|x| *x as f64).collect::<Vec<_>>()).unwrap()
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    fs::write(dataset_dir.join("vectors.jsonl"), jsonl(vectors)).unwrap();
    fs::write(dataset_dir.join("queries.jsonl"), jsonl(queries)).unwrap();
    let nb = neighbors
        .iter()
        .map(|n| serde_json::to_string(n).unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(dataset_dir.join("neighbours.jsonl"), nb).unwrap();

    let datasets_json = serde_json::json!([{
        "name": dataset_name, "type": "jsonl", "path": format!("{}/", dataset_name),
        "distance": "l2", "vector_size": dim, "vector_count": vectors.len(),
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

fn run_qdrant_binary(root: &std::path::Path, engine: &str, dataset: &str) -> bool {
    let out = std::process::Command::new(binary_path())
        .args([
            "--engines",
            engine,
            "--datasets",
            dataset,
            "--host",
            "localhost",
            "--skip-if-exists",
            "false",
        ])
        .env("QDRANT_GRPC_PORT", QDRANT_GRPC_PORT.to_string())
        .env("QDRANT_REST_PORT", QDRANT_REST_PORT.to_string())
        .current_dir(root)
        .output()
        .expect("run vector-db-benchmark");
    if !out.status.success() {
        eprintln!(
            "stdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    out.status.success()
}

fn read_precision(root: &std::path::Path, engine: &str) -> f64 {
    use std::fs;
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
    v["results"]["mean_precisions"].as_f64().unwrap()
}

/// End-to-end via the real engine: a plain search (covers the query_points
/// migration) and a prefetch/two-stage search both return high-recall results.
#[test]
fn test_binary_qdrant_query_points_and_prefetch() {
    wait_for_qdrant();

    let dim = 8;
    let (_ids, vectors) = generate_test_vectors(200, dim);
    let queries: Vec<Vec<f32>> = vectors[..10].to_vec();
    let top = 10;
    let neighbors: Vec<Vec<i64>> = queries
        .iter()
        .map(|q| brute_force_neighbors_l2(q, &vectors, top))
        .collect();

    let configs = serde_json::json!([
        {
            "name": "qdrant-qp", "engine": "qdrant",
            "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
            "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 128}}],
            "upload_params": {"parallel": 1, "batch_size": 100}
        },
        {
            "name": "qdrant-pf", "engine": "qdrant",
            "connection_params": {"timeout": 60}, "collection_params": {"timeout": 60},
            "search_params": [{"parallel": 1, "search_params": {
                "hnsw_ef": 128, "prefetch": {"limit": 50, "params": {"hnsw_ef": 256}}
            }}],
            "upload_params": {"parallel": 1, "batch_size": 100}
        }
    ]);
    let root = write_dense_project(
        "qp-test",
        &serde_json::to_string(&configs).unwrap(),
        &vectors,
        &queries,
        &neighbors,
        dim,
    );

    assert!(
        run_qdrant_binary(&root, "qdrant-qp", "qp-test"),
        "plain run failed"
    );
    let p_plain = read_precision(&root, "qdrant-qp");
    assert!(
        p_plain >= 0.9,
        "query_points precision {:.3} < 0.9",
        p_plain
    );

    assert!(
        run_qdrant_binary(&root, "qdrant-pf", "qp-test"),
        "prefetch run failed"
    );
    let p_pf = read_precision(&root, "qdrant-pf");
    assert!(p_pf >= 0.9, "prefetch precision {:.3} < 0.9", p_pf);
    println!(
        "qdrant query_points precision={:.3}, prefetch precision={:.3}",
        p_plain, p_pf
    );
}
