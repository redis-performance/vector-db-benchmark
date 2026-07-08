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
