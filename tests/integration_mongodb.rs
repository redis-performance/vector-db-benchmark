//! Integration tests for the MongoDB engine.
//!
//! Requires MongoDB 8.x with Atlas Search running on port 27018 (replica set).
//! Start with: docker compose -f tests/docker-compose.test.yml up -d mongodb-search --wait
//! Run with:   MONGODB_PORT=27018 cargo test --test integration_mongodb --release -- --test-threads=1

use std::thread;
use std::time::{Duration, Instant};

use mongodb::bson::{doc, Document};
use mongodb::sync::Client;
use rand::Rng;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const MONGODB_PORT: u16 = 27018;
const MONGODB_HOST: &str = "127.0.0.1";
const TEST_DB: &str = "bench_test";
const TEST_COLLECTION: &str = "vectors";
const TEST_INDEX: &str = "vector_index";

fn mongodb_uri() -> String {
    let port: u16 = std::env::var("MONGODB_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(MONGODB_PORT);
    format!(
        "mongodb://{}:{}/?directConnection=true",
        MONGODB_HOST, port
    )
}

fn mongodb_client() -> Client {
    Client::with_uri_str(&mongodb_uri()).expect("Failed to create MongoDB client")
}

fn wait_for_mongodb() {
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        if let Ok(client) = Client::with_uri_str(&mongodb_uri()) {
            let db = client.database("admin");
            if db.run_command(doc! { "ping": 1 }).run().is_ok() {
                return;
            }
        }
        if Instant::now() > deadline {
            panic!(
                "MongoDB not available on port {} after 120s",
                MONGODB_PORT
            );
        }
        thread::sleep(Duration::from_millis(1000));
    }
}

/// Drop the search index (if any), wait for it to disappear, then drop the
/// collection and wait for it to be gone.  Mirrors the engine's configure()
/// cleanup so tests exercise the same Atlas-safe path.
fn drop_test_collection() {
    let client = mongodb_client();
    let db = client.database(TEST_DB);

    // Drop search index explicitly
    let _ = db
        .run_command(doc! {
            "dropSearchIndex": TEST_COLLECTION,
            "name": TEST_INDEX,
        })
        .run();

    // Wait for index to disappear
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        let cmd = doc! { "listSearchIndexes": TEST_COLLECTION };
        let index_exists = db.run_command(cmd).run().ok().map_or(false, |result| {
            result
                .get_document("cursor")
                .ok()
                .and_then(|c| c.get_array("firstBatch").ok())
                .map_or(false, |batch| {
                    batch.iter().any(|idx| {
                        idx.as_document()
                            .and_then(|d| d.get_str("name").ok())
                            .map_or(false, |n| n == TEST_INDEX)
                    })
                })
        });
        if !index_exists {
            break;
        }
        if Instant::now() > deadline {
            eprintln!("Warning: search index still exists after 60s");
            break;
        }
        thread::sleep(Duration::from_secs(2));
    }

    // Drop collection
    let coll = db.collection::<Document>(TEST_COLLECTION);
    let _ = coll.drop().run();

    // Wait for collection to disappear
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let names = db.list_collection_names().run().unwrap_or_default();
        if !names.contains(&TEST_COLLECTION.to_string()) {
            break;
        }
        if Instant::now() > deadline {
            break;
        }
        thread::sleep(Duration::from_secs(2));
    }
}

fn generate_test_vectors(count: usize, dim: usize) -> (Vec<i64>, Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    let ids: Vec<i64> = (0..count as i64).collect();
    let vectors: Vec<Vec<f32>> = (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    (ids, vectors)
}

fn insert_vectors(client: &Client, ids: &[i64], vectors: &[Vec<f32>]) {
    let coll = client
        .database(TEST_DB)
        .collection::<Document>(TEST_COLLECTION);

    let docs: Vec<Document> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(&id, vec)| {
            let bson_vec: Vec<mongodb::bson::Bson> = vec
                .iter()
                .map(|&f| mongodb::bson::Bson::Double(f as f64))
                .collect();
            doc! { "_id": id, "vector": bson_vec }
        })
        .collect();

    coll.insert_many(docs).run().expect("Insert failed");
}

fn create_vector_index(client: &Client, dim: usize, similarity: &str) {
    let db = client.database(TEST_DB);

    // Create collection if not exists
    let _ = db.create_collection(TEST_COLLECTION).run();

    // Insert a dummy doc so index has data
    let coll = db.collection::<Document>(TEST_COLLECTION);
    let dummy: Vec<mongodb::bson::Bson> =
        (0..dim).map(|_| mongodb::bson::Bson::Double(0.0)).collect();
    let _ = coll.insert_one(doc! { "_id": -1i64, "vector": dummy }).run();

    // Create search index
    let index_def = doc! {
        "name": TEST_INDEX,
        "type": "vectorSearch",
        "definition": {
            "fields": [{
                "type": "vector",
                "path": "vector",
                "numDimensions": dim as i32,
                "similarity": similarity,
            }]
        }
    };

    let cmd = doc! {
        "createSearchIndexes": TEST_COLLECTION,
        "indexes": [index_def],
    };

    db.run_command(cmd)
        .run()
        .expect("Failed to create vector search index");

    // Wait for index readiness
    let deadline = Instant::now() + Duration::from_secs(120);
    loop {
        let cmd = doc! { "listSearchIndexes": TEST_COLLECTION };
        if let Ok(result) = db.run_command(cmd).run() {
            if let Ok(cursor) = result.get_document("cursor") {
                if let Ok(batch) = cursor.get_array("firstBatch") {
                    for index in batch {
                        if let Some(index_doc) = index.as_document() {
                            let name = index_doc.get_str("name").unwrap_or("");
                            let status = index_doc.get_str("status").unwrap_or("");
                            let queryable = index_doc.get_bool("queryable").unwrap_or(false);
                            if name == TEST_INDEX
                                && (status == "READY" || status == "ACTIVE")
                                && queryable
                            {
                                // Remove dummy
                                let _ = coll.delete_one(doc! { "_id": -1i64 }).run();
                                return;
                            }
                        }
                    }
                }
            }
        }
        if Instant::now() > deadline {
            panic!("Vector search index did not become ready within 120s");
        }
        thread::sleep(Duration::from_secs(1));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_mongodb_collection_crud() {
    wait_for_mongodb();
    drop_test_collection();

    let client = mongodb_client();
    let db = client.database(TEST_DB);

    // Create collection
    db.create_collection(TEST_COLLECTION)
        .run()
        .expect("Failed to create collection");

    // Insert a document
    let coll = db.collection::<Document>(TEST_COLLECTION);
    coll.insert_one(doc! { "_id": 1i64, "value": "test" })
        .run()
        .expect("Failed to insert");

    // Count documents
    let count = coll.count_documents(doc! {}).run().expect("Failed to count");
    assert_eq!(count, 1);

    // Drop
    drop_test_collection();

    // Verify empty
    let count = coll.count_documents(doc! {}).run().unwrap_or(0);
    assert_eq!(count, 0);
}

#[test]
fn test_mongodb_insert_and_search() {
    wait_for_mongodb();
    drop_test_collection();

    let client = mongodb_client();
    let dim = 4;

    create_vector_index(&client, dim, "euclidean");

    // Insert known vectors
    let ids = vec![0i64, 1, 2, 3, 4];
    let vectors: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.9, 0.1, 0.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];
    insert_vectors(&client, &ids, &vectors);

    // Wait for indexing
    thread::sleep(Duration::from_secs(2));

    // Vector search for [1, 0, 0, 0]
    let coll = client
        .database(TEST_DB)
        .collection::<Document>(TEST_COLLECTION);

    let pipeline = vec![
        doc! {
            "$vectorSearch": {
                "index": TEST_INDEX,
                "path": "vector",
                "queryVector": [1.0f64, 0.0, 0.0, 0.0],
                "numCandidates": 20i64,
                "limit": 3i64,
            }
        },
        doc! {
            "$project": {
                "_id": 1,
                "score": { "$meta": "vectorSearchScore" },
            }
        },
    ];

    let cursor = coll.aggregate(pipeline).run().expect("Search failed");
    let results: Vec<Document> = cursor.filter_map(|r| r.ok()).collect();
    assert!(!results.is_empty(), "Expected search results");

    // First result should be id=0 (exact match)
    let first_id = results[0].get_i64("_id").unwrap();
    assert_eq!(first_id, 0, "First result should be exact match");

    drop_test_collection();
}

#[test]
fn test_mongodb_precision() {
    wait_for_mongodb();
    drop_test_collection();

    let client = mongodb_client();
    let dim = 8;
    let n = 200;
    let k = 10;

    create_vector_index(&client, dim, "euclidean");

    let (ids, vectors) = generate_test_vectors(n, dim);
    insert_vectors(&client, &ids, &vectors);

    // Wait for indexing
    thread::sleep(Duration::from_secs(3));

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

    // Vector search
    let coll = client
        .database(TEST_DB)
        .collection::<Document>(TEST_COLLECTION);

    let bson_query: Vec<mongodb::bson::Bson> = query
        .iter()
        .map(|&f| mongodb::bson::Bson::Double(f as f64))
        .collect();

    let pipeline = vec![
        doc! {
            "$vectorSearch": {
                "index": TEST_INDEX,
                "path": "vector",
                "queryVector": bson_query,
                "numCandidates": (k * 20) as i64,
                "limit": k as i64,
            }
        },
        doc! {
            "$project": {
                "_id": 1,
                "score": { "$meta": "vectorSearchScore" },
            }
        },
    ];

    let cursor = coll.aggregate(pipeline).run().expect("Search failed");
    let results: Vec<Document> = cursor.filter_map(|r| r.ok()).collect();

    let found: std::collections::HashSet<i64> = results
        .iter()
        .filter_map(|doc| doc.get_i64("_id").ok())
        .collect();

    let overlap = ground_truth.intersection(&found).count();
    let precision = overlap as f64 / k as f64;
    println!(
        "MongoDB euclidean precision@{}: {:.2} ({}/{})",
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
fn test_mongodb_full_cycle() {
    wait_for_mongodb();
    drop_test_collection();

    let client = mongodb_client();
    let dim = 4;

    // Create + index
    create_vector_index(&client, dim, "euclidean");

    // Upload
    let (ids, vectors) = generate_test_vectors(20, dim);
    insert_vectors(&client, &ids, &vectors);
    thread::sleep(Duration::from_secs(2));

    // Search
    let coll = client
        .database(TEST_DB)
        .collection::<Document>(TEST_COLLECTION);
    let bson_query: Vec<mongodb::bson::Bson> = vectors[0]
        .iter()
        .map(|&f| mongodb::bson::Bson::Double(f as f64))
        .collect();

    let pipeline = vec![
        doc! {
            "$vectorSearch": {
                "index": TEST_INDEX,
                "path": "vector",
                "queryVector": bson_query,
                "numCandidates": 50i64,
                "limit": 5i64,
            }
        },
        doc! {
            "$project": {
                "_id": 1,
                "score": { "$meta": "vectorSearchScore" },
            }
        },
    ];
    let cursor = coll.aggregate(pipeline).run().expect("Search failed");
    let results: Vec<Document> = cursor.filter_map(|r| r.ok()).collect();
    assert_eq!(results.len(), 5);

    // Delete
    drop_test_collection();

    let count = coll.count_documents(doc! {}).run().unwrap_or(0);
    assert_eq!(count, 0);
}

/// Two back-to-back benchmark cycles with different dimensions.
/// Verifies that index cleanup between runs is correct — the second run
/// must create a fresh index with a different dimension and still return
/// accurate results.
#[test]
fn test_mongodb_multi_dataset_runs() {
    wait_for_mongodb();
    drop_test_collection();

    let client = mongodb_client();

    // ── Run 1: dim=4, euclidean, 20 vectors ───────────────────────
    println!("=== Run 1: dim=4, euclidean ===");
    {
        let dim = 4;
        create_vector_index(&client, dim, "euclidean");

        let (ids, vectors) = generate_test_vectors(20, dim);
        insert_vectors(&client, &ids, &vectors);
        thread::sleep(Duration::from_secs(2));

        let coll = client
            .database(TEST_DB)
            .collection::<Document>(TEST_COLLECTION);
        let bson_query: Vec<mongodb::bson::Bson> = vectors[0]
            .iter()
            .map(|&f| mongodb::bson::Bson::Double(f as f64))
            .collect();

        let pipeline = vec![
            doc! {
                "$vectorSearch": {
                    "index": TEST_INDEX,
                    "path": "vector",
                    "queryVector": bson_query,
                    "numCandidates": 50i64,
                    "limit": 5i64,
                }
            },
            doc! {
                "$project": {
                    "_id": 1,
                    "score": { "$meta": "vectorSearchScore" },
                }
            },
        ];
        let cursor = coll.aggregate(pipeline).run().expect("Run 1 search failed");
        let results: Vec<Document> = cursor.filter_map(|r| r.ok()).collect();
        assert_eq!(results.len(), 5, "Run 1: expected 5 results");
        let first_id = results[0].get_i64("_id").unwrap();
        assert_eq!(first_id, ids[0], "Run 1: first result should be query vector");
    }

    // ── Cleanup between runs (mirrors engine configure()) ─────────
    println!("=== Cleanup between runs ===");
    drop_test_collection();

    // ── Run 2: dim=8, cosine, 50 vectors ──────────────────────────
    println!("=== Run 2: dim=8, cosine ===");
    {
        let dim = 8;
        create_vector_index(&client, dim, "cosine");

        let (ids, vectors) = generate_test_vectors(50, dim);
        insert_vectors(&client, &ids, &vectors);
        thread::sleep(Duration::from_secs(2));

        let coll = client
            .database(TEST_DB)
            .collection::<Document>(TEST_COLLECTION);
        let bson_query: Vec<mongodb::bson::Bson> = vectors[0]
            .iter()
            .map(|&f| mongodb::bson::Bson::Double(f as f64))
            .collect();

        let pipeline = vec![
            doc! {
                "$vectorSearch": {
                    "index": TEST_INDEX,
                    "path": "vector",
                    "queryVector": bson_query,
                    "numCandidates": 100i64,
                    "limit": 10i64,
                }
            },
            doc! {
                "$project": {
                    "_id": 1,
                    "score": { "$meta": "vectorSearchScore" },
                }
            },
        ];
        let cursor = coll.aggregate(pipeline).run().expect("Run 2 search failed");
        let results: Vec<Document> = cursor.filter_map(|r| r.ok()).collect();
        assert_eq!(results.len(), 10, "Run 2: expected 10 results");

        // Verify doc count is from run 2 only (no leftover from run 1)
        let count = coll.count_documents(doc! {}).run().expect("count failed");
        assert_eq!(count, 50, "Run 2: should have exactly 50 docs, not leftovers from run 1");
    }

    drop_test_collection();
}
