//! Integration tests for the PgVector engine.
//!
//! Requires PostgreSQL with pgvector extension running on port 5433.
//! Start with: docker compose -f tests/docker-compose.test.yml up -d pgvector
//! Run with:   PGVECTOR_PORT=5433 cargo test --test integration_pgvector -- --test-threads=1

use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;

mod common;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const PG_PORT: u16 = 5433;
const PG_HOST: &str = "127.0.0.1";
const PG_DB: &str = "postgres";
const PG_USER: &str = "postgres";
const PG_PASSWORD: &str = "passwd";

fn connect() -> postgres::Client {
    let conn_str = format!(
        "host={} port={} dbname={} user={} password={}",
        PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
    );
    postgres::Client::connect(&conn_str, postgres::NoTls).expect("Failed to connect to PostgreSQL")
}

fn wait_for_postgres() {
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        if connect_try().is_ok() {
            return;
        }
        if Instant::now() > deadline {
            panic!("PostgreSQL not available on port {} after 60s", PG_PORT);
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn connect_try() -> Result<postgres::Client, postgres::Error> {
    let conn_str = format!(
        "host={} port={} dbname={} user={} password={}",
        PG_HOST, PG_PORT, PG_DB, PG_USER, PG_PASSWORD
    );
    postgres::Client::connect(&conn_str, postgres::NoTls)
}

fn cleanup(conn: &mut postgres::Client) {
    let _ = conn.execute("DROP TABLE IF EXISTS items CASCADE", &[]);
}

fn setup_table(conn: &mut postgres::Client, dim: usize) {
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector", &[])
        .unwrap();
    conn.execute("DROP TABLE IF EXISTS items CASCADE", &[])
        .unwrap();
    conn.execute(
        &format!(
            "CREATE TABLE items (id SERIAL PRIMARY KEY, embedding vector({}) NOT NULL)",
            dim
        ),
        &[],
    )
    .unwrap();
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

fn brute_force_neighbors_cosine(query: &[f32], vectors: &[Vec<f32>], top: usize) -> Vec<i64> {
    let q_norm: f64 = query
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let mut dists: Vec<(i64, f64)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let v_norm: f64 = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
            let dot: f64 = query
                .iter()
                .zip(v.iter())
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum();
            let sim = if q_norm > 0.0 && v_norm > 0.0 {
                dot / (q_norm * v_norm)
            } else {
                0.0
            };
            (i as i64, 1.0 - sim) // cosine distance
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.iter().take(top).map(|(id, _)| *id).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_pgvector_create_table_and_extension() {
    wait_for_postgres();
    let mut conn = connect();
    cleanup(&mut conn);

    conn.execute("CREATE EXTENSION IF NOT EXISTS vector", &[])
        .unwrap();
    conn.execute(
        "CREATE TABLE items (id SERIAL PRIMARY KEY, embedding vector(4) NOT NULL)",
        &[],
    )
    .unwrap();

    // Verify table exists
    let row = conn
        .query_one(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'items'",
            &[],
        )
        .unwrap();
    let count: i64 = row.get(0);
    assert_eq!(count, 1, "items table should exist");

    cleanup(&mut conn);
}

#[test]
fn test_pgvector_copy_upload() {
    wait_for_postgres();
    let mut conn = connect();
    let dim = 4;
    let count = 50;
    setup_table(&mut conn, dim);

    let (ids, vectors) = generate_test_vectors(count, dim);

    // COPY bulk upload
    {
        let mut writer = conn
            .copy_in("COPY items (id, embedding) FROM STDIN WITH (FORMAT text)")
            .unwrap();
        use std::io::Write;
        for i in 0..ids.len() {
            let vec_str: String = vectors[i]
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(writer, "{}\t[{}]", ids[i], vec_str).unwrap();
        }
        writer.finish().unwrap();
    }

    // Verify count
    let row = conn.query_one("SELECT COUNT(*) FROM items", &[]).unwrap();
    let db_count: i64 = row.get(0);
    assert_eq!(db_count, count as i64, "All vectors should be uploaded");

    cleanup(&mut conn);
}

#[test]
fn test_pgvector_knn_l2_search() {
    wait_for_postgres();
    let mut conn = connect();
    let dim = 8;
    let count = 100;
    let top = 10;
    setup_table(&mut conn, dim);

    let (ids, vectors) = generate_test_vectors(count, dim);

    // Upload with COPY
    {
        let mut writer = conn
            .copy_in("COPY items (id, embedding) FROM STDIN WITH (FORMAT text)")
            .unwrap();
        use std::io::Write;
        for i in 0..ids.len() {
            let vec_str: String = vectors[i]
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(writer, "{}\t[{}]", ids[i], vec_str).unwrap();
        }
        writer.finish().unwrap();
    }

    // Create HNSW index for L2
    conn.execute(
        "CREATE INDEX ON items USING hnsw(embedding vector_l2_ops) WITH (m = 16, ef_construction = 200)",
        &[],
    )
    .unwrap();

    // Search using pgvector::Vector parameterized query
    let query_vec = pgvector::Vector::from(vectors[0].clone());
    let rows = conn
        .query(
            &format!(
                "SELECT id FROM items ORDER BY embedding <-> $1 LIMIT {}",
                top
            ),
            &[&query_vec],
        )
        .unwrap();

    let result_ids: Vec<i64> = rows.iter().map(|r| r.get::<_, i32>(0) as i64).collect();
    assert!(!result_ids.is_empty(), "Search should return results");
    assert_eq!(
        result_ids[0], 0,
        "Query vector (id=0) should be its own nearest neighbor"
    );

    cleanup(&mut conn);
}

#[test]
fn test_pgvector_knn_cosine_search() {
    wait_for_postgres();
    let mut conn = connect();
    let dim = 4;
    let count = 20;
    setup_table(&mut conn, dim);

    let (ids, vectors) = generate_test_vectors(count, dim);

    {
        let mut writer = conn
            .copy_in("COPY items (id, embedding) FROM STDIN WITH (FORMAT text)")
            .unwrap();
        use std::io::Write;
        for i in 0..ids.len() {
            let vec_str: String = vectors[i]
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(writer, "{}\t[{}]", ids[i], vec_str).unwrap();
        }
        writer.finish().unwrap();
    }

    // Create cosine index
    conn.execute(
        "CREATE INDEX ON items USING hnsw(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200)",
        &[],
    )
    .unwrap();

    let query_vec = pgvector::Vector::from(vectors[0].clone());
    let rows = conn
        .query(
            "SELECT id FROM items ORDER BY embedding <=> $1 LIMIT 5",
            &[&query_vec],
        )
        .unwrap();

    let result_ids: Vec<i64> = rows.iter().map(|r| r.get::<_, i32>(0) as i64).collect();
    assert!(
        !result_ids.is_empty(),
        "Cosine search should return results"
    );
    assert_eq!(result_ids[0], 0, "Self should be top-1 for cosine");

    cleanup(&mut conn);
}

#[test]
fn test_pgvector_precision_l2() {
    wait_for_postgres();
    let mut conn = connect();
    let dim = 8;
    let count = 100;
    let top = 10;
    setup_table(&mut conn, dim);

    let (ids, vectors) = generate_test_vectors(count, dim);

    {
        let mut writer = conn
            .copy_in("COPY items (id, embedding) FROM STDIN WITH (FORMAT text)")
            .unwrap();
        use std::io::Write;
        for i in 0..ids.len() {
            let vec_str: String = vectors[i]
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(writer, "{}\t[{}]", ids[i], vec_str).unwrap();
        }
        writer.finish().unwrap();
    }

    conn.execute(
        "CREATE INDEX ON items USING hnsw(embedding vector_l2_ops) WITH (m = 16, ef_construction = 200)",
        &[],
    )
    .unwrap();

    // Set high ef_search for near-exact results
    conn.execute("SET hnsw.ef_search = 200", &[]).unwrap();

    let query_idx = 42;
    let expected = brute_force_neighbors_l2(&vectors[query_idx], &vectors, top);

    let query_vec = pgvector::Vector::from(vectors[query_idx].clone());
    let rows = conn
        .query(
            &format!(
                "SELECT id FROM items ORDER BY embedding <-> $1 LIMIT {}",
                top
            ),
            &[&query_vec],
        )
        .unwrap();

    let result_ids: Vec<i64> = rows.iter().map(|r| r.get::<_, i32>(0) as i64).collect();

    let expected_set: std::collections::HashSet<i64> = expected.into_iter().collect();
    let found_set: std::collections::HashSet<i64> = result_ids.into_iter().collect();
    let hits = expected_set.intersection(&found_set).count();
    let precision = hits as f64 / top as f64;

    assert!(
        precision >= 0.9,
        "L2 precision should be >= 0.9 for small dataset with high ef_search, got {}",
        precision
    );

    cleanup(&mut conn);
}

#[test]
fn test_pgvector_precision_cosine() {
    wait_for_postgres();
    let mut conn = connect();
    let dim = 8;
    let count = 100;
    let top = 10;
    setup_table(&mut conn, dim);

    let (ids, vectors) = generate_test_vectors(count, dim);

    {
        let mut writer = conn
            .copy_in("COPY items (id, embedding) FROM STDIN WITH (FORMAT text)")
            .unwrap();
        use std::io::Write;
        for i in 0..ids.len() {
            let vec_str: String = vectors[i]
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(writer, "{}\t[{}]", ids[i], vec_str).unwrap();
        }
        writer.finish().unwrap();
    }

    conn.execute(
        "CREATE INDEX ON items USING hnsw(embedding vector_cosine_ops) WITH (m = 16, ef_construction = 200)",
        &[],
    )
    .unwrap();

    conn.execute("SET hnsw.ef_search = 200", &[]).unwrap();

    let query_idx = 42;
    let expected = brute_force_neighbors_cosine(&vectors[query_idx], &vectors, top);

    let query_vec = pgvector::Vector::from(vectors[query_idx].clone());
    let rows = conn
        .query(
            &format!(
                "SELECT id FROM items ORDER BY embedding <=> $1 LIMIT {}",
                top
            ),
            &[&query_vec],
        )
        .unwrap();

    let result_ids: Vec<i64> = rows.iter().map(|r| r.get::<_, i32>(0) as i64).collect();

    let expected_set: std::collections::HashSet<i64> = expected.into_iter().collect();
    let found_set: std::collections::HashSet<i64> = result_ids.into_iter().collect();
    let hits = expected_set.intersection(&found_set).count();
    let precision = hits as f64 / top as f64;

    assert!(
        precision >= 0.9,
        "Cosine precision should be >= 0.9, got {}",
        precision
    );

    cleanup(&mut conn);
}

#[test]
fn test_pgvector_full_cycle() {
    wait_for_postgres();
    let mut conn = connect();
    let dim = 4;
    let count = 30;
    let top = 5;
    setup_table(&mut conn, dim);

    let (ids, vectors) = generate_test_vectors(count, dim);

    // 1. Upload
    {
        let mut writer = conn
            .copy_in("COPY items (id, embedding) FROM STDIN WITH (FORMAT text)")
            .unwrap();
        use std::io::Write;
        for i in 0..ids.len() {
            let vec_str: String = vectors[i]
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(",");
            writeln!(writer, "{}\t[{}]", ids[i], vec_str).unwrap();
        }
        writer.finish().unwrap();
    }

    // Verify count
    let row = conn.query_one("SELECT COUNT(*) FROM items", &[]).unwrap();
    let db_count: i64 = row.get(0);
    assert_eq!(db_count, count as i64);

    // 2. Create index
    conn.execute(
        "CREATE INDEX ON items USING hnsw(embedding vector_l2_ops) WITH (m = 16, ef_construction = 200)",
        &[],
    )
    .unwrap();

    // 3. Search
    let query_vec = pgvector::Vector::from(vectors[0].clone());
    let rows = conn
        .query(
            &format!(
                "SELECT id FROM items ORDER BY embedding <-> $1 LIMIT {}",
                top
            ),
            &[&query_vec],
        )
        .unwrap();
    assert_eq!(rows.len(), top, "Should return top-{} results", top);

    // 4. Delete
    conn.execute("DROP TABLE IF EXISTS items CASCADE", &[])
        .unwrap();

    // Verify deleted
    let row = conn
        .query_one(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'items'",
            &[],
        )
        .unwrap();
    let count: i64 = row.get(0);
    assert_eq!(count, 0, "items table should be deleted");
}

/// End-to-end `match_any`: filter a single-valued keyword field to an OR-set and
/// assert the engine returns the filtered nearest neighbours (recall vs ground
/// truth brute-forced over only the matching docs). Covers the keyword
/// contains-any arm on scalar data.
#[test]
fn test_binary_pgvector_match_any() {
    wait_for_postgres();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "pg-ma", "engine": "pgvector",
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
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
            "pg-ma",
            "match-any-test",
            "127.0.0.1",
            &[("PGVECTOR_PORT", "5433")],
        ),
        "pgvector match_any run failed"
    );

    let recall = common::read_recall(&proj.root, "pg-ma");
    println!("pgvector match_any recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "pgvector match_any recall {:.3} < 0.9",
        recall
    );
}

/// End-to-end coverage for the two `match_any` arms the scalar fixture can't
/// exercise: (1) **multi-valued** keyword docs, stored as a `;`-joined TEXT
/// scalar, which only match via the array-overlap clause (a scalar `IN` would
/// silently drop them), and (2) a **numeric** `match_any` on an int column.
/// Ground truth is brute-forced over the truly-matching docs, so a scalar-`IN`
/// regression on the keyword arm would score low recall and fail here.
#[test]
fn test_binary_pgvector_match_any_multivalue_and_numeric() {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::fs;
    use vector_db_benchmark::readers::write_npy_vectors;

    wait_for_postgres();

    let dim = 8usize;
    let n = 400usize;
    let top = 10usize;
    let n_q = 10usize; // queries per condition; averaging smooths HNSW recall variance
    let mut rng = StdRng::seed_from_u64(0xBEEF);
    let gen =
        |rng: &mut StdRng| -> Vec<f32> { (0..dim).map(|_| rng.gen_range(-1.0f32..1.0)).collect() };
    let vectors: Vec<Vec<f32>> = (0..n).map(|_| gen(&mut rng)).collect();
    let q_colors: Vec<Vec<f32>> = (0..n_q).map(|_| gen(&mut rng)).collect();
    let q_sizes: Vec<Vec<f32>> = (0..n_q).map(|_| gen(&mut rng)).collect();

    // Per-doc payloads: id%4==0 is MULTI-valued ["red","green"] (stored
    // "red;green"); others are single-valued. size = id%3.
    let color_json = |id: usize| -> serde_json::Value {
        match id % 4 {
            0 => serde_json::json!(["red", "green"]),
            1 => serde_json::json!("green"),
            2 => serde_json::json!("blue"),
            _ => serde_json::json!("yellow"),
        }
    };
    // Contains-any over {red, yellow}: id%4==0 (has "red") and id%4==3 ("yellow").
    // A scalar `IN ('red','yellow')` would MISS id%4==0 ('red;green').
    let color_matches = |id: usize| -> bool { id.is_multiple_of(4) || id % 4 == 3 };
    let size_matches = |id: usize| -> bool { id.is_multiple_of(3) || (id % 3) == 2 };

    let l2 = |a: &[f32], b: &[f32]| -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
            .sum()
    };
    let gt = |q: &[f32], pred: &dyn Fn(usize) -> bool| -> Vec<i64> {
        let mut s: Vec<(i64, f64)> = (0..n)
            .filter(|id| pred(*id))
            .map(|id| (id as i64, l2(q, &vectors[id])))
            .collect();
        s.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        s.iter().take(top).map(|(id, _)| *id).collect()
    };

    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path().to_path_buf();
    std::mem::forget(tmp);
    let ds = root.join("datasets").join("pg-mv");
    fs::create_dir_all(&ds).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();

    write_npy_vectors(ds.join("vectors.npy").to_str().unwrap(), &vectors).unwrap();
    let payloads: String = (0..n)
        .map(|id| serde_json::json!({"color": color_json(id), "size": (id % 3) as i64}).to_string())
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(ds.join("payloads.jsonl"), payloads).unwrap();

    let mut tests: Vec<serde_json::Value> = Vec::new();
    for q in &q_colors {
        tests.push(serde_json::json!({
            "query": q.iter().map(|x| *x as f64).collect::<Vec<_>>(),
            "conditions": {"and": [{"color": {"match": {"any": ["red", "yellow"]}}}]},
            "closest_ids": gt(q, &color_matches),
        }));
    }
    for q in &q_sizes {
        tests.push(serde_json::json!({
            "query": q.iter().map(|x| *x as f64).collect::<Vec<_>>(),
            "conditions": {"and": [{"size": {"match": {"any": [0, 2]}}}]},
            "closest_ids": gt(q, &size_matches),
        }));
    }
    fs::write(
        ds.join("tests.jsonl"),
        tests
            .iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join("\n"),
    )
    .unwrap();

    let datasets = serde_json::json!([{
        "name": "pg-mv", "type": "tar", "path": "pg-mv/",
        "distance": "l2", "vector_size": dim, "vector_count": n,
        "schema": {"color": "keyword", "size": "int"},
    }]);
    fs::write(
        root.join("datasets/datasets.json"),
        serde_json::to_string_pretty(&datasets).unwrap(),
    )
    .unwrap();
    let configs = serde_json::json!([{
        "name": "pg-mv", "engine": "pgvector",
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    fs::write(
        root.join("experiments/configurations/test.json"),
        serde_json::to_string(&configs).unwrap(),
    )
    .unwrap();

    // Sanity: enough matching docs for a top-10 query on each arm.
    assert!((0..n).filter(|id| color_matches(*id)).count() >= top);
    assert!((0..n).filter(|id| size_matches(*id)).count() >= top);

    assert!(
        common::run_binary(
            &root,
            "pg-mv",
            "pg-mv",
            "127.0.0.1",
            &[("PGVECTOR_PORT", "5433")]
        ),
        "pgvector multi-value/numeric match_any run failed"
    );
    let recall = common::read_recall(&root, "pg-mv");
    println!(
        "pgvector multi-value+numeric match_any recall={:.3}",
        recall
    );
    assert!(
        recall >= 0.9,
        "recall {:.3} < 0.9 (multi-value overlap or numeric IN regressed)",
        recall
    );
}
