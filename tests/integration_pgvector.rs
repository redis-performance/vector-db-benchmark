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

/// Bool-field equality filter end-to-end. Regression for the missing-column bug:
/// `pg_column_type("bool")` returned None so no column was created and the filter
/// referenced a non-existent column (SQL error). With a BOOLEAN column, COPY
/// coerces the reader's "true"/"false" string and `flag = $1` selects the evens.
#[test]
fn test_binary_pgvector_bool() {
    wait_for_postgres();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "pg-bool", "engine": "pgvector",
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_bool_project("bool-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "pg-bool",
            "bool-test",
            "127.0.0.1",
            &[("PGVECTOR_PORT", "5433")]
        ),
        "pgvector bool run failed"
    );
    let recall = common::read_recall(&proj.root, "pg-bool");
    println!("pgvector bool recall={:.3}", recall);
    assert!(recall >= 0.9, "pgvector bool recall {:.3} < 0.9", recall);
}

/// Datetime range filter end-to-end. Regression for two bugs: no TIMESTAMPTZ
/// column was created, and the range builder inlined the ISO bound as a bare
/// (identifier-quoted) token → SQL error. With a TIMESTAMPTZ column and a
/// `$n::timestamptz` bind, the `[day 100, day 300)` window is selected.
#[test]
fn test_binary_pgvector_datetime() {
    wait_for_postgres();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "pg-dt", "engine": "pgvector",
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_datetime_project("dt-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "pg-dt",
            "dt-test",
            "127.0.0.1",
            &[("PGVECTOR_PORT", "5433")]
        ),
        "pgvector datetime run failed"
    );
    let recall = common::read_recall(&proj.root, "pg-dt");
    println!("pgvector datetime recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "pgvector datetime recall {:.3} < 0.9",
        recall
    );
}

/// Full-text filter end-to-end. Regression: a `{match:{text}}` clause was dropped
/// (the match arm only handled `value`/`any`), so the kNN query ran UNFILTERED
/// while recall was scored against the filtered ground truth. Now the TEXT column
/// is filtered with Postgres FTS (`to_tsvector @@ plainto_tsquery`), selecting the
/// docs whose body contains the query token.
#[test]
fn test_binary_pgvector_fulltext() {
    wait_for_postgres();
    let dim = 8;
    let configs = serde_json::json!([{
        "name": "pg-ft", "engine": "pgvector",
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
    }]);
    let proj =
        common::write_fulltext_project("ft-test", &serde_json::to_string(&configs).unwrap(), dim);
    assert!(proj.matching_docs >= proj.top);
    assert!(
        common::run_binary(
            &proj.root,
            "pg-ft",
            "ft-test",
            "127.0.0.1",
            &[("PGVECTOR_PORT", "5433")]
        ),
        "pgvector fulltext run failed"
    );
    let recall = common::read_recall(&proj.root, "pg-ft");
    println!("pgvector fulltext recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "pgvector fulltext recall {:.3} < 0.9",
        recall
    );
}

/// End-to-end `match_any` on a MULTI-VALUED keyword field (`labels`, #88). The
/// ';'-joined TEXT column is filtered with array-overlap (`match_any`) and set
/// membership (exact-match); this exercises the full binary path against a live
/// Postgres, complementing the direct-SQL overlap test.
#[test]
fn test_binary_pgvector_match_any_labels() {
    wait_for_postgres();

    let dim = 8;
    let configs = serde_json::json!([{
        "name": "pg-mal", "engine": "pgvector",
        "search_params": [{"parallel": 1, "search_params": {"hnsw_ef": 400}}],
        "upload_params": {"parallel": 1, "batch_size": 100}
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
            "pg-mal",
            "match-any-labels-test",
            "127.0.0.1",
            &[("PGVECTOR_PORT", "5433")],
        ),
        "pgvector match_any labels run failed"
    );

    let recall = common::read_recall(&proj.root, "pg-mal");
    println!("pgvector match_any labels recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "pgvector multi-valued labels match_any recall {:.3} < 0.9",
        recall
    );
}

/// Direct-SQL correctness for the keyword contains-any clause the `match_any`
/// filter builder emits. Multi-valued keyword payloads are stored as a
/// ';'-joined TEXT scalar, so the array-overlap must match a row that contains
/// ANY listed value — precisely the case a plain scalar `IN` (the pre-fix
/// behaviour) would silently drop. This runs the exact clause shape asserted by
/// the `match_any_string_list_emits_array_overlap` unit test against Postgres,
/// so together they cover the fix end-to-end without HNSW-recall flakiness.
#[test]
fn test_pgvector_match_any_overlap_matches_multivalue() {
    wait_for_postgres();
    let mut conn = connect();
    conn.execute("DROP TABLE IF EXISTS ma_overlap_test", &[])
        .unwrap();
    conn.execute(
        "CREATE TABLE ma_overlap_test (id INT PRIMARY KEY, color TEXT)",
        &[],
    )
    .unwrap();
    // id 1: multi-valued 'red;green' (matches {red,yellow} via 'red');
    // id 2: 'yellow' (matches); id 3: 'blue' (no); id 4: 'green' (no).
    conn.execute(
        "INSERT INTO ma_overlap_test VALUES (1,'red;green'),(2,'yellow'),(3,'blue'),(4,'green')",
        &[],
    )
    .unwrap();

    // Exactly the clause build_pg_clause emits for match_any ["red","yellow"].
    let overlap: Vec<i32> = conn
        .query(
            "SELECT id FROM ma_overlap_test \
             WHERE string_to_array(\"color\", ';') && ARRAY['red', 'yellow']::text[] \
             ORDER BY id",
            &[],
        )
        .unwrap()
        .iter()
        .map(|r| r.get::<_, i32>(0))
        .collect();
    assert_eq!(
        overlap,
        vec![1, 2],
        "array-overlap must match the multi-valued 'red;green' (via red) and 'yellow'"
    );

    // Sanity: the pre-fix scalar `IN` misses the multi-valued row, so this data
    // genuinely distinguishes the correct implementation from a regression.
    let scalar_in: Vec<i32> = conn
        .query(
            "SELECT id FROM ma_overlap_test WHERE \"color\" IN ('red','yellow') ORDER BY id",
            &[],
        )
        .unwrap()
        .iter()
        .map(|r| r.get::<_, i32>(0))
        .collect();
    assert_eq!(
        scalar_in,
        vec![2],
        "scalar IN drops the multi-valued row (confirms the test discriminates)"
    );

    conn.execute("DROP TABLE IF EXISTS ma_overlap_test", &[])
        .unwrap();
}

/// Direct-SQL correctness for the EXACT-MATCH clause on a multi-valued keyword
/// field (#88). `build_pg_clause` emits `$1 = ANY(string_to_array(col, ';'))`
/// for a `match.value` on `labels`; this must match a row whose ';'-joined set
/// CONTAINS the value — the case a scalar `col = $1` (the pre-fix behaviour)
/// silently drops. Complements the unit test that pins the SQL shape.
#[test]
fn test_pgvector_exact_match_labels_set_membership() {
    wait_for_postgres();
    let mut conn = connect();
    conn.execute("DROP TABLE IF EXISTS ma_exact_test", &[])
        .unwrap();
    conn.execute(
        "CREATE TABLE ma_exact_test (id INT PRIMARY KEY, labels TEXT)",
        &[],
    )
    .unwrap();
    // id 1: multi-valued 'red;green' (contains 'red'); id 2: 'red' (scalar);
    // id 3: 'blue' (no).
    conn.execute(
        "INSERT INTO ma_exact_test VALUES (1,'red;green'),(2,'red'),(3,'blue')",
        &[],
    )
    .unwrap();

    // Exactly the clause build_pg_clause emits for match.value "red" on labels.
    let member: Vec<i32> = conn
        .query(
            "SELECT id FROM ma_exact_test \
             WHERE $1 = ANY(string_to_array(\"labels\", ';')) ORDER BY id",
            &[&"red"],
        )
        .unwrap()
        .iter()
        .map(|r| r.get::<_, i32>(0))
        .collect();
    assert_eq!(
        member,
        vec![1, 2],
        "set-membership must match the multi-valued 'red;green' (via red) and scalar 'red'"
    );

    // Sanity: the pre-fix scalar `=` misses the multi-valued row.
    let scalar_eq: Vec<i32> = conn
        .query(
            "SELECT id FROM ma_exact_test WHERE \"labels\" = $1 ORDER BY id",
            &[&"red"],
        )
        .unwrap()
        .iter()
        .map(|r| r.get::<_, i32>(0))
        .collect();
    assert_eq!(
        scalar_eq,
        vec![2],
        "scalar = drops the multi-valued row (confirms the test discriminates)"
    );

    conn.execute("DROP TABLE IF EXISTS ma_exact_test", &[])
        .unwrap();
}
