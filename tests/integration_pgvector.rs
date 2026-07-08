//! Integration tests for the PgVector engine.
//!
//! Requires PostgreSQL with pgvector extension running on port 5433.
//! Start with: docker compose -f tests/docker-compose.test.yml up -d pgvector
//! Run with:   PGVECTOR_PORT=5433 cargo test --test integration_pgvector -- --test-threads=1

use std::thread;
use std::time::{Duration, Instant};

use rand::Rng;

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
