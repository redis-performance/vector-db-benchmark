//! PgVector engine implementation.
//!
//! Uses the `postgres` crate for PostgreSQL connectivity with the pgvector extension.
//! Supports HNSW index with configurable m/ef_construction, COPY bulk upload,
//! and distance operators <-> (L2) and <=> (cosine).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};

pub struct PgVectorEngine {
    name: String,
    host: String,
    port: u16,
    dbname: String,
    user: String,
    password: String,
    m: i64,
    ef_construction: i64,
    batch_size: usize,
    parallel: usize,
    search_params: Vec<SearchParams>,
    distance_op: String,
    hnsw_ops_class: String,
}

impl PgVectorEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("PGVECTOR_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(5432);

        let dbname = std::env::var("PGVECTOR_DB").unwrap_or_else(|_| "postgres".to_string());
        let user = std::env::var("PGVECTOR_USER").unwrap_or_else(|_| "postgres".to_string());
        let password = std::env::var("PGVECTOR_PASSWORD").unwrap_or_else(|_| "passwd".to_string());

        // Extract HNSW config
        let (m, ef_construction) = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.hnsw_config.as_ref())
            .map(|h| (h.m.unwrap_or(16), h.ef_construction.unwrap_or(128)))
            .unwrap_or((16, 128));

        let parallel = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("parallel"))
            .and_then(|v| v.as_i64())
            .unwrap_or(1) as usize;

        let batch_size = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("batch_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(1024) as usize;

        Ok(Self {
            name: engine_config.name.clone(),
            host: host.to_string(),
            port,
            dbname,
            user,
            password,
            m,
            ef_construction,
            batch_size,
            parallel,
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            // Default: will be set during configure based on dataset distance
            distance_op: String::new(),
            hnsw_ops_class: String::new(),
        })
    }

    fn connection_string(&self) -> String {
        format!(
            "host={} port={} dbname={} user={} password={}",
            self.host, self.port, self.dbname, self.user, self.password
        )
    }

    fn connect(&self) -> Result<postgres::Client, String> {
        postgres::Client::connect(&self.connection_string(), postgres::NoTls)
            .map_err(|e| format!("PostgreSQL connection failed: {}", e))
    }

    fn create_progress_bar(&self, total: usize) -> ProgressBar {
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec_int}/s)")
                .unwrap()
                .with_key("per_sec_int", |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                    write!(w, "{}", HumanCount(state.per_sec() as u64)).unwrap()
                })
                .progress_chars("#>-"),
        );
        pb
    }
}

impl Engine for PgVectorEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        let distance = dataset.distance();
        let vector_size = dataset.vector_size();

        let dist_lower = distance.to_lowercase();

        // Set distance operator and HNSW ops class
        match dist_lower.as_str() {
            "l2" | "euclidean" => {
                self.distance_op = "<->".to_string();
                self.hnsw_ops_class = "vector_l2_ops".to_string();
            }
            "cosine" | "angular" => {
                self.distance_op = "<=>".to_string();
                self.hnsw_ops_class = "vector_cosine_ops".to_string();
            }
            other => {
                return Err(format!(
                    "PgVector does not support distance metric: {}",
                    other
                ))
            }
        }

        let mut conn = self.connect()?;

        // Ensure vector extension exists
        println!("Creating vector extension...");
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector", &[])
            .map_err(|e| format!("Failed to create vector extension: {}", e))?;

        // Drop existing table
        println!("Dropping existing items table...");
        conn.execute("DROP TABLE IF EXISTS items CASCADE", &[])
            .map_err(|e| format!("Failed to drop table: {}", e))?;

        // Create table
        println!("Creating items table (vector dimension {})...", vector_size);
        let create_sql = format!(
            "CREATE TABLE items (id SERIAL PRIMARY KEY, embedding vector({}) NOT NULL)",
            vector_size
        );
        conn.execute(&create_sql, &[])
            .map_err(|e| format!("Failed to create table: {}", e))?;

        // Set storage to PLAIN for better performance
        conn.execute(
            "ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN",
            &[],
        )
        .map_err(|e| format!("Failed to alter storage: {}", e))?;

        // Create HNSW index
        println!(
            "Creating HNSW index (m={}, ef_construction={}, ops={})...",
            self.m, self.ef_construction, self.hnsw_ops_class
        );
        let index_sql = format!(
            "CREATE INDEX ON items USING hnsw(embedding {}) WITH (m = {}, ef_construction = {})",
            self.hnsw_ops_class, self.m, self.ef_construction
        );
        conn.execute(&index_sql, &[])
            .map_err(|e| format!("Failed to create HNSW index: {}", e))?;

        println!("PgVector configured successfully.");

        Ok(())
    }

    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        let normalize = dataset.needs_normalization();
        let dataset_path = dataset.get_path()?;
        println!("Reading dataset from {}...", dataset_path.display());
        let read_start = Instant::now();
        let (ids, vectors, _metadata) = dataset.read_vectors(normalize)?;
        let read_time = read_start.elapsed().as_secs_f64();

        println!(
            "Read {} vectors ({}d) in {:.3}s",
            vectors.len(),
            vectors.first().map(|v| v.len()).unwrap_or(0),
            read_time,
        );

        // PgVector uses COPY for bulk upload
        // Use batched INSERT for parallel upload since COPY requires exclusive connection
        let pb = self.create_progress_bar(ids.len());
        let upload_start = Instant::now();

        let batches: Vec<(usize, usize)> = (0..ids.len())
            .step_by(self.batch_size)
            .map(|start| (start, (start + self.batch_size).min(ids.len())))
            .collect();

        let total_batches = batches.len();
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let conn_str = self.connection_string();

        std::thread::scope(|s| {
            for _ in 0..self.parallel {
                let conn_str = conn_str.clone();
                let batches = &batches;
                let ids = &ids;
                let vectors = &vectors;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let mut conn = match postgres::Client::connect(&conn_str, postgres::NoTls) {
                        Ok(c) => c,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e.to_string());
                            return;
                        }
                    };

                    loop {
                        let idx = batch_idx.fetch_add(1, Ordering::SeqCst);
                        if idx >= total_batches {
                            break;
                        }
                        if error.lock().unwrap().is_some() {
                            break;
                        }

                        let (batch_start, batch_end) = batches[idx];

                        // Use COPY for bulk insert
                        let copy_result = (|| -> Result<(), String> {
                            let mut writer = conn
                                .copy_in("COPY items (id, embedding) FROM STDIN WITH (FORMAT text)")
                                .map_err(|e| format!("COPY start failed: {}", e))?;

                            use std::io::Write;
                            for i in batch_start..batch_end {
                                let vec_str: String = vectors[i]
                                    .iter()
                                    .map(|v| v.to_string())
                                    .collect::<Vec<_>>()
                                    .join(",");
                                writeln!(writer, "{}\t[{}]", ids[i], vec_str)
                                    .map_err(|e| format!("COPY write failed: {}", e))?;
                            }

                            writer
                                .finish()
                                .map_err(|e| format!("COPY finish failed: {}", e))?;
                            Ok(())
                        })();

                        if let Err(e) = copy_result {
                            *error.lock().unwrap() = Some(e);
                            break;
                        }

                        pb.inc((batch_end - batch_start) as u64);
                    }
                });
            }
        });

        pb.finish_with_message("Upload complete");

        if let Some(e) = error.lock().unwrap().take() {
            return Err(e);
        }

        let upload_time = upload_start.elapsed().as_secs_f64();
        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        let total_time = read_time + upload_time;

        Ok(UploadStats {
            upload_time,
            total_time,
            upload_count: vectors.len(),
            parallel: self.parallel,
            batch_size: self.batch_size,
            memory_usage: None,
        })
    }

    fn search(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        let parallel = params.parallel.unwrap_or(1) as usize;

        // Extract hnsw_ef from search params
        let hnsw_ef = params
            .search_params
            .as_ref()
            .and_then(|sp| {
                sp.ef.or_else(|| {
                    sp.extra
                        .as_ref()
                        .and_then(|e| e.get("hnsw_ef"))
                        .and_then(|v| v.as_i64())
                })
            })
            .unwrap_or(128);

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<String>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_pg_conditions))
            .collect();

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        let search_times: Arc<Mutex<Vec<f64>>> =
            Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let precisions: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let recalls: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let mrrs: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let ndcgs: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);
        let start_time = Instant::now();
        let conn_str = self.connection_string();
        let distance_op = self.distance_op.clone();

        std::thread::scope(|s| {
            for _ in 0..parallel {
                let conn_str = conn_str.clone();
                let distance_op = distance_op.clone();
                let queries = &queries;
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
                let search_times = Arc::clone(&search_times);
                let precisions = Arc::clone(&precisions);
                let recalls = Arc::clone(&recalls);
                let mrrs = Arc::clone(&mrrs);
                let ndcgs = Arc::clone(&ndcgs);
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                s.spawn(move || {
                    let mut conn = match postgres::Client::connect(&conn_str, postgres::NoTls) {
                        Ok(c) => c,
                        Err(_) => return,
                    };

                    // Set ef_search for this connection
                    let _ = conn.execute(
                        &format!("SET hnsw.ef_search = {}", hnsw_ef),
                        &[],
                    );

                    loop {
                        let idx = query_idx.fetch_add(1, Ordering::SeqCst);
                        if idx >= num_to_run {
                            break;
                        }

                        let top = explicit_top.unwrap_or_else(|| {
                            let n = neighbors[idx].len();
                            if n > 0 { n } else { 10 }
                        });

                        let query_vec = pgvector::Vector::from(queries[idx].clone());

                        let where_clause = parsed_filters[idx]
                            .as_deref()
                            .map(|f| format!(" WHERE {}", f))
                            .unwrap_or_default();

                        let query_sql = format!(
                            "SELECT id, embedding {} $1 AS _score FROM items{} ORDER BY _score LIMIT {}",
                            distance_op, where_clause, top
                        );

                        let query_start = Instant::now();
                        let results = conn.query(&query_sql, &[&query_vec]);
                        let query_time = query_start.elapsed().as_secs_f64();

                        match results {
                            Ok(rows) => {
                                search_times.lock().unwrap().push(query_time);
                                let ordered_ids: Vec<i64> = rows
                                    .iter()
                                    .map(|row| {
                                        let id: i32 = row.get(0);
                                        id as i64
                                    })
                                    .collect();
                                let m = crate::metrics::compute_metrics(
                                    &ordered_ids,
                                    &neighbors[idx],
                                    top,
                                );
                                precisions.lock().unwrap().push(m.precision);
                                recalls.lock().unwrap().push(m.recall);
                                mrrs.lock().unwrap().push(m.mrr);
                                ndcgs.lock().unwrap().push(m.ndcg);
                            }
                            Err(e) => {
                                eprintln!("Search query {} failed: {}", idx, e);
                            }
                        }
                        pb.inc(1);
                    }
                });
            }
        });

        pb.finish_and_clear();
        let total_time = start_time.elapsed().as_secs_f64();

        let times = search_times.lock().unwrap();
        let precs = precisions.lock().unwrap();
        let recs = recalls.lock().unwrap();
        let mrr_vals = mrrs.lock().unwrap();
        let ndcg_vals = ndcgs.lock().unwrap();

        if times.is_empty() {
            return Err("No searches completed".to_string());
        }

        let rps = times.len() as f64 / total_time;
        let mean_precision = precs.iter().sum::<f64>() / precs.len() as f64;
        let mean_recall = recs.iter().sum::<f64>() / recs.len() as f64;
        let mean_mrr = mrr_vals.iter().sum::<f64>() / mrr_vals.len() as f64;
        let mean_ndcg = ndcg_vals.iter().sum::<f64>() / ndcg_vals.len() as f64;
        let mean_time = times.iter().sum::<f64>() / times.len() as f64;
        let std_time = (times.iter().map(|t| (t - mean_time).powi(2)).sum::<f64>()
            / times.len() as f64)
            .sqrt();
        let min_time = times.iter().copied().fold(f64::INFINITY, f64::min);
        let max_time = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let mut sorted_times: Vec<f64> = times.clone();
        sorted_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50_idx = (sorted_times.len() as f64 * 0.50) as usize;
        let p95_idx = (sorted_times.len() as f64 * 0.95) as usize;
        let p99_idx = (sorted_times.len() as f64 * 0.99) as usize;

        let p50_time = sorted_times.get(p50_idx).copied().unwrap_or(0.0);
        let p95_time = sorted_times
            .get(p95_idx.min(sorted_times.len() - 1))
            .copied()
            .unwrap_or(0.0);
        let p99_time = sorted_times
            .get(p99_idx.min(sorted_times.len() - 1))
            .copied()
            .unwrap_or(0.0);

        Ok(SearchResults {
            total_time,
            mean_time,
            mean_precision,
            mean_recall,
            mean_mrr,
            mean_ndcg,
            std_time,
            min_time,
            max_time,
            rps,
            p50_time,
            p95_time,
            p99_time,
            precisions: precs.to_vec(),
            recalls: recs.to_vec(),
            mrrs: mrr_vals.to_vec(),
            ndcgs: ndcg_vals.to_vec(),
            latencies: times.to_vec(),
            top: explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10)),
            num_queries: times.len(),
            parallel,
            ..Default::default()
        })
    }

    fn delete(&mut self) -> Result<(), String> {
        let mut conn = self.connect()?;
        conn.execute("DROP TABLE IF EXISTS items CASCADE", &[])
            .map_err(|e| format!("Failed to drop table: {}", e))?;
        Ok(())
    }
}

// ── PgVector condition parser ─────────────────────────────────────

fn parse_pg_conditions(conditions: &serde_json::Value) -> Option<String> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    let mut clauses = Vec::new();

    if let Some(and_entries) = obj.get("and").and_then(|v| v.as_array()) {
        let sub: Vec<String> = and_entries.iter().filter_map(build_pg_clause).collect();
        if !sub.is_empty() {
            clauses.push(format!("({})", sub.join(" AND ")));
        }
    }

    if let Some(or_entries) = obj.get("or").and_then(|v| v.as_array()) {
        let sub: Vec<String> = or_entries.iter().filter_map(build_pg_clause).collect();
        if !sub.is_empty() {
            clauses.push(format!("({})", sub.join(" OR ")));
        }
    }

    if clauses.is_empty() {
        None
    } else {
        Some(clauses.join(" AND "))
    }
}

fn build_pg_clause(entry: &serde_json::Value) -> Option<String> {
    let entry_obj = entry.as_object()?;
    let mut parts = Vec::new();
    for (field_name, field_filters) in entry_obj {
        let filter_obj = field_filters.as_object()?;
        for (condition_type, criteria) in filter_obj {
            match condition_type.as_str() {
                "match" => {
                    if let Some(value) = criteria.get("value") {
                        if let Some(s) = value.as_str() {
                            parts.push(format!("{} = '{}'", field_name, s.replace('\'', "''")));
                        } else {
                            parts.push(format!("{} = {}", field_name, value));
                        }
                    }
                }
                "range" => {
                    if let Some(co) = criteria.as_object() {
                        for (op, val) in co {
                            let sql_op = match op.as_str() {
                                "lt" => "<",
                                "gt" => ">",
                                "lte" => "<=",
                                "gte" => ">=",
                                _ => continue,
                            };
                            if !val.is_null() {
                                parts.push(format!("{} {} {}", field_name, sql_op, val));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" AND "))
    }
}
