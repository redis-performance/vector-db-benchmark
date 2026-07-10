//! Valkey engine implementation.
//!
//! Implements the Engine trait for Valkey Search vector similarity.
//! Valkey is a Redis fork that speaks the same RESP protocol and supports
//! FT.* search commands via the Valkey Search module.
//!
//! # Why `redis` crate instead of Valkey GLIDE?
//!
//! | Option              | Status                                          |
//! |---------------------|-------------------------------------------------|
//! | `valkey-glide` Rust | No published crate. GitHub issue                |
//! |                     | valkey-io/valkey-glide#828 closed NOT_PLANNED.   |
//! |                     | Supported langs: Java, Python, Node.js, Go.     |
//! |                     | Rust is not on the roadmap.                      |
//! | `redis` crate       | Recommended by GLIDE maintainers for Rust.       |
//! |                     | GLIDE team upstreams improvements to redis-rs.   |
//! |                     | Works with Valkey via RESP protocol compat.      |
//!
//! Reference: <https://github.com/valkey-io/valkey-glide/issues/828>

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::SeedableRng;

use super::redis_utils;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use redis::Connection;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UpdateSearchRatio, UploadStats};
use vector_db_benchmark::readers::metadata::{MetadataItem, MetadataValue};

/// Valkey engine configuration
#[derive(Clone)]
pub struct ValkeyEngineConfig {
    pub m: i64,
    pub ef_construction: i64,
    pub data_type: String,
    pub algorithm: String,
    pub batch_size: usize,
    pub parallel: usize,
    pub skip_vector_index: bool,
    /// Schema fields declared `datetime`: their ISO-8601 payload values are
    /// converted to epoch seconds at HSET time so the NUMERIC index/range
    /// filters match. Populated in `configure()`. `Arc` so per-thread config
    /// clones share one set.
    pub datetime_fields: Arc<HashSet<String>>,
    /// Schema fields declared `text`. Valkey Search has no TEXT field type, so
    /// full-text is DEGRADED to a whitespace-tokenised multi-value TAG: on upload
    /// the text is split on whitespace and stored as a `;`-joined TAG set, and a
    /// `{"match":{"text":"word"}}` query becomes a single-token TAG match. This
    /// supports single-term full-text matching (any doc containing the term); it
    /// does NOT support phrase / multi-term full-text queries.
    pub text_tag_fields: Arc<HashSet<String>>,
}

pub struct ValkeyEngine {
    name: String,
    host: String,
    port: u16,
    config: ValkeyEngineConfig,
    search_params: Vec<SearchParams>,
    commandstats_baseline: Option<redis_utils::CommandStatsBaseline>,
}

impl ValkeyEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("VALKEY_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6379);

        // Extract HNSW config
        let (m, ef_construction) = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.hnsw_config.as_ref())
            .map(|h| (h.m.unwrap_or(16), h.ef_construction.unwrap_or(128)))
            .unwrap_or((16, 128));

        let algorithm = engine_config
            .algorithm
            .clone()
            .unwrap_or_else(|| "hnsw".to_string());

        let data_type = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("data_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("FLOAT32")
            .to_string();

        let parallel = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("parallel"))
            .and_then(|v| v.as_i64())
            .unwrap_or(100) as usize;

        let batch_size = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("batch_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(64) as usize;

        Ok(Self {
            name: engine_config.name.clone(),
            host: host.to_string(),
            port,
            config: ValkeyEngineConfig {
                m,
                ef_construction,
                data_type,
                algorithm,
                batch_size,
                parallel,
                skip_vector_index: engine_config.skip_vector_index,
                datetime_fields: Arc::new(HashSet::new()),
                text_tag_fields: Arc::new(HashSet::new()),
            },
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            commandstats_baseline: None,
        })
    }

    fn get_connection(&self) -> Result<Connection, String> {
        Self::connect(&self.host, self.port)
    }

    fn connect(host: &str, port: u16) -> Result<Connection, String> {
        let auth = std::env::var("VALKEY_AUTH").ok();
        let user = std::env::var("VALKEY_USER").ok();

        let auth_part = match (&user, &auth) {
            (Some(u), Some(p)) => format!("{}:{}@", u, p),
            (None, Some(p)) => format!(":{}@", p),
            _ => String::new(),
        };

        let url = format!(
            "redis://{}{}:{}/{}",
            auth_part,
            host,
            port,
            valkey_url_suffix()
        );
        let client = redis::Client::open(url.as_str()).map_err(|e| e.to_string())?;
        let conn = client.get_connection().map_err(|e| e.to_string())?;
        // Safety timeout: prevents indefinite hangs from pipeline stalls.
        let timeout = std::time::Duration::from_secs(300);
        conn.set_read_timeout(Some(timeout)).ok();
        conn.set_write_timeout(Some(timeout)).ok();
        Ok(conn)
    }

    fn create_index(&self, conn: &mut Connection, dataset: &Dataset) -> Result<(), String> {
        let distance = dataset.distance();
        let vector_size = dataset.vector_size();

        // Drop existing index if any (Valkey Search does not support DD flag)
        let _ = redis::cmd("FT.DROPINDEX").arg("idx").query::<()>(conn);
        // Flush all keys to clean up data from previous runs
        let _ = redis::cmd("FLUSHALL").query::<()>(conn);

        // Map distance metric
        let distance_metric = match distance.to_lowercase().as_str() {
            "cosine" | "angular" => "COSINE",
            "euclidean" | "l2" => "L2",
            "dot" | "ip" => "IP",
            _ => "COSINE",
        };

        // Build FT.CREATE command
        let mut cmd = redis::cmd("FT.CREATE");
        cmd.arg("idx")
            .arg("ON")
            .arg("HASH")
            .arg("PREFIX")
            .arg("1")
            .arg("");

        cmd.arg("SCHEMA");

        // Vector field with HNSW params
        // Skipped when skip_vector_index is set (filter-only benchmark)
        if !self.config.skip_vector_index {
            let num_attrs = 6 + 2 + 2; // TYPE+DIM+DISTANCE_METRIC + M + EF_CONSTRUCTION
            cmd.arg("vector")
                .arg("VECTOR")
                .arg(self.config.algorithm.to_uppercase())
                .arg(num_attrs);
            cmd.arg("TYPE").arg(&self.config.data_type);
            cmd.arg("DIM").arg(vector_size);
            cmd.arg("DISTANCE_METRIC").arg(distance_metric);
            cmd.arg("M").arg(self.config.m);
            cmd.arg("EF_CONSTRUCTION").arg(self.config.ef_construction);
        }

        // Add schema fields from dataset config for filtering.
        // Note: Valkey Search does not support SORTABLE, TEXT, or GEO field types.
        // Only TAG and NUMERIC are supported as filter fields.
        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                for (field_name, field_type) in schema_obj {
                    let ft = field_type.as_str().unwrap_or("");
                    match ft {
                        // keyword/uuid/bool are exact-match TAG fields.
                        "keyword" | "uuid" | "bool" => {
                            cmd.arg(field_name).arg("TAG").arg("SEPARATOR").arg(";");
                        }
                        "int" | "float" => {
                            cmd.arg(field_name).arg("NUMERIC");
                        }
                        // datetime is stored as epoch seconds (see upload) → NUMERIC.
                        "datetime" => {
                            cmd.arg(field_name).arg("NUMERIC");
                        }
                        // Valkey Search has no TEXT field type: DEGRADE full-text
                        // to a whitespace-tokenised multi-value TAG (see
                        // ValkeyEngineConfig::text_tag_fields). Single-term
                        // full-text matching works; phrase queries do not.
                        "text" => {
                            cmd.arg(field_name).arg("TAG").arg("SEPARATOR").arg(";");
                        }
                        // Valkey Search does not support GEO; skip silently.
                        _ => {}
                    }
                }
            }
        }

        cmd.query::<()>(conn)
            .map_err(|e| format!("Failed to create index: {}", e))?;

        Ok(())
    }

    fn upload_sequential(
        &self,
        ids: &[i64],
        vectors: &[Vec<f32>],
        metadata: &[Option<MetadataItem>],
    ) -> Result<(), String> {
        let mut conn = self.get_connection()?;
        let pb = self.create_progress_bar(ids.len());

        for batch_start in (0..ids.len()).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(ids.len());
            self.upload_batch(
                &mut conn,
                &ids[batch_start..batch_end],
                &vectors[batch_start..batch_end],
                &metadata[batch_start..batch_end],
            )?;
            pb.inc((batch_end - batch_start) as u64);
        }

        pb.finish_with_message("Upload complete");
        Ok(())
    }

    fn upload_parallel(
        &self,
        ids: &[i64],
        vectors: &[Vec<f32>],
        metadata: &[Option<MetadataItem>],
    ) -> Result<(), String> {
        let pb = self.create_progress_bar(ids.len());
        let batches: Vec<(usize, usize)> = (0..ids.len())
            .step_by(self.config.batch_size)
            .map(|start| (start, (start + self.config.batch_size).min(ids.len())))
            .collect();

        let total_batches = batches.len();
        let num_threads = self.config.parallel.min(total_batches);
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                let host = self.host.clone();
                let port = self.port;
                let config = self.config.clone();
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let mut conn = match ValkeyEngine::connect(&host, port) {
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
                        let (batch_start, batch_end) = batches[idx];
                        if let Err(e) = upload_batch_internal(
                            &mut conn,
                            &config,
                            &ids[batch_start..batch_end],
                            &vectors[batch_start..batch_end],
                            &metadata[batch_start..batch_end],
                        ) {
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
        Ok(())
    }

    fn upload_batch(
        &self,
        conn: &mut Connection,
        ids: &[i64],
        vectors: &[Vec<f32>],
        metadata: &[Option<MetadataItem>],
    ) -> Result<(), String> {
        upload_batch_internal(conn, &self.config, ids, vectors, metadata)
    }

    /// Wait until FT.INFO reports num_docs >= expected and indexing/backfill is done.
    ///
    /// Checks both Redis Search's `indexing` flag and Valkey Search's
    /// `backfill_in_progress` / `state` fields so this works with either engine.
    fn wait_for_indexing(&self, expected: usize) -> Result<(), String> {
        let mut conn = self.get_connection()?;
        let max_wait = 600; // seconds – large HNSW indices can take minutes
        let start = Instant::now();

        loop {
            let info: redis::Value = redis::cmd("FT.INFO")
                .arg("idx")
                .query(&mut conn)
                .map_err(|e| format!("FT.INFO error: {}", e))?;

            let mut num_docs: usize = 0;
            let mut indexing: bool = false;

            fn extract_usize(val: &redis::Value) -> usize {
                match val {
                    redis::Value::BulkString(s) => String::from_utf8_lossy(s).parse().unwrap_or(0),
                    redis::Value::Int(n) => *n as usize,
                    redis::Value::Double(f) => *f as usize,
                    redis::Value::SimpleString(s) => s.parse().unwrap_or(0),
                    _ => 0,
                }
            }

            fn extract_bool_nonzero(val: &redis::Value) -> bool {
                match val {
                    redis::Value::BulkString(s) => s != b"0",
                    redis::Value::Int(n) => *n != 0,
                    redis::Value::Double(f) => *f != 0.0,
                    redis::Value::SimpleString(s) => s != "0",
                    redis::Value::Boolean(b) => *b,
                    _ => false,
                }
            }

            fn extract_string(val: &redis::Value) -> String {
                match val {
                    redis::Value::BulkString(s) => String::from_utf8_lossy(s).to_string(),
                    redis::Value::SimpleString(s) => s.clone(),
                    _ => String::new(),
                }
            }

            match &info {
                redis::Value::Array(arr) => {
                    for i in (0..arr.len()).step_by(2) {
                        let key_str = match &arr[i] {
                            redis::Value::BulkString(s) => String::from_utf8_lossy(s).to_string(),
                            redis::Value::SimpleString(s) => s.clone(),
                            _ => continue,
                        };
                        if let Some(val) = arr.get(i + 1) {
                            match key_str.as_str() {
                                "num_docs" => num_docs = extract_usize(val),
                                // Redis Search field
                                "indexing" => indexing = indexing || extract_bool_nonzero(val),
                                // Valkey Search fields
                                "backfill_in_progress" => {
                                    indexing = indexing || extract_bool_nonzero(val)
                                }
                                "state" => {
                                    let state = extract_string(val);
                                    if state != "ready" && !state.is_empty() {
                                        indexing = true;
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                redis::Value::Map(map) => {
                    for (k, v) in map {
                        let key_str = match k {
                            redis::Value::BulkString(s) => String::from_utf8_lossy(s).to_string(),
                            redis::Value::SimpleString(s) => s.clone(),
                            _ => continue,
                        };
                        match key_str.as_str() {
                            "num_docs" => num_docs = extract_usize(v),
                            "indexing" => indexing = indexing || extract_bool_nonzero(v),
                            "backfill_in_progress" => {
                                indexing = indexing || extract_bool_nonzero(v)
                            }
                            "state" => {
                                let state = extract_string(v);
                                if state != "ready" && !state.is_empty() {
                                    indexing = true;
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {
                    eprintln!("Unexpected FT.INFO response type: {:?}", info);
                }
            }

            if num_docs >= expected && !indexing {
                println!(
                    "Indexing complete: {} docs in {:.1}s",
                    num_docs,
                    start.elapsed().as_secs_f64()
                );
                return Ok(());
            }

            if start.elapsed().as_secs() > max_wait {
                println!(
                    "Warning: indexing timeout after {}s (num_docs={}/{}, indexing={})",
                    max_wait, num_docs, expected, indexing
                );
                return Ok(());
            }

            if start.elapsed().as_secs().is_multiple_of(10) && start.elapsed().as_secs() > 0 {
                println!(
                    "Waiting for indexing: {} docs, indexing={} ({:.0}s)",
                    num_docs,
                    indexing,
                    start.elapsed().as_secs_f64()
                );
            }

            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }

    /// Filter-only search: run FT.SEARCH with filter conditions only (no KNN).
    fn search_filter_only(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        let parallel = params.parallel.unwrap_or(1) as usize;
        let query_timeout: i64 = std::env::var("VALKEY_QUERY_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60_000);

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (_queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<ParsedFilter>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_conditions))
            .collect();

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);

        let runnable_indices: Vec<usize> = (0..parsed_filters.len())
            .filter(|&i| parsed_filters[i].is_some())
            .collect();

        if runnable_indices.is_empty() {
            return Err("No queries with filter conditions for filter-only search".to_string());
        }

        // Round-robin: if num_queries > available queries, cycle through them
        let num_to_run = if num_queries > 0 {
            num_queries as usize
        } else {
            runnable_indices.len()
        };

        let search_times: Arc<Mutex<Vec<f64>>> =
            Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);
        let start_time = Instant::now();

        std::thread::scope(|s| {
            for _ in 0..parallel {
                let host = self.host.clone();
                let port = self.port;
                let parsed_filters = &parsed_filters;
                let runnable_indices = &runnable_indices;
                let neighbors = &neighbors;
                let search_times = Arc::clone(&search_times);
                let errors = Arc::clone(&errors);
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                s.spawn(move || {
                    let auth = std::env::var("VALKEY_AUTH").ok();
                    let user = std::env::var("VALKEY_USER").ok();
                    let auth_part = match (&user, &auth) {
                        (Some(u), Some(p)) => format!("{}:{}@", u, p),
                        (None, Some(p)) => format!(":{}@", p),
                        _ => String::new(),
                    };
                    let url = format!(
                        "redis://{}{}:{}/{}",
                        auth_part,
                        host,
                        port,
                        valkey_url_suffix()
                    );
                    let client = match redis::Client::open(url.as_str()) {
                        Ok(c) => c,
                        Err(_) => return,
                    };
                    let mut conn = match client.get_connection() {
                        Ok(c) => c,
                        Err(_) => return,
                    };

                    loop {
                        let seq = query_idx.fetch_add(1, Ordering::SeqCst);
                        if seq >= num_to_run {
                            break;
                        }
                        let idx = runnable_indices[seq % runnable_indices.len()];

                        let top = explicit_top.unwrap_or_else(|| {
                            let n = neighbors[idx].len();
                            if n > 0 {
                                n
                            } else {
                                10
                            }
                        });

                        let query_start = Instant::now();
                        let result = ft_search_filter_only(
                            &mut conn,
                            top,
                            query_timeout,
                            parsed_filters[idx].as_ref().unwrap(),
                        );
                        let query_time = query_start.elapsed().as_secs_f64();

                        if let Err(e) = result {
                            let mut errs = errors.lock().unwrap();
                            if errs.len() < 3 {
                                errs.push(e);
                            }
                        }

                        search_times.lock().unwrap().push(query_time);
                        pb.inc(1);
                    }
                });
            }
        });

        let logged_errors = errors.lock().unwrap();
        if !logged_errors.is_empty() {
            for e in logged_errors.iter() {
                eprintln!("\tFilter-only search error: {}", e);
            }
        }

        pb.finish_and_clear();
        let total_time = start_time.elapsed().as_secs_f64();

        let times = search_times.lock().unwrap();
        if times.is_empty() {
            return Err("No filter-only searches completed".to_string());
        }

        let rps = times.len() as f64 / total_time;
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

        let mut check_conn = self.get_connection()?;
        redis_utils::check_commandstats(
            &mut check_conn,
            &["FT.SEARCH"],
            "search",
            self.commandstats_baseline.as_ref(),
        )?;

        Ok(SearchResults {
            total_time,
            mean_time,
            mean_precision: -1.0,
            std_time,
            min_time,
            max_time,
            rps,
            p50_time,
            p95_time,
            p99_time,
            precisions: vec![],
            latencies: times.to_vec(),
            top: 0,
            num_queries: times.len(),
            parallel,
            ..Default::default()
        })
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

/// Maximum RESP wire bytes per pipeline flush.
/// Valkey Search HNSW indexing can stall pipelines when the payload fills
/// the TCP send buffer (default 16 KB on Linux). With concurrent upload
/// threads the server must interleave reads across connections, amplifying
/// the effect. Keeping each sub-batch well below the TCP buffer prevents
/// blocking writes while still amortising round-trip overhead.
const MAX_PIPE_BYTES: usize = 4_096;

/// Internal batch upload function.
///
/// Sends HSET commands in sub-batched pipelines whose total serialised size
/// stays under `MAX_PIPE_BYTES`. This avoids a known interaction between
/// the Rust `redis` crate's synchronous pipeline writer and Valkey Search's
/// HNSW indexing that can stall large single-write pipelines.
fn upload_batch_internal(
    conn: &mut Connection,
    config: &ValkeyEngineConfig,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
) -> Result<(), String> {
    let mut pipe = redis::pipe();
    let mut pipe_bytes: usize = 0;

    for i in 0..ids.len() {
        let key = ids[i].to_string();
        let vec_bytes: Vec<u8> = match config.data_type.as_str() {
            "FLOAT64" => vectors[i]
                .iter()
                .map(|&f| f as f64)
                .flat_map(|f| f.to_le_bytes())
                .collect(),
            "FLOAT16" => vectors[i]
                .iter()
                .map(|&f| half::f16::from_f32(f).to_bits())
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            "BFLOAT16" => vectors[i]
                .iter()
                .map(|&f| half::bf16::from_f32(f).to_bits())
                .flat_map(|v| v.to_le_bytes())
                .collect(),
            _ => vectors[i].iter().flat_map(|f| f.to_le_bytes()).collect(),
        };

        let mut fields: Vec<(Vec<u8>, Vec<u8>)> = vec![("vector".as_bytes().to_vec(), vec_bytes)];

        if let Some(meta) = &metadata[i] {
            for (k, v) in &meta.fields {
                match v {
                    MetadataValue::String(s) => {
                        fields.push((
                            k.as_bytes().to_vec(),
                            encode_string_field(config, k, s).into_bytes(),
                        ));
                    }
                    MetadataValue::Labels(labels) => {
                        fields.push((k.as_bytes().to_vec(), labels.join(";").into_bytes()));
                    }
                    MetadataValue::Geo { lon, lat } => {
                        let lat_clamped = lat.clamp(-85.05112878, 85.05112878);
                        let geo_str = format!("{},{}", lon, lat_clamped);
                        fields.push((k.as_bytes().to_vec(), geo_str.into_bytes()));
                    }
                }
            }
        }

        // Estimate RESP wire size: *N\r\n + $4\r\nHSET\r\n + $K\r\nkey\r\n + fields
        let num_args = 1 + 1 + fields.len() * 2; // HSET + key + (field_name, field_value)*
        let mut cmd_bytes = format!("*{}\r\n", num_args).len()
            + 10 // $4\r\nHSET\r\n
            + format!("${}\r\n", key.len()).len() + key.len() + 2;
        for (fk, fv) in &fields {
            cmd_bytes += format!("${}\r\n", fk.len()).len() + fk.len() + 2;
            cmd_bytes += format!("${}\r\n", fv.len()).len() + fv.len() + 2;
        }

        // Flush the current pipeline if adding this command would exceed the limit
        if pipe_bytes > 0 && pipe_bytes + cmd_bytes > MAX_PIPE_BYTES {
            pipe.query::<()>(conn).map_err(|e| e.to_string())?;
            pipe = redis::pipe();
            pipe_bytes = 0;
        }

        let mut hset_cmd = redis::cmd("HSET");
        hset_cmd.arg(key.as_str());
        for (field_key, field_val) in &fields {
            hset_cmd.arg(&field_key[..]).arg(&field_val[..]);
        }
        pipe.add_command(hset_cmd).ignore();
        pipe_bytes += cmd_bytes;
    }

    // Flush remaining commands
    if pipe_bytes > 0 {
        pipe.query::<()>(conn).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Convert a redis::Value to serde_json::Value for serialization.
fn redis_value_to_json(val: &redis::Value) -> serde_json::Value {
    match val {
        redis::Value::Nil => serde_json::Value::Null,
        redis::Value::Int(n) => serde_json::json!(n),
        redis::Value::Double(f) => serde_json::json!(f),
        redis::Value::Boolean(b) => serde_json::json!(b),
        redis::Value::SimpleString(s) => serde_json::json!(s),
        redis::Value::BulkString(bytes) => match String::from_utf8(bytes.clone()) {
            Ok(s) => serde_json::json!(s),
            Err(_) => serde_json::json!(format!("<{} bytes>", bytes.len())),
        },
        redis::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(redis_value_to_json).collect())
        }
        redis::Value::Map(pairs) => {
            let mut map = serde_json::Map::new();
            for (k, v) in pairs {
                let key = match k {
                    redis::Value::SimpleString(s) => s.clone(),
                    redis::Value::BulkString(b) => String::from_utf8_lossy(b).to_string(),
                    other => format!("{:?}", other),
                };
                map.insert(key, redis_value_to_json(v));
            }
            serde_json::Value::Object(map)
        }
        other => serde_json::json!(format!("{:?}", other)),
    }
}

// ── Condition parser ─────────────────────────────────────────────────────
// Converts JSON filter conditions into Valkey Search query filter syntax.
// Note: Valkey Search does not support $param inside TAG {…} brackets,
// so TAG values are inlined directly (with escaping). Numeric and geo
// filters still use parameterised PARAMS.

#[derive(Debug, Clone)]
enum FilterParamValue {
    Int(i64),
    Float(f64),
}

type ParsedFilter = (String, HashMap<String, FilterParamValue>);

fn parse_conditions(conditions: &serde_json::Value) -> Option<ParsedFilter> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    let mut counter: usize = 0;

    let and_entries = obj.get("and").and_then(|v| v.as_array());
    let or_entries = obj.get("or").and_then(|v| v.as_array());

    let and_subfilters = and_entries.map(|entries| build_subfilters(entries, &mut counter));
    let or_subfilters = or_entries.map(|entries| build_subfilters(entries, &mut counter));

    build_condition(and_subfilters, or_subfilters)
}

fn build_subfilters(entries: &[serde_json::Value], counter: &mut usize) -> Vec<ParsedFilter> {
    let mut filters = Vec::new();
    for entry in entries {
        if let Some(entry_obj) = entry.as_object() {
            for (field_name, field_filters) in entry_obj {
                if let Some(filter_obj) = field_filters.as_object() {
                    for (condition_type, criteria) in filter_obj {
                        if let Some(f) = build_filter(field_name, condition_type, criteria, counter)
                        {
                            filters.push(f);
                        }
                    }
                }
            }
        }
    }
    filters
}

fn build_filter(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
    counter: &mut usize,
) -> Option<ParsedFilter> {
    match condition_type {
        "match" => {
            // match_any (IN-list) > exact {value} > full-text {text}.
            if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                Some(build_match_any_filter(field_name, any, counter))
            } else if let Some(text) = criteria.get("text").and_then(|v| v.as_str()) {
                Some(build_text_filter(field_name, text))
            } else {
                build_exact_match_filter(field_name, criteria, counter)
            }
        }
        "range" => build_range_filter(field_name, criteria, counter),
        "geo" => build_geo_filter(field_name, criteria, counter),
        _ => None,
    }
}

/// Escape the TAG-structural characters when inlining a value into a `{…}`
/// clause (Valkey Search does not accept `$param` refs inside TAG brackets, so
/// values are inlined). Only the characters that would otherwise break the OR
/// (`|`) or the braces are escaped; other characters are passed raw, matching
/// the exact-match path.
fn escape_tag_value(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace('{', "\\{")
        .replace('}', "\\}")
}

/// Build a `match_any` (IN-list) filter, the OR-of-values semantics that mirror
/// qdrant's `Condition::matches(field, Vec)`.
///
/// - All-integer list -> NUMERIC OR `(@f:[$a $a] | @f:[$b $b])` (params).
/// - Otherwise -> TAG OR `@f:{a | b}` over the non-empty string values, inlined
///   (Valkey Search rejects `$param` inside TAG `{…}`). Empty-string tokens are
///   dropped (invalid TAG syntax).
/// - Empty / no representable values -> a never-match `(@f:{s} -@f:{s})`
///   contradiction so an empty IN-set matches NOTHING rather than being dropped
///   (which, as the sole clause, would run kNN over ALL docs). Assumes a TAG
///   field, the realistic case for a keyword IN-list.
///
/// NOTE: Valkey Search TAG matching is case-INSENSITIVE, whereas qdrant keyword
/// match is case-sensitive; all shipped keyword datasets use consistent casing.
fn build_match_any_filter(
    field_name: &str,
    any: &[serde_json::Value],
    counter: &mut usize,
) -> ParsedFilter {
    let mut params = HashMap::new();

    if !any.is_empty() && any.iter().all(|v| v.is_i64()) {
        let clauses: Vec<String> = any
            .iter()
            .filter_map(|v| v.as_i64())
            .map(|i| {
                let p = format!("{}_{}", field_name, counter);
                *counter += 1;
                params.insert(p.clone(), FilterParamValue::Int(i));
                format!("@{}:[${} ${}]", field_name, p, p)
            })
            .collect();
        return (format!("({})", clauses.join(" | ")), params);
    }

    let tokens: Vec<String> = any
        .iter()
        .filter_map(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(escape_tag_value)
        .collect();

    if tokens.is_empty() {
        let never = "__match_any_never_match__";
        return (
            format!("(@{0}:{{{1}}} -@{0}:{{{1}}})", field_name, never),
            params,
        );
    }

    (
        format!("@{}:{{{}}}", field_name, tokens.join(" | ")),
        params,
    )
}

fn build_exact_match_filter(
    field_name: &str,
    criteria: &serde_json::Value,
    counter: &mut usize,
) -> Option<ParsedFilter> {
    let value = criteria.get("value")?;
    let param_name = format!("{}_{}", field_name, counter);
    *counter += 1;

    let mut params = HashMap::new();

    // bool → inlined TAG match on the literal "true"/"false" token. Checked
    // before numeric arms (serde bools are neither i64 nor f64 nor str).
    if let Some(b) = value.as_bool() {
        let token = if b { "true" } else { "false" };
        return Some((format!("@{}:{{{}}}", field_name, token), params));
    }

    if let Some(s) = value.as_str() {
        // Valkey Search does not support $param references inside TAG {…}
        // brackets, and does NOT require backslash-escaping of spaces, dots,
        // or other special characters. Values are passed raw inside {…}.
        // (uuid strings are handled here too — a uuid is just a keyword.)
        Some((format!("@{}:{{{}}}", field_name, s), params))
    } else if let Some(i) = value.as_i64() {
        params.insert(param_name.clone(), FilterParamValue::Int(i));
        Some((
            format!("@{}:[${} ${}]", field_name, param_name, param_name),
            params,
        ))
    } else if let Some(f) = value.as_f64() {
        params.insert(param_name.clone(), FilterParamValue::Float(f));
        Some((
            format!("@{}:[${} ${}]", field_name, param_name, param_name),
            params,
        ))
    } else {
        None
    }
}

/// Build a full-text filter — DEGRADED on Valkey Search (no TEXT field type) to
/// a single-token TAG match. The `text` field is stored as a whitespace-
/// tokenised TAG multi-value on upload (see `tokenize_text_to_tag`), so a
/// single-term query `@field:{term}` matches any doc whose text contains that
/// term. The term is inlined (Valkey rejects `$param` inside TAG `{…}`) and its
/// TAG-structural characters escaped. A blank query degrades to a never-match
/// contradiction rather than an empty clause (which would run kNN over ALL docs).
///
/// LIMITATION: only single-term matching is supported; a multi-word `text` query
/// is treated as one TAG token and will generally not match (phrase / multi-term
/// full-text is not available on Valkey Search).
fn build_text_filter(field_name: &str, text: &str) -> ParsedFilter {
    let params = HashMap::new();
    let trimmed = text.trim();
    if trimmed.is_empty() {
        let never = "__text_never_match__";
        return (
            format!("(@{0}:{{{1}}} -@{0}:{{{1}}})", field_name, never),
            params,
        );
    }
    // Use the first whitespace token; escape TAG-structural chars.
    let term = trimmed.split_whitespace().next().unwrap_or(trimmed);
    (
        format!("@{}:{{{}}}", field_name, escape_tag_value(term)),
        params,
    )
}

fn build_range_filter(
    field_name: &str,
    criteria: &serde_json::Value,
    counter: &mut usize,
) -> Option<ParsedFilter> {
    // Bump the counter for param-name parity with sibling filters even though the
    // bounds are INLINED: Valkey Search does not substitute `$param` inside
    // NUMERIC range brackets (`@f:[$p +inf]` → "Invalid number"); only literal
    // numbers are accepted there (unlike RediSearch). Bounds are numbers /
    // epochs from trusted dataset conditions, so inlining is safe.
    *counter += 1;

    let mut clauses = Vec::new();
    if let Some(v) = criteria.get("lt").and_then(number_literal) {
        clauses.push(format!("@{}:[-inf ({}]", field_name, v));
    }
    if let Some(v) = criteria.get("gt").and_then(number_literal) {
        clauses.push(format!("@{}:[({} +inf]", field_name, v));
    }
    if let Some(v) = criteria.get("lte").and_then(number_literal) {
        clauses.push(format!("@{}:[-inf {}]", field_name, v));
    }
    if let Some(v) = criteria.get("gte").and_then(number_literal) {
        clauses.push(format!("@{}:[{} +inf]", field_name, v));
    }

    if clauses.is_empty() {
        return None;
    }
    Some((clauses.join(" "), HashMap::new()))
}

/// Render a JSON number / ISO-8601 datetime / numeric string as a literal NUMERIC
/// bound (integers verbatim, ISO-8601 → epoch seconds, numeric strings as-is).
fn number_literal(value: &serde_json::Value) -> Option<String> {
    if let Some(i) = value.as_i64() {
        Some(i.to_string())
    } else if let Some(f) = value.as_f64() {
        Some(format!("{}", f))
    } else if let Some(s) = value.as_str() {
        if let Some(epoch) = datetime_to_epoch_secs(s) {
            Some((epoch as i64).to_string())
        } else if s.parse::<f64>().is_ok() {
            Some(s.to_string())
        } else {
            None
        }
    } else {
        None
    }
}

fn build_geo_filter(
    field_name: &str,
    criteria: &serde_json::Value,
    counter: &mut usize,
) -> Option<ParsedFilter> {
    let param_prefix = format!("{}_{}", field_name, counter);
    *counter += 1;

    let mut params = HashMap::new();

    let lon_name = format!("{}_lon", param_prefix);
    let lat_name = format!("{}_lat", param_prefix);
    let radius_name = format!("{}_radius", param_prefix);

    insert_number_param(&mut params, &lon_name, criteria.get("lon")?);
    insert_number_param(&mut params, &lat_name, criteria.get("lat")?);
    insert_number_param(&mut params, &radius_name, criteria.get("radius")?);

    Some((
        format!(
            "@{}:[${} ${} ${} m]",
            field_name, lon_name, lat_name, radius_name
        ),
        params,
    ))
}

/// Insert a JSON number (or ISO-8601 datetime / numeric string) as a NUMERIC
/// bound: integers/floats verbatim, ISO-8601 strings as epoch **seconds**, and
/// other numeric strings parsed as f64 (so both ISO and raw-epoch datetime
/// bounds work — upstream is ISO-only).
fn insert_number_param(
    params: &mut HashMap<String, FilterParamValue>,
    name: &str,
    value: &serde_json::Value,
) {
    if let Some(i) = value.as_i64() {
        params.insert(name.to_string(), FilterParamValue::Int(i));
    } else if let Some(f) = value.as_f64() {
        params.insert(name.to_string(), FilterParamValue::Float(f));
    } else if let Some(s) = value.as_str() {
        if let Some(epoch) = datetime_to_epoch_secs(s) {
            // Epoch is whole seconds — emit as an integer param. Valkey Search's
            // NUMERIC-range param substitution rejects the float rendering
            // ("Invalid number"), whereas integer params are accepted (as the
            // match_any NUMERIC path already relies on).
            params.insert(name.to_string(), FilterParamValue::Int(epoch as i64));
        } else if let Ok(i) = s.parse::<i64>() {
            params.insert(name.to_string(), FilterParamValue::Int(i));
        } else if let Ok(f) = s.parse::<f64>() {
            params.insert(name.to_string(), FilterParamValue::Float(f));
        }
    }
}

fn build_condition(
    and_subfilters: Option<Vec<ParsedFilter>>,
    or_subfilters: Option<Vec<ParsedFilter>>,
) -> Option<ParsedFilter> {
    let mut clause_parts = Vec::new();
    let mut all_params = HashMap::new();

    if let Some(and_filters) = and_subfilters {
        if !and_filters.is_empty() {
            let and_clauses: Vec<String> = and_filters.iter().map(|(c, _)| c.clone()).collect();
            for (_, p) in &and_filters {
                all_params.extend(p.clone());
            }
            clause_parts.push(format!("({})", and_clauses.join(" ")));
        }
    }

    if let Some(or_filters) = or_subfilters {
        if !or_filters.is_empty() {
            let or_clauses: Vec<String> = or_filters.iter().map(|(c, _)| c.clone()).collect();
            for (_, p) in &or_filters {
                all_params.extend(p.clone());
            }
            clause_parts.push(format!("({})", or_clauses.join(" | ")));
        }
    }

    if clause_parts.is_empty() {
        return None;
    }

    Some((clause_parts.join(" "), all_params))
}

// ── FT.SEARCH ────────────────────────────────────────────────────────────

/// Execute filter-only FT.SEARCH (no KNN vector query).
fn ft_search_filter_only(
    conn: &mut Connection,
    top: usize,
    query_timeout: i64,
    filter: &ParsedFilter,
) -> Result<usize, String> {
    let (filter_expr, params) = filter;

    let mut cmd = redis::cmd("FT.SEARCH");
    cmd.arg("idx")
        .arg(filter_expr.as_str())
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("DIALECT")
        .arg(2) // Valkey uses DIALECT 2
        .arg("TIMEOUT")
        .arg(query_timeout);

    if !params.is_empty() {
        cmd.arg("PARAMS").arg(params.len() * 2);
        for (name, value) in params {
            cmd.arg(name.as_str());
            match value {
                FilterParamValue::Int(i) => {
                    cmd.arg(*i);
                }
                FilterParamValue::Float(f) => {
                    cmd.arg(*f);
                }
            }
        }
    }

    let response: Vec<redis::Value> = cmd
        .query(conn)
        .map_err(|e| format!("FT.SEARCH filter-only error: {}", e))?;

    if let Some(first) = response.first() {
        match first {
            redis::Value::Int(n) => Ok(*n as usize),
            redis::Value::BulkString(s) => Ok(String::from_utf8_lossy(s).parse().unwrap_or(0)),
            _ => Ok(0),
        }
    } else {
        Ok(0)
    }
}

#[allow(clippy::too_many_arguments)]
fn ft_search_knn(
    conn: &mut Connection,
    query_vector: &[f32],
    top: usize,
    ef: i64,
    _algorithm: &str,
    _hybrid_policy: &str,
    query_timeout: i64,
    filter: Option<&ParsedFilter>,
) -> Result<Vec<(i64, f64)>, String> {
    let vec_bytes: Vec<u8> = query_vector.iter().flat_map(|f| f.to_le_bytes()).collect();

    // Valkey Search KNN query syntax. EF_RUNTIME is a supported per-query HNSW
    // attribute (validated by valkey-search ft_search_parser.cc) — without it,
    // every ef in the search sweep runs at the index default, collapsing the
    // precision/recall curve to a single point. Passed as a $EF param below.
    let prefilter = filter
        .as_ref()
        .map(|(expr, _)| expr.as_str())
        .unwrap_or("*");

    let query_str = format!(
        "{}=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]",
        prefilter
    );

    // Valkey Search: DIALECT 2 only, no SORTBY on computed fields
    let mut cmd = redis::cmd("FT.SEARCH");
    cmd.arg("idx")
        .arg(&query_str)
        .arg("LIMIT")
        .arg(0)
        .arg(top)
        .arg("RETURN")
        .arg(1)
        .arg("vector_score")
        .arg("DIALECT")
        .arg(2)
        .arg("TIMEOUT")
        .arg(query_timeout);

    // Params: vec_param + K + EF + filter params
    let filter_param_count = filter.as_ref().map(|(_, p)| p.len() * 2).unwrap_or(0);
    let total_param_count = 6 + filter_param_count; // vec_param(2) + K(2) + EF(2) + filter params

    cmd.arg("PARAMS").arg(total_param_count);
    cmd.arg("vec_param").arg(&vec_bytes[..]);
    cmd.arg("K").arg(top.to_string());
    cmd.arg("EF").arg(ef.to_string());

    if let Some((_, params)) = filter {
        for (name, value) in params {
            cmd.arg(name.as_str());
            match value {
                FilterParamValue::Int(i) => {
                    cmd.arg(*i);
                }
                FilterParamValue::Float(f) => {
                    cmd.arg(*f);
                }
            }
        }
    }

    // Query the raw Value (not Vec<Value>) so both a RESP2 array and a RESP3 map
    // deserialize; parse_ft_search_response dispatches on the shape.
    let response: redis::Value = cmd
        .query(conn)
        .map_err(|e| format!("FT.SEARCH error: {}", e))?;

    parse_ft_search_response(&response)
}

/// Parse an FT.SEARCH reply under EITHER protocol:
/// - RESP2: a flat array `[count, id, fields, id, fields, ...]`
/// - RESP3: a map `{ results: [ { id, extra_attributes: { vector_score, .. }, .. } ], .. }`
///
/// The engine connects with RESP2 by default, but a caller can negotiate RESP3
/// (e.g. `REDIS_URI=redis://host/?protocol=resp3`), which returns a different
/// shape. Handling both keeps recall correct regardless of protocol.
fn parse_ft_search_response(response: &redis::Value) -> Result<Vec<(i64, f64)>, String> {
    match response {
        redis::Value::Array(items) => Ok(parse_ft_search_resp2(items)),
        redis::Value::Map(pairs) => Ok(parse_ft_search_resp3(pairs)),
        _ => Ok(Vec::new()),
    }
}

/// RESP2 flat array: `[count, id, fields, id, fields, ...]`.
fn parse_ft_search_resp2(response: &[redis::Value]) -> Vec<(i64, f64)> {
    let mut results = Vec::new();
    let mut i = 1;
    while i < response.len() {
        let id = value_as_i64(&response[i]);
        i += 1;

        if i < response.len() {
            let score = match &response[i] {
                redis::Value::Array(fields) => extract_vector_score(fields),
                _ => 0.0,
            };
            results.push((id, score));
            i += 1;
        }
    }
    results
}

/// RESP3 map: top-level map with a `results` array; each result is a map with an
/// `id` and an `extra_attributes` map carrying `vector_score`.
fn parse_ft_search_resp3(pairs: &[(redis::Value, redis::Value)]) -> Vec<(i64, f64)> {
    let docs = match pairs
        .iter()
        .find(|(k, _)| value_as_string(k).as_deref() == Some("results"))
        .map(|(_, v)| v)
    {
        Some(redis::Value::Array(docs)) => docs.as_slice(),
        _ => return Vec::new(),
    };

    let mut out = Vec::with_capacity(docs.len());
    for doc in docs {
        let redis::Value::Map(fields) = doc else {
            continue;
        };
        let mut id = 0i64;
        let mut score = 0.0f64;
        for (k, v) in fields {
            match value_as_string(k).as_deref() {
                Some("id") => id = value_as_string(v).and_then(|s| s.parse().ok()).unwrap_or(0),
                Some("extra_attributes") => {
                    if let redis::Value::Map(attrs) = v {
                        for (ak, av) in attrs {
                            if value_as_string(ak).as_deref() == Some("vector_score") {
                                score = value_as_string(av)
                                    .and_then(|s| s.parse().ok())
                                    .unwrap_or(0.0);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        out.push((id, score));
    }
    out
}

/// Best-effort string view of a RESP value (BulkString/SimpleString).
fn value_as_string(v: &redis::Value) -> Option<String> {
    match v {
        redis::Value::BulkString(b) => Some(String::from_utf8_lossy(b).into_owned()),
        redis::Value::SimpleString(s) => Some(s.clone()),
        _ => None,
    }
}

/// Parse a RESP value as an i64 doc id (bulk/simple string or integer).
fn value_as_i64(v: &redis::Value) -> i64 {
    match v {
        redis::Value::BulkString(data) => String::from_utf8_lossy(data).parse::<i64>().unwrap_or(0),
        redis::Value::Int(n) => *n,
        redis::Value::SimpleString(s) => s.parse().unwrap_or(0),
        _ => 0,
    }
}

/// Optional connection-URL suffix so Valkey can be benchmarked over RESP3
/// (`VALKEY_PROTOCOL=resp3`). Defaults to RESP2 (empty suffix). The FT.SEARCH
/// response parser handles both shapes, so recall is identical either way.
fn valkey_url_suffix() -> &'static str {
    if std::env::var("VALKEY_PROTOCOL")
        .map(|v| v.eq_ignore_ascii_case("resp3"))
        .unwrap_or(false)
    {
        "?protocol=resp3"
    } else {
        ""
    }
}

fn extract_vector_score(fields: &[redis::Value]) -> f64 {
    let mut i = 0;
    while i + 1 < fields.len() {
        if let redis::Value::BulkString(name) = &fields[i] {
            if name == b"vector_score" {
                if let redis::Value::BulkString(val) = &fields[i + 1] {
                    return String::from_utf8_lossy(val).parse::<f64>().unwrap_or(0.0);
                }
            }
        }
        i += 2;
    }
    0.0
}

/// Single-record HSET update (for mixed benchmark).
fn hset_single(
    conn: &mut Connection,
    config: &ValkeyEngineConfig,
    id: i64,
    vector: &[f32],
    metadata: Option<&MetadataItem>,
) -> Result<(), String> {
    let key = id.to_string();
    let vec_bytes: Vec<u8> = match config.data_type.as_str() {
        "FLOAT64" => vector
            .iter()
            .map(|&f| f as f64)
            .flat_map(|f| f.to_le_bytes())
            .collect(),
        "FLOAT16" => vector
            .iter()
            .map(|&f| half::f16::from_f32(f).to_bits())
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        "BFLOAT16" => vector
            .iter()
            .map(|&f| half::bf16::from_f32(f).to_bits())
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        _ => vector.iter().flat_map(|f| f.to_le_bytes()).collect(),
    };

    let mut cmd = redis::cmd("HSET");
    cmd.arg(key.as_str()).arg("vector").arg(&vec_bytes[..]);

    if let Some(meta) = metadata {
        for (k, v) in &meta.fields {
            match v {
                MetadataValue::String(s) => {
                    cmd.arg(k.as_str()).arg(encode_string_field(config, k, s));
                }
                MetadataValue::Labels(labels) => {
                    cmd.arg(k.as_str()).arg(labels.join(";"));
                }
                MetadataValue::Geo { lon, lat } => {
                    let lat_clamped = lat.clamp(-85.05112878, 85.05112878);
                    cmd.arg(k.as_str()).arg(format!("{},{}", lon, lat_clamped));
                }
            }
        }
    }

    cmd.query::<()>(conn)
        .map_err(|e| format!("HSET update error: {}", e))
}

/// Format a string metadata field for storage on Valkey.
///
/// - `datetime` fields: ISO-8601 → epoch seconds (already-numeric epochs pass
///   through), for the NUMERIC index.
/// - `text` fields: whitespace-tokenised into a `;`-joined TAG multi-value so a
///   single-term full-text query can match via TAG (Valkey has no TEXT type).
/// - everything else: verbatim.
fn encode_string_field(config: &ValkeyEngineConfig, key: &str, value: &str) -> String {
    if config.datetime_fields.contains(key) {
        if let Some(epoch) = datetime_to_epoch_secs(value) {
            return (epoch as i64).to_string();
        }
        return value.to_string();
    }
    if config.text_tag_fields.contains(key) {
        return tokenize_text_to_tag(value);
    }
    value.to_string()
}

/// Split text on whitespace and re-join with `;` (the TAG separator), escaping
/// any literal `;` inside a token, so it can be indexed as a multi-value TAG.
fn tokenize_text_to_tag(text: &str) -> String {
    text.split_whitespace()
        .map(|w| w.replace(';', "\\;"))
        .collect::<Vec<_>>()
        .join(";")
}

/// Collect schema fields typed `datetime` and `text` (in that order).
fn schema_transform_fields(dataset: &Dataset) -> (HashSet<String>, HashSet<String>) {
    let mut datetime_fields = HashSet::new();
    let mut text_fields = HashSet::new();
    if let Some(schema) = dataset.config.schema.as_ref().and_then(|s| s.as_object()) {
        for (field, ty) in schema {
            match ty.as_str() {
                Some("datetime") => {
                    datetime_fields.insert(field.clone());
                }
                Some("text") => {
                    text_fields.insert(field.clone());
                }
                _ => {}
            }
        }
    }
    (datetime_fields, text_fields)
}

/// Parse an ISO-8601 / RFC 3339 timestamp to epoch **seconds**. Returns `None`
/// for non-datetime strings (e.g. a numeric-epoch string), letting callers fall
/// back to numeric handling.
fn datetime_to_epoch_secs(s: &str) -> Option<f64> {
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.timestamp() as f64)
}

// ── Engine trait implementation ──────────────────────────────────────────

impl Engine for ValkeyEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        let mut conn = self.get_connection()?;

        // Record datetime fields (ISO → epoch seconds) and text fields (degraded
        // to whitespace-tokenised TAG) so upload can transform their values.
        let (dt_fields, text_fields) = schema_transform_fields(dataset);
        self.config.datetime_fields = Arc::new(dt_fields);
        self.config.text_tag_fields = Arc::new(text_fields);

        if self.config.skip_vector_index {
            println!("Skipping vector index (filter-only mode)");
        } else {
            println!(
                "Using algorithm {} with config {{'M': {}, 'EF_CONSTRUCTION': {}}}",
                self.config.algorithm, self.config.m, self.config.ef_construction
            );
        }

        self.create_index(&mut conn, dataset)?;
        self.commandstats_baseline = redis_utils::reset_commandstats(&mut conn)?;
        Ok(())
    }

    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        let normalize = dataset.needs_normalization();

        let dataset_path = dataset.get_path()?;
        println!("Reading dataset from {}...", dataset_path.display());
        let read_start = Instant::now();
        let (ids, vectors, metadata) = dataset.read_vectors(normalize)?;
        let read_time = read_start.elapsed().as_secs_f64();

        println!(
            "Read {} vectors ({}d) in {:.3}s ({:.0} vectors/sec)",
            vectors.len(),
            vectors.first().map(|v| v.len()).unwrap_or(0),
            read_time,
            vectors.len() as f64 / read_time
        );

        println!(
            "Starting upload with {} threads, batch size {}...",
            self.config.parallel, self.config.batch_size
        );
        let upload_start = Instant::now();

        if self.config.parallel <= 1 {
            self.upload_sequential(&ids, &vectors, &metadata)?;
        } else {
            self.upload_parallel(&ids, &vectors, &metadata)?;
        }

        let upload_time = upload_start.elapsed().as_secs_f64();

        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        // Include the index-build wait in total_time for cross-engine
        // comparability (mirrors mongodb; matches v0's post_upload() timing).
        let expected = vectors.len();
        let index_start = Instant::now();
        self.wait_for_indexing(expected)?;
        let index_time = index_start.elapsed().as_secs_f64();

        let total_time = read_time + upload_time + index_time;
        println!(
            "Index time: {:.3}s, Total time (read+upload+index): {:.3}s",
            index_time, total_time
        );

        // Verify no HSET failures occurred during upload
        let mut conn = self.get_connection()?;
        redis_utils::check_commandstats(
            &mut conn,
            &["hset"],
            "upload",
            self.commandstats_baseline.as_ref(),
        )?;

        Ok(UploadStats {
            upload_time,
            total_time,
            upload_count: vectors.len(),
            parallel: self.config.parallel,
            batch_size: self.config.batch_size,
            memory_usage: None,
        })
    }

    fn search(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        if self.config.skip_vector_index {
            return self.search_filter_only(dataset, params, num_queries);
        }

        let ef = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.ef)
            .unwrap_or(64);
        let parallel = params.parallel.unwrap_or(1) as usize;
        let hybrid_policy = std::env::var("VALKEY_HYBRID_POLICY").unwrap_or_default();
        // Valkey Search caps TIMEOUT at 60000ms
        let query_timeout: i64 = std::env::var("VALKEY_QUERY_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60_000);

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<ParsedFilter>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_conditions))
            .collect();

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        // Per-thread sample buffers merged on join — no per-query Mutex<Vec>
        // contention in the timed loop (see redis.rs::search). Metrics are
        // order-independent so results are unchanged; work counter uses Relaxed.
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);
        let start_time = Instant::now();

        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut precs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut recs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut mrr_vals: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut ndcg_vals: Vec<f64> = Vec::with_capacity(num_to_run);

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel);
            for _ in 0..parallel {
                let host = self.host.clone();
                let port = self.port;
                let algorithm = self.config.algorithm.clone();
                let hybrid_policy = hybrid_policy.clone();
                let queries = &queries;
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                handles.push(s.spawn(move || {
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();

                    let auth = std::env::var("VALKEY_AUTH").ok();
                    let user = std::env::var("VALKEY_USER").ok();
                    let auth_part = match (&user, &auth) {
                        (Some(u), Some(p)) => format!("{}:{}@", u, p),
                        (None, Some(p)) => format!(":{}@", p),
                        _ => String::new(),
                    };
                    let url = format!(
                        "redis://{}{}:{}/{}",
                        auth_part,
                        host,
                        port,
                        valkey_url_suffix()
                    );
                    let client = match redis::Client::open(url.as_str()) {
                        Ok(c) => c,
                        Err(_) => return (t, p, r, mr, nd),
                    };
                    let mut conn = match client.get_connection() {
                        Ok(c) => c,
                        Err(_) => return (t, p, r, mr, nd),
                    };

                    loop {
                        let idx = query_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_to_run {
                            break;
                        }

                        let top = explicit_top.unwrap_or_else(|| {
                            let n = neighbors[idx].len();
                            if n > 0 {
                                n
                            } else {
                                10
                            }
                        });

                        let query_start = Instant::now();
                        let results = ft_search_knn(
                            &mut conn,
                            &queries[idx],
                            top,
                            ef,
                            &algorithm,
                            &hybrid_policy,
                            query_timeout,
                            parsed_filters[idx].as_ref(),
                        );
                        let query_time = query_start.elapsed().as_secs_f64();

                        match &results {
                            Ok(result_ids) => {
                                let ordered_ids: Vec<i64> =
                                    result_ids.iter().map(|(id, _)| *id).collect();
                                let m = crate::metrics::compute_metrics(
                                    &ordered_ids,
                                    &neighbors[idx],
                                    top,
                                );
                                t.push(query_time);
                                p.push(m.precision);
                                r.push(m.recall);
                                mr.push(m.mrr);
                                nd.push(m.ndcg);
                            }
                            Err(e) => {
                                eprintln!("Search query {} failed: {}", idx, e);
                            }
                        }
                        pb.inc(1);
                    }
                    (t, p, r, mr, nd)
                }));
            }

            for h in handles {
                let (t, p, r, mr, nd) = h.join().unwrap();
                times.extend(t);
                precs.extend(p);
                recs.extend(r);
                mrr_vals.extend(mr);
                ndcg_vals.extend(nd);
            }
        });

        pb.finish_and_clear();
        let total_time = start_time.elapsed().as_secs_f64();

        if times.is_empty() {
            return Err("No searches completed".to_string());
        }

        // Verify no FT.SEARCH failures occurred
        let mut check_conn = self.get_connection()?;
        redis_utils::check_commandstats(
            &mut check_conn,
            &["FT.SEARCH"],
            "search",
            self.commandstats_baseline.as_ref(),
        )?;

        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));
        crate::engine::compute_search_stats(
            &times, &precs, &recs, &mrr_vals, &ndcg_vals, total_time, top, parallel, num_to_run,
        )
    }

    fn search_mixed(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
        ratio: &UpdateSearchRatio,
    ) -> Result<SearchResults, String> {
        let ef = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.ef)
            .unwrap_or(64);
        let parallel = params.parallel.unwrap_or(1) as usize;
        let hybrid_policy = std::env::var("VALKEY_HYBRID_POLICY").unwrap_or_default();
        let query_timeout: i64 = std::env::var("VALKEY_QUERY_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60_000);

        // Read queries and ground truth
        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<ParsedFilter>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_conditions))
            .collect();

        // Read vectors for updates
        let normalize = dataset.needs_normalization();
        println!("\tReading vectors for updates...");
        let (upd_ids, upd_vectors, upd_metadata) = dataset.read_vectors(normalize)?;

        // Create deterministic shuffled update sequence
        let mut update_seq: Vec<usize> = (0..upd_ids.len()).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        update_seq.shuffle(&mut rng);

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        let search_times: Arc<Mutex<Vec<f64>>> =
            Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let update_times: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
        let precisions: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let recalls: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let mrrs: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let ndcgs: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let search_idx = Arc::new(AtomicUsize::new(0));
        let update_idx = Arc::new(AtomicUsize::new(0));

        let ratio_searches = ratio.searches as usize;
        let ratio_updates = ratio.updates as usize;
        let update_seq_len = update_seq.len();

        let pb = self.create_progress_bar(num_to_run);
        let start_time = Instant::now();

        std::thread::scope(|s| {
            for _ in 0..parallel {
                let host = self.host.clone();
                let port = self.port;
                let config = self.config.clone();
                let algorithm = self.config.algorithm.clone();
                let hybrid_policy = hybrid_policy.clone();
                let queries = &queries;
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
                let upd_ids = &upd_ids;
                let upd_vectors = &upd_vectors;
                let upd_metadata = &upd_metadata;
                let update_seq = &update_seq;
                let search_times = Arc::clone(&search_times);
                let update_times = Arc::clone(&update_times);
                let precisions = Arc::clone(&precisions);
                let recalls = Arc::clone(&recalls);
                let mrrs = Arc::clone(&mrrs);
                let ndcgs = Arc::clone(&ndcgs);
                let search_idx = Arc::clone(&search_idx);
                let update_idx = Arc::clone(&update_idx);
                let pb = &pb;

                s.spawn(move || {
                    let mut conn = match ValkeyEngine::connect(&host, port) {
                        Ok(c) => c,
                        Err(_) => return,
                    };

                    'outer: loop {
                        // Search phase: do S searches
                        for _ in 0..ratio_searches {
                            let idx = search_idx.fetch_add(1, Ordering::SeqCst);
                            if idx >= num_to_run {
                                break 'outer;
                            }

                            let top = explicit_top.unwrap_or_else(|| {
                                let n = neighbors[idx].len();
                                if n > 0 {
                                    n
                                } else {
                                    10
                                }
                            });

                            let query_start = Instant::now();
                            let results = ft_search_knn(
                                &mut conn,
                                &queries[idx],
                                top,
                                ef,
                                &algorithm,
                                &hybrid_policy,
                                query_timeout,
                                parsed_filters[idx].as_ref(),
                            );
                            let query_time = query_start.elapsed().as_secs_f64();

                            match &results {
                                Ok(result_ids) => {
                                    search_times.lock().unwrap().push(query_time);
                                    let ordered_ids: Vec<i64> =
                                        result_ids.iter().map(|(id, _)| *id).collect();
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

                        // Update phase: do U updates
                        for _ in 0..ratio_updates {
                            let uidx = update_idx.fetch_add(1, Ordering::SeqCst);
                            let data_idx = update_seq[uidx % update_seq_len];

                            let update_start = Instant::now();
                            let _ = hset_single(
                                &mut conn,
                                &config,
                                upd_ids[data_idx],
                                &upd_vectors[data_idx],
                                upd_metadata[data_idx].as_ref(),
                            );
                            let update_time = update_start.elapsed().as_secs_f64();
                            update_times.lock().unwrap().push(update_time);
                        }
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
        let u_times = update_times.lock().unwrap();

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

        // Update latency stats
        let (update_count, update_rps, update_mean_time, update_p50, update_p95, update_p99) =
            if !u_times.is_empty() {
                let u_rps = u_times.len() as f64 / total_time;
                let u_mean = u_times.iter().sum::<f64>() / u_times.len() as f64;
                let mut u_sorted: Vec<f64> = u_times.clone();
                u_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let u_p50 = u_sorted
                    .get((u_sorted.len() as f64 * 0.50) as usize)
                    .copied()
                    .unwrap_or(0.0);
                let u_p95 = u_sorted
                    .get(((u_sorted.len() as f64 * 0.95) as usize).min(u_sorted.len() - 1))
                    .copied()
                    .unwrap_or(0.0);
                let u_p99 = u_sorted
                    .get(((u_sorted.len() as f64 * 0.99) as usize).min(u_sorted.len() - 1))
                    .copied()
                    .unwrap_or(0.0);
                (
                    Some(u_times.len()),
                    Some(u_rps),
                    Some(u_mean),
                    Some(u_p50),
                    Some(u_p95),
                    Some(u_p99),
                )
            } else {
                (None, None, None, None, None, None)
            };

        // Verify no failures occurred
        let mut check_conn = self.get_connection()?;
        redis_utils::check_commandstats(
            &mut check_conn,
            &["FT.SEARCH", "hset"],
            "mixed",
            self.commandstats_baseline.as_ref(),
        )?;

        Ok(SearchResults {
            total_time,
            mean_time,
            mean_precision,
            mean_recall,
            mean_mrr,
            mean_ndcg,
            recalls: recs.to_vec(),
            mrrs: mrr_vals.to_vec(),
            ndcgs: ndcg_vals.to_vec(),
            std_time,
            min_time,
            max_time,
            rps,
            p50_time,
            p95_time,
            p99_time,
            precisions: precs.to_vec(),
            latencies: times.to_vec(),
            top: explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10)),
            num_queries: times.len(),
            requested_queries: num_to_run,
            failed_queries: num_to_run.saturating_sub(times.len()),
            parallel,
            update_count,
            update_rps,
            update_mean_time,
            update_p50_time: update_p50,
            update_p95_time: update_p95,
            update_p99_time: update_p99,
            update_latencies: Some(u_times.to_vec()),
            update_search_ratio: Some(format!("{}:{}", ratio.updates, ratio.searches)),
            ..Default::default()
        })
    }

    fn delete(&mut self) -> Result<(), String> {
        let mut conn = self.get_connection()?;
        // Valkey Search does not support DD flag on FT.DROPINDEX
        let _ = redis::cmd("FT.DROPINDEX").arg("idx").query::<()>(&mut conn);
        // Flush all keys to clean up uploaded data
        let _ = redis::cmd("FLUSHALL").query::<()>(&mut conn);
        Ok(())
    }

    fn get_memory_usage(&mut self) -> Option<serde_json::Value> {
        let mut conn = self.get_connection().ok()?;

        let info_str: String = redis::cmd("INFO").arg("memory").query(&mut conn).ok()?;
        let used_memory: i64 = info_str
            .lines()
            .find(|l| l.starts_with("used_memory:"))
            .and_then(|l| l.strip_prefix("used_memory:"))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0);

        let ft_info: Option<serde_json::Value> = redis::cmd("FT.INFO")
            .arg("idx")
            .query::<redis::Value>(&mut conn)
            .ok()
            .map(|v| redis_value_to_json(&v));

        Some(serde_json::json!({
            "used_memory": [used_memory],
            "index_info": ft_info,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::{parse_conditions, FilterParamValue};

    #[test]
    fn match_any_string_list_emits_inlined_tag_or() {
        let cond = serde_json::json!({"and":[{"color":{"match":{"any":["red","blue"]}}}]});
        let (q, _params) = parse_conditions(&cond).unwrap();
        // Values inlined (no $param inside TAG braces on Valkey Search).
        assert!(q.contains("@color:{red | blue}"), "q={}", q);
    }

    #[test]
    fn match_any_int_list_emits_numeric_or() {
        let cond = serde_json::json!({"and":[{"size":{"match":{"any":[1,2]}}}]});
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@size:[$size_0 $size_0]"), "q={}", q);
        assert!(q.contains("@size:[$size_1 $size_1]"), "q={}", q);
        assert!(matches!(
            params.get("size_0"),
            Some(FilterParamValue::Int(1))
        ));
    }

    #[test]
    fn match_any_empty_list_matches_nothing() {
        let cond = serde_json::json!({"and":[{"color":{"match":{"any":[]}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("-@color:{"), "expected never-match, q={}", q);
    }

    #[test]
    fn match_any_escapes_or_delimiter() {
        // A value containing '|' must not break the OR structure.
        let cond = serde_json::json!({"and":[{"color":{"match":{"any":["a|b"]}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("a\\|b"), "q={}", q);
    }

    #[test]
    fn match_exact_value_still_inlined_tag() {
        let cond = serde_json::json!({"and":[{"color":{"match":{"value":"red"}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@color:{red}"), "q={}", q);
    }

    // ── New filter datatypes: bool / uuid / full-text / datetime ───────────
    use super::{
        datetime_to_epoch_secs, encode_string_field, tokenize_text_to_tag, ValkeyEngineConfig,
    };
    use std::collections::HashSet;
    use std::sync::Arc;

    #[test]
    fn match_bool_emits_inlined_tag_token() {
        let cond = serde_json::json!({"and":[{"flag":{"match":{"value": true}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@flag:{true}"), "q={}", q);
        let cond = serde_json::json!({"and":[{"flag":{"match":{"value": false}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@flag:{false}"), "q={}", q);
    }

    #[test]
    fn match_uuid_value_inlined_tag() {
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        let cond = serde_json::json!({"and":[{"uid":{"match":{"value": uuid}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains(&format!("@uid:{{{}}}", uuid)), "q={}", q);
    }

    #[test]
    fn match_text_degrades_to_single_token_tag() {
        // Full-text on Valkey → single-term TAG match (degraded).
        let cond = serde_json::json!({"and":[{"body":{"match":{"text": "quick"}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@body:{quick}"), "q={}", q);
    }

    #[test]
    fn match_text_empty_is_never_match() {
        let cond = serde_json::json!({"and":[{"body":{"match":{"text": "  "}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("-@body:{"), "expected never-match, q={}", q);
    }

    #[test]
    fn range_datetime_iso_bounds_inline_epoch() {
        let cond = serde_json::json!({"and":[{"ts":{"range":{
            "gte": "2021-01-01T00:00:00Z",
            "lt":  "2022-01-01T00:00:00Z"
        }}}]});
        // 2021-01-01T00:00:00Z == 1609459200, 2022-01-01T00:00:00Z == 1640995200.
        // Valkey doesn't substitute $param in NUMERIC brackets → bounds are
        // inlined as integer epochs; no params emitted for the range.
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@ts:[1609459200 +inf]"), "q={}", q);
        assert!(q.contains("@ts:[-inf (1640995200]"), "q={}", q);
        assert!(!params.keys().any(|k| k.starts_with("ts_")), "no range params");
    }

    #[test]
    fn range_datetime_numeric_epoch_bounds_inline() {
        let cond = serde_json::json!({"and":[{"ts":{"range":{"gte": 1609459200}}}]});
        let (q, _params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@ts:[1609459200 +inf]"), "q={}", q);
    }

    #[test]
    fn datetime_to_epoch_secs_parses_rfc3339_and_rejects_plain() {
        assert_eq!(
            datetime_to_epoch_secs("2021-01-01T00:00:00Z").map(|f| f as i64),
            Some(1609459200)
        );
        assert!(datetime_to_epoch_secs("not-a-date").is_none());
        assert!(datetime_to_epoch_secs("1609459200").is_none());
    }

    #[test]
    fn tokenize_text_to_tag_splits_on_whitespace() {
        assert_eq!(tokenize_text_to_tag("the sky is blue"), "the;sky;is;blue");
        assert_eq!(tokenize_text_to_tag("solo"), "solo");
    }

    #[test]
    fn encode_string_field_transforms_datetime_and_text() {
        let mut dt = HashSet::new();
        dt.insert("ts".to_string());
        let mut txt = HashSet::new();
        txt.insert("body".to_string());
        let cfg = ValkeyEngineConfig {
            m: 16,
            ef_construction: 128,
            data_type: "FLOAT32".to_string(),
            algorithm: "hnsw".to_string(),
            batch_size: 1,
            parallel: 1,
            skip_vector_index: false,
            datetime_fields: Arc::new(dt),
            text_tag_fields: Arc::new(txt),
        };
        assert_eq!(
            encode_string_field(&cfg, "ts", "2021-01-01T00:00:00Z"),
            "1609459200"
        );
        assert_eq!(encode_string_field(&cfg, "ts", "1609459200"), "1609459200");
        assert_eq!(
            encode_string_field(&cfg, "body", "the sky is blue"),
            "the;sky;is;blue"
        );
        assert_eq!(encode_string_field(&cfg, "color", "red"), "red");
    }

    // ── redis::Value response parsing ──────────────────────────────────────
    // These guard the manual RESP-array parsing (the surface most exposed to a
    // redis-crate upgrade / RESP2-vs-RESP3 change).
    use super::{extract_vector_score, parse_ft_search_response, redis_value_to_json};
    use redis::Value;

    fn bulk(s: &str) -> Value {
        Value::BulkString(s.as_bytes().to_vec())
    }

    #[test]
    fn parse_ft_search_response_resp2_reads_id_score_pairs() {
        // RESP2 FT.SEARCH shape: [count, id1, fields1, id2, fields2, ...]
        let resp = Value::Array(vec![
            Value::Int(2),
            bulk("7"),
            Value::Array(vec![bulk("vector_score"), bulk("0.25")]),
            Value::Int(42), // id may also arrive as an integer
            Value::Array(vec![bulk("vector_score"), bulk("1.5")]),
        ]);
        let hits = parse_ft_search_response(&resp).unwrap();
        assert_eq!(hits, vec![(7, 0.25), (42, 1.5)]);
    }

    #[test]
    fn parse_ft_search_response_resp3_map_reads_results() {
        // RESP3 FT.SEARCH shape: map with a `results` array of per-doc maps.
        let doc = |id: &str, score: &str| {
            Value::Map(vec![
                (bulk("id"), bulk(id)),
                (
                    bulk("extra_attributes"),
                    Value::Map(vec![(bulk("vector_score"), bulk(score))]),
                ),
            ])
        };
        let resp = Value::Map(vec![
            (
                bulk("results"),
                Value::Array(vec![doc("7", "0.25"), doc("42", "1.5")]),
            ),
            (bulk("total_results"), Value::Int(2)),
        ]);
        let hits = parse_ft_search_response(&resp).unwrap();
        assert_eq!(hits, vec![(7, 0.25), (42, 1.5)]);
    }

    #[test]
    fn parse_ft_search_response_empty_and_unknown_variants() {
        assert_eq!(
            parse_ft_search_response(&Value::Array(vec![])).unwrap(),
            vec![]
        );
        assert_eq!(parse_ft_search_response(&Value::Nil).unwrap(), vec![]);
        // A non-string/int id falls back to 0; a non-array field block → score 0.
        let resp = Value::Array(vec![Value::Int(1), Value::Nil, Value::Nil]);
        assert_eq!(parse_ft_search_response(&resp).unwrap(), vec![(0, 0.0)]);
    }

    #[test]
    fn extract_vector_score_finds_field_or_defaults_zero() {
        let fields = vec![
            bulk("__key"),
            bulk("doc:1"),
            bulk("vector_score"),
            bulk("0.75"),
        ];
        assert!((extract_vector_score(&fields) - 0.75).abs() < 1e-9);
        // Missing field → 0.0
        assert_eq!(extract_vector_score(&[bulk("other"), bulk("x")]), 0.0);
    }

    #[test]
    fn redis_value_to_json_covers_scalars_arrays_maps_and_fallthrough() {
        assert_eq!(redis_value_to_json(&Value::Nil), serde_json::Value::Null);
        assert_eq!(redis_value_to_json(&Value::Int(5)), serde_json::json!(5));
        assert_eq!(
            redis_value_to_json(&Value::Boolean(true)),
            serde_json::json!(true)
        );
        assert_eq!(redis_value_to_json(&bulk("hi")), serde_json::json!("hi"));
        assert_eq!(
            redis_value_to_json(&Value::Array(vec![Value::Int(1), bulk("a")])),
            serde_json::json!([1, "a"])
        );
        assert_eq!(
            redis_value_to_json(&Value::Map(vec![(bulk("k"), Value::Int(9))])),
            serde_json::json!({"k": 9})
        );
        // Non-exhaustive/other variants (e.g. Okay) must not panic and must not be
        // silently dropped — they debug-format to a non-empty string rather than
        // vanish. (The exact string is a redis-crate impl detail, so don't pin it.)
        let okay = redis_value_to_json(&Value::Okay);
        assert!(
            okay.as_str().map(|s| !s.is_empty()).unwrap_or(false),
            "Okay should map to a non-empty JSON string, got {:?}",
            okay
        );
    }
}
