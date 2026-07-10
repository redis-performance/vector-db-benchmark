//! Redis-rs engine implementation.
//!
//! Implements the Engine trait for RediSearch vector similarity.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::SeedableRng;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};

use super::redis_utils;
use redis::Connection;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UpdateSearchRatio, UploadStats};
use crate::metrics::compute_metrics;
use vector_db_benchmark::readers::metadata::{MetadataItem, MetadataValue};

/// Redis engine configuration
#[derive(Clone)]
pub struct RedisEngineConfig {
    pub m: i64,
    pub ef_construction: i64,
    pub data_type: String,
    pub algorithm: String,
    pub batch_size: usize,
    pub parallel: usize,
    pub skip_vector_index: bool,
    /// Schema field names declared as `datetime`. Values for these fields are
    /// ISO-8601 strings in the payload; they are converted to epoch seconds at
    /// HSET time so the NUMERIC index/range filters match. Populated in
    /// `configure()` from the dataset schema. Wrapped in `Arc` so the per-thread
    /// config clones share one set.
    pub datetime_fields: Arc<HashSet<String>>,
}

pub struct RedisEngine {
    name: String,
    redis_url: String,
    config: RedisEngineConfig,
    search_params: Vec<SearchParams>,
    commandstats_baseline: Option<redis_utils::CommandStatsBaseline>,
}

impl RedisEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let redis_url = crate::engine::build_redis_url(host);

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
            redis_url,
            config: RedisEngineConfig {
                m,
                ef_construction,
                data_type,
                algorithm,
                batch_size,
                parallel,
                skip_vector_index: engine_config.skip_vector_index,
                datetime_fields: Arc::new(HashSet::new()),
            },
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            commandstats_baseline: None,
        })
    }

    fn get_connection(&self) -> Result<Connection, String> {
        let client = redis::Client::open(self.redis_url.as_str()).map_err(|e| e.to_string())?;
        client.get_connection().map_err(|e| e.to_string())
    }

    fn create_index(&self, conn: &mut Connection, dataset: &Dataset) -> Result<(), String> {
        let distance = dataset.distance();
        let vector_size = dataset.vector_size();

        // Drop existing index if any
        let _ = redis::cmd("FT.DROPINDEX")
            .arg("idx")
            .arg("DD")
            .query::<()>(conn);

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

        // Vector field with HNSW params (matches Python v0 configure.py)
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

        // Add schema fields from dataset config for filtering
        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                for (field_name, field_type) in schema_obj {
                    let ft = field_type.as_str().unwrap_or("");
                    match ft {
                        // keyword/uuid/bool are all exact-match TAG fields.
                        // uuid & bool values are single tokens (a UUID string, or
                        // "true"/"false"); SEPARATOR ; + SORTABLE UNF matches the
                        // keyword treatment so exact TAG matching works.
                        "keyword" | "uuid" | "bool" => {
                            cmd.arg(field_name)
                                .arg("TAG")
                                .arg("SEPARATOR")
                                .arg(";")
                                .arg("SORTABLE")
                                .arg("UNF");
                        }
                        "int" | "float" => {
                            cmd.arg(field_name).arg("NUMERIC").arg("SORTABLE");
                        }
                        // datetime is stored as epoch seconds (see upload) and
                        // indexed NUMERIC so range filters work with either ISO or
                        // numeric bounds.
                        "datetime" => {
                            cmd.arg(field_name).arg("NUMERIC").arg("SORTABLE");
                        }
                        "text" => {
                            cmd.arg(field_name).arg("TEXT").arg("SORTABLE");
                        }
                        "geo" => {
                            cmd.arg(field_name).arg("GEO").arg("SORTABLE");
                        }
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
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        std::thread::scope(|s| {
            for _ in 0..self.config.parallel {
                let redis_url = self.redis_url.clone();
                let config = self.config.clone();
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let client = match redis::Client::open(redis_url.as_str()) {
                        Ok(c) => c,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e.to_string());
                            return;
                        }
                    };
                    let mut conn = match client.get_connection() {
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

    /// Wait until FT.INFO reports num_docs >= expected.
    fn wait_for_indexing(&self, expected: usize) -> Result<(), String> {
        let mut conn = self.get_connection()?;
        let max_wait = 120; // seconds
        let start = Instant::now();

        loop {
            let info: redis::Value = redis::cmd("FT.INFO")
                .arg("idx")
                .query(&mut conn)
                .map_err(|e| format!("FT.INFO error: {}", e))?;

            // Parse num_docs, indexing, and percent_indexed from FT.INFO response.
            // The response can be a flat array (RESP2) or a Map (RESP3).
            let mut num_docs: usize = 0;
            let mut indexing: bool = false;
            let mut percent_indexed: f64 = 1.0;

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

            fn extract_f64(val: &redis::Value) -> f64 {
                match val {
                    redis::Value::BulkString(s) => {
                        String::from_utf8_lossy(s).parse().unwrap_or(1.0)
                    }
                    redis::Value::Int(n) => *n as f64,
                    redis::Value::Double(f) => *f,
                    redis::Value::SimpleString(s) => s.parse().unwrap_or(1.0),
                    _ => 1.0,
                }
            }

            match &info {
                redis::Value::Array(arr) => {
                    // RESP2: alternating key-value pairs (keys can be BulkString or SimpleString)
                    for i in (0..arr.len()).step_by(2) {
                        let key_str = match &arr[i] {
                            redis::Value::BulkString(s) => String::from_utf8_lossy(s).to_string(),
                            redis::Value::SimpleString(s) => s.clone(),
                            _ => continue,
                        };
                        if key_str == "num_docs" {
                            if let Some(val) = arr.get(i + 1) {
                                num_docs = extract_usize(val);
                            }
                        }
                        if key_str == "indexing" {
                            if let Some(val) = arr.get(i + 1) {
                                indexing = extract_bool_nonzero(val);
                            }
                        }
                        if key_str == "percent_indexed" {
                            if let Some(val) = arr.get(i + 1) {
                                percent_indexed = extract_f64(val);
                            }
                        }
                    }
                }
                redis::Value::Map(map) => {
                    // RESP3: key-value map
                    for (k, v) in map {
                        let key_str = match k {
                            redis::Value::BulkString(s) => String::from_utf8_lossy(s).to_string(),
                            redis::Value::SimpleString(s) => s.clone(),
                            _ => continue,
                        };
                        if key_str == "num_docs" {
                            num_docs = extract_usize(v);
                        }
                        if key_str == "indexing" {
                            indexing = extract_bool_nonzero(v);
                        }
                        if key_str == "percent_indexed" {
                            percent_indexed = extract_f64(v);
                        }
                    }
                }
                _ => {
                    eprintln!("Unexpected FT.INFO response type: {:?}", info);
                }
            }

            if num_docs >= expected && !indexing && percent_indexed >= 1.0 {
                println!(
                    "Indexing complete: {} docs in {:.1}s",
                    num_docs,
                    start.elapsed().as_secs_f64()
                );
                return Ok(());
            }

            if start.elapsed().as_secs() > max_wait {
                println!(
                    "Warning: indexing timeout after {}s (num_docs={}/{}, percent_indexed={:.2})",
                    max_wait, num_docs, expected, percent_indexed
                );
                return Ok(());
            }

            if start.elapsed().as_secs() > 0 && start.elapsed().as_secs().is_multiple_of(10) {
                println!(
                    "  indexing... num_docs={}/{}, percent_indexed={:.2}, indexing={}",
                    num_docs, expected, percent_indexed, indexing
                );
            }

            std::thread::sleep(std::time::Duration::from_millis(500));
        }
    }

    /// Filter-only search: run FT.SEARCH with filter conditions only (no KNN).
    /// No precision calculation (no ground truth for filter-only queries).
    fn search_filter_only(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        let parallel = params.parallel.unwrap_or(1) as usize;
        let query_timeout: i64 = std::env::var("REDIS_QUERY_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(90_000);

        // Read queries (we only need the filter conditions)
        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (_queries, neighbors, conditions) = dataset.read_queries()?;

        // Parse all conditions up front
        let parsed_filters: Vec<Option<ParsedFilter>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_conditions))
            .collect();

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);

        // Only queries that have filter conditions
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
                let redis_url = self.redis_url.clone();
                let parsed_filters = &parsed_filters;
                let runnable_indices = &runnable_indices;
                let neighbors = &neighbors;
                let search_times = Arc::clone(&search_times);
                let errors = Arc::clone(&errors);
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                s.spawn(move || {
                    let client = match redis::Client::open(redis_url.as_str()) {
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
                        // Round-robin over available filter queries
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

        // Verify no FT.SEARCH failures occurred
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
            mean_precision: -1.0, // Not applicable for filter-only
            std_time,
            min_time,
            max_time,
            rps,
            p50_time,
            p95_time,
            p99_time,
            precisions: vec![], // No precision for filter-only
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

/// Encode a vector to the little-endian byte layout RediSearch expects for the
/// given `TYPE`. Integer types (INT8/UINT8) round each component and saturate to
/// the type's range; unknown types fall back to FLOAT32.
fn encode_vector(data_type: &str, vector: &[f32]) -> Vec<u8> {
    match data_type {
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
        // INT8/UINT8: quantized vectors (e.g. Cohere int8 embeddings) arrive as
        // f32; round and clamp to the integer range (float→int casts saturate).
        "INT8" => vector
            .iter()
            .map(|&f| (f.round().clamp(-128.0, 127.0) as i8).to_le_bytes())
            .flat_map(|b| b.into_iter())
            .collect(),
        "UINT8" => vector
            .iter()
            .map(|&f| f.round().clamp(0.0, 255.0) as u8)
            .collect(),
        _ => vector.iter().flat_map(|f| f.to_le_bytes()).collect(),
    }
}

/// Internal batch upload function
fn upload_batch_internal(
    conn: &mut Connection,
    config: &RedisEngineConfig,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
) -> Result<(), String> {
    let mut pipe = redis::pipe();

    for i in 0..ids.len() {
        let key = ids[i].to_string();
        let vec_bytes = encode_vector(&config.data_type, &vectors[i]);

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

        let mut hset_cmd = redis::cmd("HSET");
        hset_cmd.arg(key.as_str());
        for (field_key, field_val) in &fields {
            hset_cmd.arg(&field_key[..]).arg(&field_val[..]);
        }
        pipe.add_command(hset_cmd);
    }

    pipe.query::<()>(conn).map_err(|e| e.to_string())?;
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
// Mirrors Python v0/engine/clients/redis/parser.py RedisConditionParser.
// Converts JSON filter conditions into RediSearch query filter syntax.

/// A filter parameter value to pass to FT.SEARCH PARAMS.
#[derive(Debug, Clone)]
enum FilterParamValue {
    Str(String),
    Int(i64),
    Float(f64),
}

/// Parsed filter: (prefilter_query_string, param_name -> param_value).
type ParsedFilter = (String, HashMap<String, FilterParamValue>);

/// Parse meta_conditions JSON into a RediSearch prefilter string + params.
/// Returns None when no conditions are present.
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

/// Build individual subfilters from an array of condition entries.
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

/// Build a single filter expression from a field condition.
fn build_filter(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
    counter: &mut usize,
) -> Option<ParsedFilter> {
    match condition_type {
        "match" => {
            // match_any (IN-list) takes precedence over exact {value}, which
            // takes precedence over full-text {text}.
            if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                Some(build_match_any_filter(field_name, any, counter))
            } else if let Some(text) = criteria.get("text").and_then(|v| v.as_str()) {
                Some(build_text_filter(field_name, text, counter))
            } else {
                build_exact_match_filter(field_name, criteria, counter)
            }
        }
        "range" => build_range_filter(field_name, criteria, counter),
        "geo" => build_geo_filter(field_name, criteria, counter),
        _ => None,
    }
}

/// Build a `match_any` (IN-list) filter, the OR-of-values semantics that mirror
/// qdrant's `Condition::matches(field, Vec)`.
///
/// - All-integer list -> NUMERIC OR of single-value ranges
///   `(@f:[$a $a] | @f:[$b $b])` (field indexed NUMERIC).
/// - Otherwise -> TAG OR `@f:{$a | $b}` over the (non-empty) string values
///   (field indexed TAG). Empty-string tokens are dropped (they are invalid
///   TAG syntax and can never match an exact keyword).
/// - Empty / no representable values -> a never-match `(@f:{$s} -@f:{$s})`
///   (a tag AND not-that-tag contradiction) so an empty IN-set matches NOTHING
///   rather than being dropped — dropping the sole clause would leave no
///   prefilter and run kNN over ALL docs (the inverse of the filter). This
///   assumes a TAG field, the realistic case for a keyword IN-list.
///
/// NOTE: RediSearch TAG matching is case-INSENSITIVE, whereas qdrant keyword
/// match is case-sensitive; for mixed-case data the two engines can select
/// different documents. All shipped keyword datasets use consistent casing.
fn build_match_any_filter(
    field_name: &str,
    any: &[serde_json::Value],
    counter: &mut usize,
) -> ParsedFilter {
    let mut params = HashMap::new();

    // All-integer list -> NUMERIC OR.
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

    // Otherwise TAG OR over the non-empty string values.
    let tokens: Vec<String> = any
        .iter()
        .filter_map(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| {
            let p = format!("{}_{}", field_name, counter);
            *counter += 1;
            params.insert(p.clone(), FilterParamValue::Str(s.to_string()));
            format!("${}", p)
        })
        .collect();

    if tokens.is_empty() {
        // Never-match: `@f:{$s} -@f:{$s}` is an unsatisfiable contradiction.
        let p = format!("{}_{}", field_name, counter);
        *counter += 1;
        params.insert(
            p.clone(),
            FilterParamValue::Str("__match_any_never_match__".to_string()),
        );
        return (
            format!("(@{0}:{{${1}}} -@{0}:{{${1}}})", field_name, p),
            params,
        );
    }

    (
        format!("@{}:{{{}}}", field_name, tokens.join(" | ")),
        params,
    )
}

/// Build exact match filter: string → @field:{$param}, numeric → @field:[$param $param]
fn build_exact_match_filter(
    field_name: &str,
    criteria: &serde_json::Value,
    counter: &mut usize,
) -> Option<ParsedFilter> {
    let value = criteria.get("value")?;
    let param_name = format!("{}_{}", field_name, counter);
    *counter += 1;

    let mut params = HashMap::new();

    // bool → TAG match on the literal "true"/"false" token. Checked before the
    // numeric arms because serde treats JSON `true`/`false` as neither i64 nor
    // f64 (and never as a string).
    if let Some(b) = value.as_bool() {
        let token = if b { "true" } else { "false" };
        params.insert(param_name.clone(), FilterParamValue::Str(token.to_string()));
        return Some((format!("@{}:{{${}}}", field_name, param_name), params));
    }

    if let Some(s) = value.as_str() {
        params.insert(param_name.clone(), FilterParamValue::Str(s.to_string()));
        Some((format!("@{}:{{${}}}", field_name, param_name), params))
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

/// Build a full-text filter over a TEXT field: `@field:($param)`.
///
/// The search terms are passed as a single Str param (DIALECT 2 substitutes it
/// into the text-search context), so RediSearch tokenizes and matches them
/// against the indexed TEXT field. An empty/blank query would be invalid text
/// syntax, so it degrades to a never-match `(@f:($p) -@f:($p))` contradiction
/// rather than an empty clause (which, as the sole prefilter, would run kNN over
/// ALL docs — the inverse of the intended filter).
fn build_text_filter(field_name: &str, text: &str, counter: &mut usize) -> ParsedFilter {
    let param_name = format!("{}_{}", field_name, counter);
    *counter += 1;

    let mut params = HashMap::new();
    let trimmed = text.trim();

    if trimmed.is_empty() {
        params.insert(
            param_name.clone(),
            FilterParamValue::Str("__text_never_match__".to_string()),
        );
        return (
            format!("(@{0}:(${1}) -@{0}:(${1}))", field_name, param_name),
            params,
        );
    }

    params.insert(
        param_name.clone(),
        FilterParamValue::Str(trimmed.to_string()),
    );
    (format!("@{}:(${})", field_name, param_name), params)
}

/// Build range filter: @field:[-inf ($param_lt], @field:[($param_gt +inf], etc.
fn build_range_filter(
    field_name: &str,
    criteria: &serde_json::Value,
    counter: &mut usize,
) -> Option<ParsedFilter> {
    let param_prefix = format!("{}_{}", field_name, counter);
    *counter += 1;

    let mut params = HashMap::new();
    let mut clauses = Vec::new();

    if let Some(lt) = criteria.get("lt") {
        let pname = format!("{}_lt", param_prefix);
        insert_number_param(&mut params, &pname, lt);
        clauses.push(format!("@{}:[-inf (${}]", field_name, pname));
    }
    if let Some(gt) = criteria.get("gt") {
        let pname = format!("{}_gt", param_prefix);
        insert_number_param(&mut params, &pname, gt);
        clauses.push(format!("@{}:[${} +inf]", field_name, pname));
    }
    if let Some(lte) = criteria.get("lte") {
        let pname = format!("{}_lte", param_prefix);
        insert_number_param(&mut params, &pname, lte);
        clauses.push(format!("@{}:[-inf ${}]", field_name, pname));
    }
    if let Some(gte) = criteria.get("gte") {
        let pname = format!("{}_gte", param_prefix);
        insert_number_param(&mut params, &pname, gte);
        clauses.push(format!("@{}:[${} +inf]", field_name, pname));
    }

    if clauses.is_empty() {
        return None;
    }

    Some((clauses.join(" "), params))
}

/// Build geo filter: @field:[$lon $lat $radius m]
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

/// Insert a JSON number (or ISO-8601 datetime / numeric string) into the params
/// map as a NUMERIC bound.
///
/// - integer / float JSON → the numeric value.
/// - ISO-8601 string → epoch **seconds** (datetime range bound). This lets a
///   `datetime` field be filtered with ISO bounds, e.g.
///   `{"range":{"gte":"2021-01-01T00:00:00Z"}}`.
/// - other numeric string → parsed as f64 (accepts a numeric-epoch bound too, so
///   both ISO and raw-epoch datetime bounds work — better than upstream's
///   ISO-only handling).
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
            params.insert(name.to_string(), FilterParamValue::Float(epoch));
        } else if let Ok(f) = s.parse::<f64>() {
            params.insert(name.to_string(), FilterParamValue::Float(f));
        }
    }
}

/// Combine AND and OR subfilters into a single prefilter expression.
/// AND clauses are space-joined, OR clauses are pipe-joined.
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
        .arg(4)
        .arg("TIMEOUT")
        .arg(query_timeout);

    // Add filter params
    if !params.is_empty() {
        cmd.arg("PARAMS").arg(params.len() * 2);
        for (name, value) in params {
            cmd.arg(name.as_str());
            match value {
                FilterParamValue::Str(s) => {
                    cmd.arg(s.as_str());
                }
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

    // First element is the total count
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

/// Execute FT.SEARCH KNN query with optional prefilter, return (id, score) pairs.
#[allow(clippy::too_many_arguments)]
fn ft_search_knn(
    conn: &mut Connection,
    query_vector: &[f32],
    top: usize,
    ef: i64,
    algorithm: &str,
    hybrid_policy: &str,
    query_timeout: i64,
    filter: Option<&ParsedFilter>,
    data_type: &str,
) -> Result<Vec<(i64, f64)>, String> {
    // Encode the query vector to match the index TYPE (e.g. INT8), otherwise a
    // FLOAT32 blob would be sent against an INT8 index.
    let vec_bytes = encode_vector(data_type, query_vector);

    // Build KNN conditions string
    let knn_conditions = if algorithm.to_uppercase() == "HNSW" && hybrid_policy != "ADHOC_BF" {
        "EF_RUNTIME $EF"
    } else {
        ""
    };

    // Build hybrid policy suffix
    let hybrid_suffix = if !hybrid_policy.is_empty() {
        format!("=>{{$HYBRID_POLICY: {} }}", hybrid_policy)
    } else {
        String::new()
    };

    // Prefilter: use filter expression or "*" (match all)
    let prefilter = filter
        .as_ref()
        .map(|(expr, _)| expr.as_str())
        .unwrap_or("*");

    let query_str = format!(
        "{}=>[KNN $K @vector $vec_param {} AS vector_score]{}",
        prefilter, knn_conditions, hybrid_suffix
    );

    // Build FT.SEARCH command
    let mut cmd = redis::cmd("FT.SEARCH");
    cmd.arg("idx")
        .arg(&query_str)
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
        .arg("TIMEOUT")
        .arg(query_timeout);

    // Count params: vec_param(2) + K(2) + optional EF(2) + filter params (2 each)
    let filter_param_count = filter.as_ref().map(|(_, p)| p.len() * 2).unwrap_or(0);
    let mut base_param_count = 4; // vec_param + K
    if algorithm.to_uppercase() == "HNSW" && hybrid_policy != "ADHOC_BF" {
        base_param_count += 2; // EF
    }
    let total_param_count = base_param_count + filter_param_count;

    cmd.arg("PARAMS").arg(total_param_count);
    cmd.arg("vec_param").arg(&vec_bytes[..]);
    cmd.arg("K").arg(top.to_string());

    if algorithm.to_uppercase() == "HNSW" && hybrid_policy != "ADHOC_BF" {
        cmd.arg("EF").arg(ef.to_string());
    }

    // Add filter params
    if let Some((_, params)) = filter {
        for (name, value) in params {
            cmd.arg(name.as_str());
            match value {
                FilterParamValue::Str(s) => {
                    cmd.arg(s.as_str());
                }
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

/// Parse FT.SEARCH response into (id, score) pairs.
/// Response format: [total_count, doc_id, [field_values...], doc_id, [field_values...], ...]
/// Parse an FT.SEARCH reply under EITHER protocol:
/// - RESP2: a flat array `[count, id, fields, id, fields, ...]`
/// - RESP3: a map `{ results: [ { id, extra_attributes: { vector_score, .. }, .. } ], .. }`
///
/// The engine connects with RESP2 by default, but a caller can negotiate RESP3
/// (e.g. `REDIS_URI=redis://host/?protocol=resp3`), which returns a completely
/// different shape. Handling both keeps recall correct regardless of protocol.
fn parse_ft_search_response(response: &redis::Value) -> Result<Vec<(i64, f64)>, String> {
    match response {
        redis::Value::Array(items) => Ok(parse_ft_search_resp2(items)),
        redis::Value::Map(pairs) => Ok(parse_ft_search_resp3(pairs)),
        // Nil (no index/empty) or any unexpected shape → no hits.
        _ => Ok(Vec::new()),
    }
}

/// RESP2 flat array: `[count, id, fields, id, fields, ...]`.
fn parse_ft_search_resp2(response: &[redis::Value]) -> Vec<(i64, f64)> {
    let mut results = Vec::new();
    // First element is total count.
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

/// Extract vector_score from field values array
fn extract_vector_score(fields: &[redis::Value]) -> f64 {
    // Fields are in format: [field_name, field_value, ...]
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
    config: &RedisEngineConfig,
    id: i64,
    vector: &[f32],
    metadata: Option<&MetadataItem>,
) -> Result<(), String> {
    let key = id.to_string();
    let vec_bytes = encode_vector(&config.data_type, vector);

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

/// Format a string metadata field for storage. `datetime` schema fields whose
/// value is an ISO-8601 string are converted to epoch seconds so the NUMERIC
/// index/range filters match; every other field is stored verbatim. A value
/// that is already numeric (epoch) is left as-is (RFC3339 parse fails → raw).
fn encode_string_field(config: &RedisEngineConfig, key: &str, value: &str) -> String {
    if config.datetime_fields.contains(key) {
        if let Some(epoch) = datetime_to_epoch_secs(value) {
            return (epoch as i64).to_string();
        }
    }
    value.to_string()
}

/// Collect the schema field names typed `datetime`.
fn datetime_fields_from_schema(dataset: &Dataset) -> HashSet<String> {
    let mut set = HashSet::new();
    if let Some(schema) = dataset.config.schema.as_ref().and_then(|s| s.as_object()) {
        for (field, ty) in schema {
            if ty.as_str() == Some("datetime") {
                set.insert(field.clone());
            }
        }
    }
    set
}

/// Parse an ISO-8601 / RFC 3339 timestamp to epoch **seconds**. Returns `None`
/// for non-datetime strings (e.g. a plain numeric-epoch string), letting callers
/// fall back to numeric handling.
fn datetime_to_epoch_secs(s: &str) -> Option<f64> {
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|dt| dt.timestamp() as f64)
}

// ── Engine trait implementation ──────────────────────────────────────────

impl Engine for RedisEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        let mut conn = self.get_connection()?;

        // Record which schema fields are `datetime` so upload can convert their
        // ISO-8601 payload values to epoch seconds for the NUMERIC index.
        self.config.datetime_fields = Arc::new(datetime_fields_from_schema(dataset));

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

        // Wait for RediSearch indexing to complete. The index-build wait is part
        // of the ingest cost and must be included in total_time for cross-engine
        // comparability (mirrors mongodb; matches v0 which times through
        // post_upload()). Excluding it made every engine but mongodb look
        // artificially fast to index.
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
        let hybrid_policy = std::env::var("REDIS_HYBRID_POLICY").unwrap_or_default();
        let query_timeout: i64 = std::env::var("REDIS_QUERY_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(90_000);

        // Read queries, ground truth, and filter conditions
        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        // Parse all conditions up front (before timing begins)
        let parsed_filters: Vec<Option<ParsedFilter>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_conditions))
            .collect();

        // When top is explicitly set, use it for all queries.
        // When not set, use per-query ground truth count (matches Python v0 behavior
        // where top defaults to len(query.expected_result) per query).
        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        // Search execution. Each worker accumulates samples into thread-local
        // buffers and returns them on join; the main thread concatenates. This
        // keeps the timed hot loop free of the per-query cross-thread Mutex<Vec>
        // pushes (5 locks/query) that serialized workers at high parallelism.
        // Metrics are order-independent (percentiles sort), so results are
        // unchanged. The work counter stays a single atomic but uses Relaxed
        // (only its own monotonicity matters, not ordering against other memory).
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);
        let start_time = Instant::now();

        let mut times: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut precs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut recs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut mrs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut nds: Vec<f64> = Vec::with_capacity(num_to_run);

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(parallel);
            for _ in 0..parallel {
                let redis_url = self.redis_url.clone();
                let algorithm = self.config.algorithm.clone();
                let data_type = self.config.data_type.clone();
                let hybrid_policy = hybrid_policy.clone();
                let queries = &queries;
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                handles.push(s.spawn(move || {
                    // Thread-local sample buffers — no cross-thread synchronization
                    // in the timed loop.
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();

                    let client = match redis::Client::open(redis_url.as_str()) {
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

                        // Per-query top: use explicit top if set, otherwise
                        // use ground truth count (matches Python v0 behavior)
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
                            &data_type,
                        );
                        let query_time = query_start.elapsed().as_secs_f64();

                        // Record latency + quality only for successful queries. A
                        // failed query is surfaced and counted (as num_to_run minus
                        // successes after the loop), never folded into the means as
                        // a 0-recall / fast-latency sample.
                        match results {
                            Ok(result_ids) => {
                                let ordered_ids: Vec<i64> =
                                    result_ids.iter().map(|(id, _)| *id).collect();
                                let m = compute_metrics(&ordered_ids, &neighbors[idx], top);
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
                mrs.extend(mr);
                nds.extend(nd);
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
            &times, &precs, &recs, &mrs, &nds, total_time, top, parallel, num_to_run,
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
        let hybrid_policy = std::env::var("REDIS_HYBRID_POLICY").unwrap_or_default();
        let query_timeout: i64 = std::env::var("REDIS_QUERY_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(90_000);

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
                let redis_url = self.redis_url.clone();
                let config = self.config.clone();
                let algorithm = self.config.algorithm.clone();
                let data_type = self.config.data_type.clone();
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
                    let client = match redis::Client::open(redis_url.as_str()) {
                        Ok(c) => c,
                        Err(_) => return,
                    };
                    let mut conn = match client.get_connection() {
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
                                &data_type,
                            );
                            let query_time = query_start.elapsed().as_secs_f64();

                            // Record latency + quality only for successful queries;
                            // failures are surfaced, never scored as 0-recall.
                            match results {
                                Ok(result_ids) => {
                                    search_times.lock().unwrap().push(query_time);
                                    let ordered_ids: Vec<i64> =
                                        result_ids.iter().map(|(id, _)| *id).collect();
                                    let m = compute_metrics(&ordered_ids, &neighbors[idx], top);
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
        let mrs = mrrs.lock().unwrap();
        let nds = ndcgs.lock().unwrap();
        let u_times = update_times.lock().unwrap();

        if times.is_empty() {
            return Err("No searches completed".to_string());
        }

        let rps = times.len() as f64 / total_time;
        let mean_precision = precs.iter().sum::<f64>() / precs.len() as f64;
        let mean_recall = recs.iter().sum::<f64>() / recs.len() as f64;
        let mean_mrr = mrs.iter().sum::<f64>() / mrs.len() as f64;
        let mean_ndcg = nds.iter().sum::<f64>() / nds.len() as f64;
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
            std_time,
            min_time,
            max_time,
            rps,
            p50_time,
            p95_time,
            p99_time,
            precisions: precs.to_vec(),
            recalls: recs.to_vec(),
            mrrs: mrs.to_vec(),
            ndcgs: nds.to_vec(),
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
        let _ = redis::cmd("FT.DROPINDEX")
            .arg("idx")
            .arg("DD")
            .query::<()>(&mut conn);
        Ok(())
    }

    fn get_memory_usage(&mut self) -> Option<serde_json::Value> {
        let mut conn = self.get_connection().ok()?;

        // Get used_memory from INFO memory (returns bulk string, parse key:value lines)
        let info_str: String = redis::cmd("INFO").arg("memory").query(&mut conn).ok()?;
        let used_memory: i64 = info_str
            .lines()
            .find(|l| l.starts_with("used_memory:"))
            .and_then(|l| l.strip_prefix("used_memory:"))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0);

        // Get FT.INFO for index stats
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
    use super::{encode_vector, parse_conditions, FilterParamValue};

    #[test]
    fn match_any_string_list_emits_tag_or() {
        let cond = serde_json::json!({"and":[{"color":{"match":{"any":["red","blue"]}}}]});
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@color:{$color_0 | $color_1}"), "q={}", q);
        assert!(matches!(params.get("color_0"), Some(FilterParamValue::Str(s)) if s == "red"));
        assert!(matches!(params.get("color_1"), Some(FilterParamValue::Str(s)) if s == "blue"));
    }

    #[test]
    fn match_any_int_list_emits_numeric_or() {
        let cond = serde_json::json!({"and":[{"size":{"match":{"any":[1,2]}}}]});
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@size:[$size_0 $size_0]"), "q={}", q);
        assert!(q.contains("@size:[$size_1 $size_1]"), "q={}", q);
        assert!(q.contains(" | "), "q={}", q);
        assert!(matches!(
            params.get("size_0"),
            Some(FilterParamValue::Int(1))
        ));
        assert!(matches!(
            params.get("size_1"),
            Some(FilterParamValue::Int(2))
        ));
    }

    #[test]
    fn match_any_empty_list_matches_nothing() {
        // Empty IN-set -> never-match contradiction, not a dropped clause.
        let cond = serde_json::json!({"and":[{"color":{"match":{"any":[]}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("-@color:{"), "expected never-match, q={}", q);
    }

    #[test]
    fn match_any_drops_empty_string_tokens() {
        // Empty-string tokens are invalid TAG syntax; with only such tokens the
        // clause degrades to the never-match rather than emitting `@f:{}`.
        let cond = serde_json::json!({"and":[{"color":{"match":{"any":[""]}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(!q.contains("{$color_0 }") && !q.contains("{}"), "q={}", q);
        assert!(q.contains("-@color:{"), "q={}", q);
    }

    #[test]
    fn match_exact_value_still_emits_tag() {
        let cond = serde_json::json!({"and":[{"color":{"match":{"value":"red"}}}]});
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@color:{$color_0}"), "q={}", q);
        assert!(matches!(params.get("color_0"), Some(FilterParamValue::Str(s)) if s == "red"));
    }

    // ── New filter datatypes: bool / uuid / full-text / datetime ───────────
    use super::{datetime_to_epoch_secs, encode_string_field, RedisEngineConfig};
    use std::collections::HashSet;
    use std::sync::Arc;

    #[test]
    fn match_bool_true_emits_tag_true_token() {
        let cond = serde_json::json!({"and":[{"flag":{"match":{"value": true}}}]});
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@flag:{$flag_0}"), "q={}", q);
        assert!(matches!(params.get("flag_0"), Some(FilterParamValue::Str(s)) if s == "true"));
    }

    #[test]
    fn match_bool_false_emits_tag_false_token() {
        let cond = serde_json::json!({"and":[{"flag":{"match":{"value": false}}}]});
        let (_q, params) = parse_conditions(&cond).unwrap();
        assert!(matches!(params.get("flag_0"), Some(FilterParamValue::Str(s)) if s == "false"));
    }

    #[test]
    fn match_uuid_value_emits_tag_param() {
        // uuid is a keyword-style string → TAG param match.
        let uuid = "550e8400-e29b-41d4-a716-446655440000";
        let cond = serde_json::json!({"and":[{"uid":{"match":{"value": uuid}}}]});
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@uid:{$uid_0}"), "q={}", q);
        assert!(matches!(params.get("uid_0"), Some(FilterParamValue::Str(s)) if s == uuid));
    }

    #[test]
    fn match_text_emits_fulltext_clause() {
        let cond = serde_json::json!({"and":[{"body":{"match":{"text": "quick"}}}]});
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@body:($body_0)"), "q={}", q);
        assert!(matches!(params.get("body_0"), Some(FilterParamValue::Str(s)) if s == "quick"));
    }

    #[test]
    fn match_text_empty_is_never_match() {
        let cond = serde_json::json!({"and":[{"body":{"match":{"text": "   "}}}]});
        let (q, _) = parse_conditions(&cond).unwrap();
        assert!(q.contains("-@body:("), "expected never-match, q={}", q);
    }

    #[test]
    fn range_datetime_iso_bounds_parse_to_epoch() {
        let cond = serde_json::json!({"and":[{"ts":{"range":{
            "gte": "2021-01-01T00:00:00Z",
            "lt":  "2022-01-01T00:00:00Z"
        }}}]});
        let (q, params) = parse_conditions(&cond).unwrap();
        assert!(q.contains("@ts:[$ts_0_gte +inf]"), "q={}", q);
        assert!(q.contains("@ts:[-inf ($ts_0_lt]"), "q={}", q);
        // 2021-01-01T00:00:00Z == 1609459200, 2022-01-01T00:00:00Z == 1640995200.
        assert!(
            matches!(params.get("ts_0_gte"), Some(FilterParamValue::Float(f)) if (*f - 1609459200.0).abs() < 1.0)
        );
        assert!(
            matches!(params.get("ts_0_lt"), Some(FilterParamValue::Float(f)) if (*f - 1640995200.0).abs() < 1.0)
        );
    }

    #[test]
    fn range_datetime_numeric_epoch_bounds_still_work() {
        // Numeric-epoch bounds (better-than-upstream: upstream is ISO-only).
        let cond = serde_json::json!({"and":[{"ts":{"range":{"gte": 1609459200}}}]});
        let (_q, params) = parse_conditions(&cond).unwrap();
        assert!(matches!(
            params.get("ts_0_gte"),
            Some(FilterParamValue::Int(1609459200))
        ));
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
    fn encode_string_field_converts_datetime_and_passes_through_others() {
        let mut dt = HashSet::new();
        dt.insert("ts".to_string());
        let cfg = RedisEngineConfig {
            m: 16,
            ef_construction: 128,
            data_type: "FLOAT32".to_string(),
            algorithm: "hnsw".to_string(),
            batch_size: 1,
            parallel: 1,
            skip_vector_index: false,
            datetime_fields: Arc::new(dt),
        };
        // datetime field → epoch seconds string
        assert_eq!(
            encode_string_field(&cfg, "ts", "2021-01-01T00:00:00Z"),
            "1609459200"
        );
        // datetime field already numeric epoch → left as-is
        assert_eq!(encode_string_field(&cfg, "ts", "1609459200"), "1609459200");
        // non-datetime field → verbatim
        assert_eq!(encode_string_field(&cfg, "color", "red"), "red");
    }

    #[test]
    fn encodes_float32_by_default() {
        let v = vec![1.0f32, -2.5];
        let bytes = encode_vector("FLOAT32", &v);
        assert_eq!(bytes.len(), 8); // 2 x f32
        assert_eq!(bytes, encode_vector("SOMETHING_UNKNOWN", &v));
        assert_eq!(&bytes[0..4], &1.0f32.to_le_bytes());
    }

    #[test]
    fn encodes_int8_one_byte_per_element_with_rounding() {
        // Values arrive as f32 (e.g. pre-quantized cohere int8 embeddings).
        let v = vec![0.0f32, 1.4, -1.6, 127.0, -128.0];
        let bytes = encode_vector("INT8", &v);
        assert_eq!(bytes.len(), 5); // 1 byte per element
        let decoded: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        assert_eq!(decoded, vec![0, 1, -2, 127, -128]);
    }

    #[test]
    fn int8_saturates_out_of_range() {
        let bytes = encode_vector("INT8", &[200.0, -200.0]);
        let decoded: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
        assert_eq!(decoded, vec![127, -128]);
    }

    #[test]
    fn encodes_uint8_clamped_to_0_255() {
        let bytes = encode_vector("UINT8", &[0.0, 255.0, 300.0, -5.0, 12.6]);
        assert_eq!(bytes, vec![0, 255, 255, 0, 13]);
    }

    // ── redis::Value (RESP FT.SEARCH) response parsing ─────────────────────
    // Guards the manual RESP-array parsing against a redis-crate / RESP2-vs-RESP3
    // change (the surface most exposed by the redis-rs 0.27 → 1.x upgrade).
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
            Value::Int(42),
            Value::Array(vec![bulk("vector_score"), bulk("1.5")]),
        ]);
        let hits = parse_ft_search_response(&resp).unwrap();
        assert_eq!(hits, vec![(7, 0.25), (42, 1.5)]);
    }

    #[test]
    fn parse_ft_search_response_resp3_map_reads_results() {
        // RESP3 FT.SEARCH shape: a map whose `results` is an array of per-doc
        // maps with `id` + `extra_attributes` (carrying vector_score).
        let doc = |id: &str, score: &str| {
            Value::Map(vec![
                (bulk("id"), bulk(id)),
                (
                    bulk("extra_attributes"),
                    Value::Map(vec![(bulk("vector_score"), bulk(score))]),
                ),
                (bulk("values"), Value::Array(vec![])),
            ])
        };
        let resp = Value::Map(vec![
            (bulk("attributes"), Value::Array(vec![])),
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
        // RESP3 map with no `results` key → no hits.
        assert_eq!(
            parse_ft_search_response(&Value::Map(vec![(bulk("total_results"), Value::Int(0))]))
                .unwrap(),
            vec![]
        );
        let resp = Value::Array(vec![Value::Int(1), Value::Nil, Value::Nil]);
        assert_eq!(parse_ft_search_response(&resp).unwrap(), vec![(0, 0.0)]);
    }

    #[test]
    fn extract_vector_score_finds_field_or_defaults_zero() {
        let fields = vec![bulk("vector_score"), bulk("0.75")];
        assert!((extract_vector_score(&fields) - 0.75).abs() < 1e-9);
        assert_eq!(extract_vector_score(&[bulk("other"), bulk("x")]), 0.0);
    }

    #[test]
    fn redis_value_to_json_covers_scalars_and_fallthrough() {
        assert_eq!(redis_value_to_json(&Value::Int(5)), serde_json::json!(5));
        assert_eq!(redis_value_to_json(&bulk("hi")), serde_json::json!("hi"));
        assert_eq!(
            redis_value_to_json(&Value::Array(vec![Value::Int(1), bulk("a")])),
            serde_json::json!([1, "a"])
        );
        // Non-exhaustive/other variant → non-empty JSON string, never dropped.
        let okay = redis_value_to_json(&Value::Okay);
        assert!(okay.as_str().map(|s| !s.is_empty()).unwrap_or(false));
    }
}
