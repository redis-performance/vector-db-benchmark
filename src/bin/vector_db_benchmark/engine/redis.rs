//! Redis-rs engine implementation.
//!
//! Implements the Engine trait for RediSearch vector similarity.

use std::collections::HashMap;
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
                        "keyword" => {
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
                        fields.push((k.as_bytes().to_vec(), s.as_bytes().to_vec()));
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
            // match_any (IN-list) takes precedence over exact {value}.
            if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                Some(build_match_any_filter(field_name, any, counter))
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

/// Insert a JSON number value into the params map.
fn insert_number_param(
    params: &mut HashMap<String, FilterParamValue>,
    name: &str,
    value: &serde_json::Value,
) {
    if let Some(i) = value.as_i64() {
        params.insert(name.to_string(), FilterParamValue::Int(i));
    } else if let Some(f) = value.as_f64() {
        params.insert(name.to_string(), FilterParamValue::Float(f));
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

    let response: Vec<redis::Value> = cmd
        .query(conn)
        .map_err(|e| format!("FT.SEARCH error: {}", e))?;

    parse_ft_search_response(&response)
}

/// Parse FT.SEARCH response into (id, score) pairs.
/// Response format: [total_count, doc_id, [field_values...], doc_id, [field_values...], ...]
fn parse_ft_search_response(response: &[redis::Value]) -> Result<Vec<(i64, f64)>, String> {
    let mut results = Vec::new();
    if response.is_empty() {
        return Ok(results);
    }

    // First element is total count
    let mut i = 1;
    while i < response.len() {
        // doc_id
        let id = match &response[i] {
            redis::Value::BulkString(data) => {
                String::from_utf8_lossy(data).parse::<i64>().unwrap_or(0)
            }
            redis::Value::Int(n) => *n,
            _ => 0,
        };
        i += 1;

        // field values array
        if i < response.len() {
            let score = match &response[i] {
                redis::Value::Array(fields) => extract_vector_score(fields),
                _ => 0.0,
            };
            results.push((id, score));
            i += 1;
        }
    }

    Ok(results)
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
                    cmd.arg(k.as_str()).arg(s.as_str());
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

        // Search execution
        let search_times: Arc<Mutex<Vec<f64>>> =
            Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let precisions: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let recalls: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let mrrs: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let ndcgs: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(num_to_run);
        let start_time = Instant::now();

        std::thread::scope(|s| {
            for _ in 0..parallel {
                let redis_url = self.redis_url.clone();
                let algorithm = self.config.algorithm.clone();
                let data_type = self.config.data_type.clone();
                let hybrid_policy = hybrid_policy.clone();
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
                    let client = match redis::Client::open(redis_url.as_str()) {
                        Ok(c) => c,
                        Err(_) => return,
                    };
                    let mut conn = match client.get_connection() {
                        Ok(c) => c,
                        Err(_) => return,
                    };

                    loop {
                        let idx = query_idx.fetch_add(1, Ordering::SeqCst);
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
                });
            }
        });

        pb.finish_and_clear();
        let total_time = start_time.elapsed().as_secs_f64();

        // Calculate statistics
        let times = search_times.lock().unwrap();
        let precs = precisions.lock().unwrap();
        let recs = recalls.lock().unwrap();
        let mrs = mrrs.lock().unwrap();
        let nds = ndcgs.lock().unwrap();

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
}
