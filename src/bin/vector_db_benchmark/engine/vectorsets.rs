//! VectorSets-rs engine implementation.
//!
//! Implements the Engine trait for Redis VectorSets (VADD/VSIM commands).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use rand::seq::SliceRandom;
use rand::SeedableRng;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use redis::Connection;

use super::redis_utils;
use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UpdateSearchRatio, UploadStats};
use vector_db_benchmark::readers::metadata::{MetadataItem, MetadataValue};

/// VectorSets engine configuration
#[derive(Clone)]
pub struct VectorSetsConfig {
    pub quant: String,
    pub m: i64,
    pub ef_construction: i64,
    pub cas: bool,
    pub batch_size: usize,
    pub parallel: usize,
}

pub struct VectorSetsEngine {
    name: String,
    redis_url: String,
    config: VectorSetsConfig,
    search_params: Vec<SearchParams>,
}

impl VectorSetsEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let redis_url = crate::engine::build_redis_url(host);

        let hnsw_config = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("hnsw_config"))
            .and_then(|v| v.as_object());

        let quant = hnsw_config
            .and_then(|h| h.get("quant"))
            .and_then(|v| v.as_str())
            .unwrap_or("NOQUANT")
            .to_string();

        let m = hnsw_config
            .and_then(|h| h.get("M"))
            .and_then(|v| v.as_i64())
            .unwrap_or(16);

        let ef_construction = hnsw_config
            .and_then(|h| h.get("EF_CONSTRUCTION"))
            .and_then(|v| v.as_i64())
            .unwrap_or(200);

        let cas = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("CAS"))
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let parallel = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("parallel"))
            .and_then(|v| v.as_i64())
            .unwrap_or(32) as usize;

        let batch_size = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("batch_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(64) as usize;

        Ok(Self {
            name: engine_config.name.clone(),
            redis_url,
            config: VectorSetsConfig {
                quant,
                m,
                ef_construction,
                cas,
                batch_size,
                parallel,
            },
            search_params: engine_config.search_params.clone().unwrap_or_default(),
        })
    }

    fn get_connection(&self) -> Result<Connection, String> {
        let client = redis::Client::open(self.redis_url.as_str()).map_err(|e| e.to_string())?;
        client.get_connection().map_err(|e| e.to_string())
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
            vadd_batch(
                &mut conn,
                &self.config,
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
                        if let Err(e) = vadd_batch(
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

/// Execute a batch of VADD commands via pipeline.
/// VADD idx FP32 <vec_bytes> <id> <quant> M <M> EF <EF_CONSTRUCTION> [CAS] [SETATTR '<json>']
fn vadd_batch(
    conn: &mut Connection,
    config: &VectorSetsConfig,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
) -> Result<(), String> {
    let mut pipe = redis::pipe();

    for i in 0..ids.len() {
        let vec_bytes: Vec<u8> = vectors[i].iter().flat_map(|f| f.to_le_bytes()).collect();

        let mut cmd = redis::cmd("VADD");
        cmd.arg("idx")
            .arg("FP32")
            .arg(&vec_bytes[..])
            .arg(ids[i].to_string())
            .arg(&config.quant)
            .arg("M")
            .arg(config.m)
            .arg("EF")
            .arg(config.ef_construction);

        if config.cas {
            cmd.arg("CAS");
        }

        // Attach metadata as JSON attributes for FILTER support
        if let Some(meta) = &metadata[i] {
            if !meta.fields.is_empty() {
                let mut map = serde_json::Map::new();
                for (k, v) in &meta.fields {
                    match v {
                        MetadataValue::String(s) => {
                            map.insert(k.clone(), serde_json::Value::String(s.clone()));
                        }
                        MetadataValue::Labels(labels) => {
                            map.insert(
                                k.clone(),
                                serde_json::Value::Array(
                                    labels
                                        .iter()
                                        .map(|l| serde_json::Value::String(l.clone()))
                                        .collect(),
                                ),
                            );
                        }
                        MetadataValue::Geo { lon, lat } => {
                            map.insert(
                                k.clone(),
                                serde_json::json!({"lon": lon, "lat": lat}),
                            );
                        }
                    }
                }
                let json_str = serde_json::Value::Object(map).to_string();
                cmd.arg("SETATTR").arg(json_str);
            }
        }

        pipe.add_command(cmd);
    }

    pipe.query::<()>(conn)
        .map_err(|e| format!("VADD batch error: {}", e))?;
    Ok(())
}

/// Execute VSIM query and return (id, score) pairs.
/// VSIM idx FP32 <vec_bytes> WITHSCORES COUNT <top> EF <ef> [FILTER '<expr>' [FILTER-EF <n>]]
/// Response: alternating [id, score, id, score, ...]
/// Score conversion: 1.0 - score (VectorSets: 1=identical, 0=opposite)
fn vsim_search(
    conn: &mut Connection,
    query_vector: &[f32],
    top: usize,
    ef: i64,
    filter: Option<&str>,
    filter_ef: Option<i64>,
) -> Result<Vec<(i64, f64)>, String> {
    let vec_bytes: Vec<u8> = query_vector.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut cmd = redis::cmd("VSIM");
    cmd.arg("idx")
        .arg("FP32")
        .arg(&vec_bytes[..])
        .arg("WITHSCORES")
        .arg("COUNT")
        .arg(top)
        .arg("EF")
        .arg(ef);

    if let Some(filter_expr) = filter {
        cmd.arg("FILTER").arg(filter_expr);
        // FILTER-EF controls how many candidate nodes the engine inspects
        // for filtered results. Default is COUNT * 100.
        // Docs say FILTER-EF 0 means "scan as many as needed", but
        // Redis 8.6.0 rejects 0 with "invalid FILTER-EF".
        // Workaround: use a very large value (10M) to approximate unlimited.
        // Configurable via search_params.filter_ef in experiment JSON.
        let fe = filter_ef.unwrap_or(10_000_000);
        cmd.arg("FILTER-EF").arg(fe);
    }

    let response: Vec<redis::Value> = cmd
        .query(conn)
        .map_err(|e| format!("VSIM error: {}", e))?;

    // Parse alternating [id, score, id, score, ...]
    let mut results = Vec::new();
    let mut i = 0;
    while i + 1 < response.len() {
        let id = match &response[i] {
            redis::Value::BulkString(data) => {
                String::from_utf8_lossy(data).parse::<i64>().unwrap_or(0)
            }
            redis::Value::Int(n) => *n,
            _ => 0,
        };

        let score = match &response[i + 1] {
            redis::Value::BulkString(data) => {
                let s = String::from_utf8_lossy(data).parse::<f64>().unwrap_or(0.0);
                1.0 - s // VectorSets: 1=identical, convert to distance
            }
            redis::Value::Double(f) => 1.0 - f,
            _ => 0.0,
        };

        results.push((id, score));
        i += 2;
    }

    Ok(results)
}

/// Single-record VADD update (for mixed benchmark).
fn vadd_single(
    conn: &mut Connection,
    config: &VectorSetsConfig,
    id: i64,
    vector: &[f32],
    metadata: Option<&MetadataItem>,
) -> Result<(), String> {
    let vec_bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();

    let mut cmd = redis::cmd("VADD");
    cmd.arg("idx")
        .arg("FP32")
        .arg(&vec_bytes[..])
        .arg(id.to_string())
        .arg(&config.quant)
        .arg("M")
        .arg(config.m)
        .arg("EF")
        .arg(config.ef_construction);

    if config.cas {
        cmd.arg("CAS");
    }

    if let Some(meta) = metadata {
        if !meta.fields.is_empty() {
            let mut map = serde_json::Map::new();
            for (k, v) in &meta.fields {
                match v {
                    MetadataValue::String(s) => {
                        map.insert(k.clone(), serde_json::Value::String(s.clone()));
                    }
                    MetadataValue::Labels(labels) => {
                        map.insert(
                            k.clone(),
                            serde_json::Value::Array(
                                labels
                                    .iter()
                                    .map(|l| serde_json::Value::String(l.clone()))
                                    .collect(),
                            ),
                        );
                    }
                    MetadataValue::Geo { lon, lat } => {
                        map.insert(k.clone(), serde_json::json!({"lon": lon, "lat": lat}));
                    }
                }
            }
            let json_str = serde_json::Value::Object(map).to_string();
            cmd.arg("SETATTR").arg(json_str);
        }
    }

    cmd.query::<()>(conn)
        .map_err(|e| format!("VADD update error: {}", e))
}

impl Engine for VectorSetsEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, _dataset: &Dataset) -> Result<(), String> {
        let mut conn = self.get_connection()?;

        println!(
            "Using VectorSets with QUANT: {}, M: {}, EF_CONSTRUCTION: {}, CAS: {}",
            self.config.quant, self.config.m, self.config.ef_construction, self.config.cas
        );

        // Delete existing key if any
        let _ = redis::cmd("DEL").arg("idx").query::<()>(&mut conn);

        redis_utils::reset_commandstats(&mut conn)?;
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
        let total_time = read_time + upload_time;

        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );
        println!("Total time: {:.3}s", total_time);

        // Verify no VADD failures occurred during upload
        let mut conn = self.get_connection()?;
        redis_utils::check_commandstats(&mut conn, &["VADD"], "upload")?;

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
        let ef = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.ef)
            .unwrap_or(64);
        let filter_ef: Option<i64> = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.extra.as_ref())
            .and_then(|extra| extra.get("filter_ef"))
            .and_then(|v| v.as_i64());
        let parallel = params.parallel.unwrap_or(1) as usize;

        // Read queries and ground truth
        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        // Pre-build filter expressions for each query
        let filters: Vec<Option<String>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(build_filter_expression))
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

        let search_times: Arc<Mutex<Vec<f64>>> =
            Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let precisions: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let query_idx = Arc::new(AtomicUsize::new(0));

        let start_time = Instant::now();

        std::thread::scope(|s| {
            for _ in 0..parallel {
                let redis_url = self.redis_url.clone();
                let queries = &queries;
                let neighbors = &neighbors;
                let filters = &filters;
                let search_times = Arc::clone(&search_times);
                let precisions = Arc::clone(&precisions);
                let query_idx = Arc::clone(&query_idx);

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

                        let filter_ref = filters[idx].as_deref();
                        let query_start = Instant::now();
                        let results =
                            vsim_search(&mut conn, &queries[idx], top, ef, filter_ref, filter_ef);
                        let query_time = query_start.elapsed().as_secs_f64();

                        search_times.lock().unwrap().push(query_time);

                        if let Ok(result_ids) = results {
                            let ground_truth: std::collections::HashSet<i64> =
                                neighbors[idx].iter().take(top).copied().collect();
                            let found: std::collections::HashSet<i64> =
                                result_ids.iter().map(|(id, _)| *id).collect();
                            let hits = ground_truth.intersection(&found).count();
                            let precision = hits as f64 / top as f64;
                            precisions.lock().unwrap().push(precision);
                        } else {
                            precisions.lock().unwrap().push(0.0);
                        }
                    }
                });
            }
        });

        let total_time = start_time.elapsed().as_secs_f64();

        let times = search_times.lock().unwrap();
        let precs = precisions.lock().unwrap();

        if times.is_empty() {
            return Err("No searches completed".to_string());
        }

        let rps = times.len() as f64 / total_time;
        let mean_precision = precs.iter().sum::<f64>() / precs.len() as f64;
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

        // Verify no VSIM failures occurred
        let mut check_conn = self.get_connection()?;
        redis_utils::check_commandstats(&mut check_conn, &["VSIM"], "search")?;

        Ok(SearchResults {
            total_time,
            mean_time,
            mean_precision,
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
            parallel,
            ..Default::default()
        })
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
        let filter_ef: Option<i64> = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.extra.as_ref())
            .and_then(|extra| extra.get("filter_ef"))
            .and_then(|v| v.as_i64());
        let parallel = params.parallel.unwrap_or(1) as usize;

        // Read queries and ground truth
        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let filters: Vec<Option<String>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(build_filter_expression))
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
        let search_idx = Arc::new(AtomicUsize::new(0));
        let update_idx = Arc::new(AtomicUsize::new(0));

        let ratio_searches = ratio.searches as usize;
        let ratio_updates = ratio.updates as usize;
        let update_seq_len = update_seq.len();

        let start_time = Instant::now();

        std::thread::scope(|s| {
            for _ in 0..parallel {
                let redis_url = self.redis_url.clone();
                let config = self.config.clone();
                let queries = &queries;
                let neighbors = &neighbors;
                let filters = &filters;
                let upd_ids = &upd_ids;
                let upd_vectors = &upd_vectors;
                let upd_metadata = &upd_metadata;
                let update_seq = &update_seq;
                let search_times = Arc::clone(&search_times);
                let update_times = Arc::clone(&update_times);
                let precisions = Arc::clone(&precisions);
                let search_idx = Arc::clone(&search_idx);
                let update_idx = Arc::clone(&update_idx);

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
                                if n > 0 { n } else { 10 }
                            });

                            let filter_ref = filters[idx].as_deref();
                            let query_start = Instant::now();
                            let results = vsim_search(
                                &mut conn,
                                &queries[idx],
                                top,
                                ef,
                                filter_ref,
                                filter_ef,
                            );
                            let query_time = query_start.elapsed().as_secs_f64();

                            search_times.lock().unwrap().push(query_time);

                            if let Ok(result_ids) = results {
                                let ground_truth: std::collections::HashSet<i64> =
                                    neighbors[idx].iter().take(top).copied().collect();
                                let found: std::collections::HashSet<i64> =
                                    result_ids.iter().map(|(id, _)| *id).collect();
                                let hits = ground_truth.intersection(&found).count();
                                let precision = hits as f64 / top as f64;
                                precisions.lock().unwrap().push(precision);
                            } else {
                                precisions.lock().unwrap().push(0.0);
                            }
                        }

                        // Update phase: do U updates
                        for _ in 0..ratio_updates {
                            let uidx = update_idx.fetch_add(1, Ordering::SeqCst);
                            let data_idx = update_seq[uidx % update_seq_len];

                            let update_start = Instant::now();
                            let _ = vadd_single(
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

        let total_time = start_time.elapsed().as_secs_f64();

        let times = search_times.lock().unwrap();
        let precs = precisions.lock().unwrap();
        let u_times = update_times.lock().unwrap();

        if times.is_empty() {
            return Err("No searches completed".to_string());
        }

        let rps = times.len() as f64 / total_time;
        let mean_precision = precs.iter().sum::<f64>() / precs.len() as f64;
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
                    .get(
                        ((u_sorted.len() as f64 * 0.95) as usize).min(u_sorted.len() - 1),
                    )
                    .copied()
                    .unwrap_or(0.0);
                let u_p99 = u_sorted
                    .get(
                        ((u_sorted.len() as f64 * 0.99) as usize).min(u_sorted.len() - 1),
                    )
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
        redis_utils::check_commandstats(&mut check_conn, &["VSIM", "VADD"], "mixed")?;

        Ok(SearchResults {
            total_time,
            mean_time,
            mean_precision,
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
        let _ = redis::cmd("DEL").arg("idx").query::<()>(&mut conn);
        Ok(())
    }

    fn get_memory_usage(&mut self) -> Option<serde_json::Value> {
        let mut conn = self.get_connection().ok()?;

        let info_str: String = redis::cmd("INFO")
            .arg("memory")
            .query(&mut conn)
            .ok()?;
        let used_memory: i64 = info_str
            .lines()
            .find(|l| l.starts_with("used_memory:"))
            .and_then(|l| l.strip_prefix("used_memory:"))
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0);

        Some(serde_json::json!({
            "used_memory": used_memory,
            "shards": 1,
        }))
    }
}

// ── Filter expression builder ────────────────────────────────────────────
// Converts JSON filter conditions into VectorSets FILTER expression syntax.
// VectorSets uses dot-notation field access with standard comparison operators:
//   .field == "value"   .field > 10   .field in ["a", "b"]

/// Build a VectorSets FILTER expression from JSON conditions.
fn build_filter_expression(conditions: &serde_json::Value) -> Option<String> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    let and_entries = obj.get("and").and_then(|v| v.as_array());
    let or_entries = obj.get("or").and_then(|v| v.as_array());

    let mut parts = Vec::new();

    if let Some(entries) = and_entries {
        let clauses = build_clauses(entries);
        if !clauses.is_empty() {
            parts.push(clauses.join(" and "));
        }
    }

    if let Some(entries) = or_entries {
        let clauses = build_clauses(entries);
        if !clauses.is_empty() {
            parts.push(format!("({})", clauses.join(" or ")));
        }
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" and "))
    }
}

fn build_clauses(entries: &[serde_json::Value]) -> Vec<String> {
    let mut clauses = Vec::new();
    for entry in entries {
        if let Some(entry_obj) = entry.as_object() {
            for (field_name, field_filters) in entry_obj {
                if let Some(filter_obj) = field_filters.as_object() {
                    for (condition_type, criteria) in filter_obj {
                        if let Some(expr) = build_clause(field_name, condition_type, criteria) {
                            clauses.push(expr);
                        }
                    }
                }
            }
        }
    }
    clauses
}

fn build_clause(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
) -> Option<String> {
    match condition_type {
        "match" => build_match_clause(field_name, criteria),
        "range" => build_range_clause(field_name, criteria),
        _ => None,
    }
}

fn build_match_clause(field_name: &str, criteria: &serde_json::Value) -> Option<String> {
    let value = criteria.get("value")?;
    if let Some(s) = value.as_str() {
        // Escape double quotes in the string value
        let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
        Some(format!(".{} == \"{}\"", field_name, escaped))
    } else if let Some(n) = value.as_i64() {
        Some(format!(".{} == {}", field_name, n))
    } else if let Some(f) = value.as_f64() {
        Some(format!(".{} == {}", field_name, f))
    } else {
        None
    }
}

fn build_range_clause(field_name: &str, criteria: &serde_json::Value) -> Option<String> {
    let mut parts = Vec::new();

    if let Some(gt) = criteria.get("gt") {
        parts.push(format!(".{} > {}", field_name, format_number(gt)));
    }
    if let Some(gte) = criteria.get("gte") {
        parts.push(format!(".{} >= {}", field_name, format_number(gte)));
    }
    if let Some(lt) = criteria.get("lt") {
        parts.push(format!(".{} < {}", field_name, format_number(lt)));
    }
    if let Some(lte) = criteria.get("lte") {
        parts.push(format!(".{} <= {}", field_name, format_number(lte)));
    }

    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" and "))
    }
}

fn format_number(value: &serde_json::Value) -> String {
    if let Some(i) = value.as_i64() {
        i.to_string()
    } else if let Some(f) = value.as_f64() {
        f.to_string()
    } else {
        "0".to_string()
    }
}
