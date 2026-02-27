//! VectorSets-rs engine implementation.
//!
//! Implements the Engine trait for Redis VectorSets (VADD/VSIM commands).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use redis::Connection;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};

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
    host: String,
    port: u16,
    config: VectorSetsConfig,
    search_params: Vec<SearchParams>,
}

impl VectorSetsEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("REDIS_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6379);

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
            host: host.to_string(),
            port,
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
        let auth = std::env::var("REDIS_AUTH").ok();
        let user = std::env::var("REDIS_USER").ok();
        let auth_part = match (&user, &auth) {
            (Some(u), Some(p)) => format!("{}:{}@", u, p),
            (None, Some(p)) => format!(":{}@", p),
            _ => String::new(),
        };
        let url = format!("redis://{}{}:{}/", auth_part, self.host, self.port);
        let client = redis::Client::open(url.as_str()).map_err(|e| e.to_string())?;
        client.get_connection().map_err(|e| e.to_string())
    }

    fn upload_sequential(&self, ids: &[i64], vectors: &[Vec<f32>]) -> Result<(), String> {
        let mut conn = self.get_connection()?;
        let pb = self.create_progress_bar(ids.len());

        for batch_start in (0..ids.len()).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(ids.len());
            vadd_batch(
                &mut conn,
                &self.config,
                &ids[batch_start..batch_end],
                &vectors[batch_start..batch_end],
            )?;
            pb.inc((batch_end - batch_start) as u64);
        }

        pb.finish_with_message("Upload complete");
        Ok(())
    }

    fn upload_parallel(&self, ids: &[i64], vectors: &[Vec<f32>]) -> Result<(), String> {
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
                let host = self.host.clone();
                let port = self.port;
                let config = self.config.clone();
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let auth = std::env::var("REDIS_AUTH").ok();
                    let user = std::env::var("REDIS_USER").ok();
                    let auth_part = match (&user, &auth) {
                        (Some(u), Some(p)) => format!("{}:{}@", u, p),
                        (None, Some(p)) => format!(":{}@", p),
                        _ => String::new(),
                    };
                    let url = format!("redis://{}{}:{}/", auth_part, host, port);
                    let client = match redis::Client::open(url.as_str()) {
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
/// VADD idx FP32 <vec_bytes> <id> <quant> M <M> EF <EF_CONSTRUCTION> CAS
fn vadd_batch(
    conn: &mut Connection,
    config: &VectorSetsConfig,
    ids: &[i64],
    vectors: &[Vec<f32>],
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

        pipe.add_command(cmd);
    }

    pipe.query::<()>(conn)
        .map_err(|e| format!("VADD batch error: {}", e))?;
    Ok(())
}

/// Execute VSIM query and return (id, score) pairs.
/// VSIM idx FP32 <vec_bytes> WITHSCORES COUNT <top> EF <ef>
/// Response: alternating [id, score, id, score, ...]
/// Score conversion: 1.0 - score (VectorSets: 1=identical, 0=opposite)
fn vsim_search(
    conn: &mut Connection,
    query_vector: &[f32],
    top: usize,
    ef: i64,
) -> Result<Vec<(i64, f64)>, String> {
    let vec_bytes: Vec<u8> = query_vector.iter().flat_map(|f| f.to_le_bytes()).collect();

    let response: Vec<redis::Value> = redis::cmd("VSIM")
        .arg("idx")
        .arg("FP32")
        .arg(&vec_bytes[..])
        .arg("WITHSCORES")
        .arg("COUNT")
        .arg(top)
        .arg("EF")
        .arg(ef)
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
            self.upload_sequential(&ids, &vectors)?;
        } else {
            self.upload_parallel(&ids, &vectors)?;
        }

        let upload_time = upload_start.elapsed().as_secs_f64();
        let total_time = read_time + upload_time;

        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );
        println!("Total time: {:.3}s", total_time);

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
        let parallel = params.parallel.unwrap_or(1) as usize;

        // Read queries and ground truth
        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, _conditions) = dataset.read_queries()?;

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
                let host = self.host.clone();
                let port = self.port;
                let queries = &queries;
                let neighbors = &neighbors;
                let search_times = Arc::clone(&search_times);
                let precisions = Arc::clone(&precisions);
                let query_idx = Arc::clone(&query_idx);

                s.spawn(move || {
                    let auth = std::env::var("REDIS_AUTH").ok();
                    let user = std::env::var("REDIS_USER").ok();
                    let auth_part = match (&user, &auth) {
                        (Some(u), Some(p)) => format!("{}:{}@", u, p),
                        (None, Some(p)) => format!(":{}@", p),
                        _ => String::new(),
                    };
                    let url = format!("redis://{}{}:{}/", auth_part, host, port);
                    let client = match redis::Client::open(url.as_str()) {
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
                        let results = vsim_search(&mut conn, &queries[idx], top, ef);
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
