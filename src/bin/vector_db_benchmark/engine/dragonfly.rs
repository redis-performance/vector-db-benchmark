//! Dragonfly engine implementation (Dragonfly Search — Beta).
//!
//! Dragonfly is a Redis-wire-compatible datastore. Dragonfly Search (v1.13+)
//! implements a RediSearch-compatible `FT.*` subset — `FT.CREATE`, `FT.SEARCH`,
//! `FT.INFO`, `FT.DROPINDEX` — with `VECTOR` fields (FLAT / HNSW) and the KNN
//! query syntax `*=>[KNN k @field $blob AS score]`. This engine speaks that
//! subset over the `redis` crate (same RESP protocol).
//!
//! # Scope: pure vector KNN only
//!
//! Dragonfly Search is Beta. This engine deliberately implements **only**
//! unfiltered vector KNN — no metadata filters, no full-text, no mixed
//! (search+update) workload, no quantization. Every search runs the `*`
//! (match-all) prefilter.
//!
//! # Vector data type: FLOAT32 only
//!
//! Dragonfly Search supports ONLY the `float32` vector type — no
//! INT8/UINT8/FP16/BF16/FP64. Vectors are therefore always encoded as FLOAT32
//! little-endian bytes.
//!
//! # EF_RUNTIME
//!
//! Dragonfly Search **accepts** the per-query `EF_RUNTIME` HNSW attribute
//! (verified live against `dragonfly:df-v1.38.1`: a KNN query with `EF_RUNTIME
//! $EF` returns results, and a non-numeric `$EF` value is rejected with a query
//! syntax error — proving the attribute is actually parsed, not ignored). It is
//! kept, matching redis.rs / valkey.rs, so the search sweep's `ef` values take
//! effect instead of collapsing to the index default.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use redis::Connection;

use super::redis_utils;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use crate::metrics::compute_metrics;
use vector_db_benchmark::readers::metadata::MetadataItem;

/// Dragonfly engine configuration.
#[derive(Clone)]
pub struct DragonflyEngineConfig {
    pub m: i64,
    pub ef_construction: i64,
    /// Always `FLOAT32` — Dragonfly Search supports no other vector type.
    pub data_type: String,
    pub algorithm: String,
    pub batch_size: usize,
    pub parallel: usize,
}

pub struct DragonflyEngine {
    name: String,
    host: String,
    port: u16,
    config: DragonflyEngineConfig,
    search_params: Vec<SearchParams>,
    commandstats_baseline: Option<redis_utils::CommandStatsBaseline>,
}

impl DragonflyEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("DRAGONFLY_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6385);

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

        // Dragonfly Search only supports float32; ignore any configured override.
        let data_type = "FLOAT32".to_string();

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
            config: DragonflyEngineConfig {
                m,
                ef_construction,
                data_type,
                algorithm,
                batch_size,
                parallel,
            },
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            commandstats_baseline: None,
        })
    }

    fn get_connection(&self) -> Result<Connection, String> {
        Self::connect(&self.host, self.port)
    }

    fn connect(host: &str, port: u16) -> Result<Connection, String> {
        let auth = std::env::var("DRAGONFLY_AUTH").ok();
        let user = std::env::var("DRAGONFLY_USER").ok();

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
            dragonfly_url_suffix()
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

        // Drop existing index if any (ignore "not found"), then flush leftover
        // keys from previous runs.
        let _ = redis::cmd("FT.DROPINDEX").arg("idx").query::<()>(conn);
        let _ = redis::cmd("FLUSHALL").query::<()>(conn);

        let distance_metric = map_distance_metric(distance);

        // Build FT.CREATE: a single VECTOR field `vector`. KNN-only, so no
        // metadata/filter schema fields are declared.
        let mut cmd = redis::cmd("FT.CREATE");
        cmd.arg("idx")
            .arg("ON")
            .arg("HASH")
            .arg("PREFIX")
            .arg("1")
            .arg("");

        cmd.arg("SCHEMA");

        // num_attrs = TYPE+DIM+DISTANCE_METRIC (6) + M (2) + EF_CONSTRUCTION (2).
        let num_attrs = 6 + 2 + 2;
        cmd.arg("vector")
            .arg("VECTOR")
            .arg(self.config.algorithm.to_uppercase())
            .arg(num_attrs);
        cmd.arg("TYPE").arg(&self.config.data_type);
        cmd.arg("DIM").arg(vector_size);
        cmd.arg("DISTANCE_METRIC").arg(distance_metric);
        cmd.arg("M").arg(self.config.m);
        cmd.arg("EF_CONSTRUCTION").arg(self.config.ef_construction);

        cmd.query::<()>(conn)
            .map_err(|e| format!("Failed to create index: {}", e))?;

        Ok(())
    }

    fn upload_sequential(&self, ids: &[i64], vectors: &[Vec<f32>]) -> Result<(), String> {
        let mut conn = self.get_connection()?;
        let pb = self.create_progress_bar(ids.len());

        for batch_start in (0..ids.len()).step_by(self.config.batch_size) {
            let batch_end = (batch_start + self.config.batch_size).min(ids.len());
            upload_batch_internal(
                &mut conn,
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
        let num_threads = self.config.parallel.min(total_batches).max(1);
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        std::thread::scope(|s| {
            for _ in 0..num_threads {
                let host = self.host.clone();
                let port = self.port;
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let mut conn = match DragonflyEngine::connect(&host, port) {
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

    /// Wait until FT.INFO reports num_docs >= expected and indexing is done.
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
            // Default 1.0 (fully indexed) so an FT.INFO that omits the field does
            // not stall the wait; Dragonfly DOES expose percent_indexed.
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

            let mut handle_pair = |key: &str, val: &redis::Value| match key {
                "num_docs" => num_docs = extract_usize(val),
                "indexing" => indexing = indexing || extract_bool_nonzero(val),
                "backfill_in_progress" => indexing = indexing || extract_bool_nonzero(val),
                "percent_indexed" => percent_indexed = extract_f64(val),
                _ => {}
            };

            match &info {
                redis::Value::Array(arr) => {
                    for i in (0..arr.len()).step_by(2) {
                        let key_str = match &arr[i] {
                            redis::Value::BulkString(s) => String::from_utf8_lossy(s).to_string(),
                            redis::Value::SimpleString(s) => s.clone(),
                            _ => continue,
                        };
                        if let Some(val) = arr.get(i + 1) {
                            handle_pair(&key_str, val);
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
                        handle_pair(&key_str, v);
                    }
                }
                _ => {
                    eprintln!("Unexpected FT.INFO response type: {:?}", info);
                }
            }

            // Require the HNSW graph to be fully built (percent_indexed >= 1.0),
            // not just the doc count, so the search sweep never runs against a
            // partially-backfilled graph (which would depress recall).
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
                    "Warning: indexing timeout after {}s (num_docs={}/{}, indexing={}, percent_indexed={:.2})",
                    max_wait, num_docs, expected, indexing, percent_indexed
                );
                return Ok(());
            }

            if start.elapsed().as_secs().is_multiple_of(10) && start.elapsed().as_secs() > 0 {
                println!(
                    "Waiting for indexing: {} docs, indexing={}, percent_indexed={:.2} ({:.0}s)",
                    num_docs,
                    indexing,
                    percent_indexed,
                    start.elapsed().as_secs_f64()
                );
            }

            std::thread::sleep(std::time::Duration::from_millis(500));
        }
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

/// Optional connection-URL suffix so Dragonfly can be benchmarked over RESP3
/// (`DRAGONFLY_PROTOCOL=resp3`). Defaults to RESP2 (empty suffix). The FT.SEARCH
/// response parser handles both shapes, so recall is identical either way.
fn dragonfly_url_suffix() -> &'static str {
    if std::env::var("DRAGONFLY_PROTOCOL")
        .map(|v| v.eq_ignore_ascii_case("resp3"))
        .unwrap_or(false)
    {
        "?protocol=resp3"
    } else {
        ""
    }
}

/// Encode a vector to the FLOAT32 little-endian blob Dragonfly Search expects.
/// Dragonfly Search supports ONLY float32, so this is the single encoding used
/// for both upload and query vectors.
fn encode_query_vector(vector: &[f32]) -> Vec<u8> {
    vector.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Upload one batch of `HSET {id} vector {float32_le_bytes}` via a pipeline.
fn upload_batch_internal(
    conn: &mut Connection,
    ids: &[i64],
    vectors: &[Vec<f32>],
) -> Result<(), String> {
    let mut pipe = redis::pipe();

    for i in 0..ids.len() {
        let key = ids[i].to_string();
        let vec_bytes = encode_query_vector(&vectors[i]);
        let mut hset_cmd = redis::cmd("HSET");
        hset_cmd.arg(key.as_str()).arg("vector").arg(&vec_bytes[..]);
        pipe.add_command(hset_cmd);
    }

    pipe.query::<()>(conn).map_err(|e| e.to_string())?;
    Ok(())
}

/// Map a dataset distance name to the Dragonfly Search `DISTANCE_METRIC` value.
/// Unknown metrics default to `COSINE`. A typo here (e.g. IP→L2) would silently
/// invert ranking, so it is unit-tested.
fn map_distance_metric(distance: &str) -> &'static str {
    match distance.to_lowercase().as_str() {
        "cosine" | "angular" => "COSINE",
        "euclidean" | "l2" => "L2",
        "dot" | "ip" => "IP",
        _ => "COSINE",
    }
}

/// Whether `EF_RUNTIME` should be emitted for the given algorithm.
///
/// `EF_RUNTIME` is an HNSW-only per-query attribute — a FLAT index rejects it
/// with a query syntax error. Gating it (query string, the `EF` PARAM, and the
/// PARAMS count) on HNSW keeps a `"algorithm":"flat"` config usable, mirroring
/// redis.rs.
fn uses_ef_runtime(algorithm: &str) -> bool {
    algorithm.eq_ignore_ascii_case("hnsw")
}

/// Build the FT.SEARCH KNN query string (unfiltered `*` prefilter).
///
/// Pure client-side string formatting, kept OUT of the per-query timed window
/// (precomputed once before the parallel region). `EF_RUNTIME $EF` is emitted
/// only for an HNSW index (verified live) — a per-query attribute FLAT rejects;
/// without it every `ef` in the search sweep runs at the index default. The
/// query vector is bound as `$vec_param`, so this string is identical across all
/// queries.
fn build_knn_query_str(algorithm: &str) -> String {
    if uses_ef_runtime(algorithm) {
        "*=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]".to_string()
    } else {
        "*=>[KNN $K @vector $vec_param AS vector_score]".to_string()
    }
}

/// Execute a Dragonfly FT.SEARCH KNN query, return (id, score) pairs.
///
/// `vec_bytes` and `query_str` are precomputed by the caller BEFORE the timed
/// window; this performs only the arg binding, the `cmd.query` RPC round-trip,
/// and the reply parse.
fn ft_search_knn(
    conn: &mut Connection,
    vec_bytes: &[u8],
    query_str: &str,
    top: usize,
    ef: i64,
    algorithm: &str,
    query_timeout: i64,
) -> Result<Vec<(i64, f64)>, String> {
    let mut cmd = redis::cmd("FT.SEARCH");
    cmd.arg("idx")
        .arg(query_str)
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
        .arg(2) // Dragonfly Search uses DIALECT 2
        .arg("TIMEOUT")
        .arg(query_timeout);

    // Params: vec_param(2) + K(2), plus EF(2) only for HNSW (EF_RUNTIME is
    // HNSW-only; binding it on a FLAT index would be a syntax error).
    let ef_runtime = uses_ef_runtime(algorithm);
    cmd.arg("PARAMS").arg(if ef_runtime { 6 } else { 4 });
    cmd.arg("vec_param").arg(vec_bytes);
    cmd.arg("K").arg(top.to_string());
    if ef_runtime {
        cmd.arg("EF").arg(ef.to_string());
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
/// - RESP3: a map `{ results: [ { id, extra_attributes: { vector_score, .. } } ] }`
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
        let mut id: Option<i64> = None;
        let mut score = 0.0f64;
        for (k, v) in fields {
            match value_as_string(k).as_deref() {
                Some("id") => id = value_as_string(v).and_then(|s| s.parse().ok()),
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
        if let Some(id) = id {
            out.push((id, score));
        }
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

/// Extract the `vector_score` field value from a RESP2 field-values array.
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

/// Parse the `used_memory:` value (bytes) out of an `INFO memory` text block.
/// Returns 0 when the line is absent or unparseable. The `used_memory:` prefix
/// is exact, so it never matches sibling keys like `used_memory_rss:`.
fn parse_used_memory(info: &str) -> i64 {
    info.lines()
        .find(|l| l.starts_with("used_memory:"))
        .and_then(|l| l.strip_prefix("used_memory:"))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0)
}

/// Convert a redis::Value to serde_json::Value for FT.INFO serialization.
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

// ── Engine trait implementation ──────────────────────────────────────────

impl Engine for DragonflyEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        let mut conn = self.get_connection()?;

        println!(
            "Using algorithm {} with config {{'M': {}, 'EF_CONSTRUCTION': {}}}",
            self.config.algorithm, self.config.m, self.config.ef_construction
        );

        self.create_index(&mut conn, dataset)?;
        // NOTE: Dragonfly's INFO commandstats omits the `failed_calls=` field
        // that redis/valkey expose, so the check_commandstats guards in
        // upload()/search() are best-effort no-ops on Dragonfly — they cannot
        // observe server-side command failures. Primary error propagation is
        // still correct: every HSET/FT.SEARCH goes through `cmd.query`, which
        // surfaces an `Err` on failure. The baseline is still reset for parity
        // (and would activate automatically if Dragonfly ever adds failed_calls).
        self.commandstats_baseline = redis_utils::reset_commandstats(&mut conn)?;
        Ok(())
    }

    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        let normalize = dataset.needs_normalization();

        let dataset_path = dataset.get_path()?;
        println!("Reading dataset from {}...", dataset_path.display());
        let read_start = Instant::now();
        // Dragonfly Search is KNN-only here: metadata is read but not indexed.
        let (ids, vectors, _metadata): (Vec<i64>, Vec<Vec<f32>>, Vec<Option<MetadataItem>>) =
            dataset.read_vectors(normalize)?;
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

        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        // Include the index-build wait in total_time for cross-engine
        // comparability (mirrors redis/valkey).
        let expected = vectors.len();
        let index_start = Instant::now();
        self.wait_for_indexing(expected)?;
        let index_time = index_start.elapsed().as_secs_f64();

        let total_time = read_time + upload_time + index_time;
        println!(
            "Index time: {:.3}s, Total time (read+upload+index): {:.3}s",
            index_time, total_time
        );

        // Best-effort HSET failure guard. Inert on Dragonfly (its commandstats
        // has no failed_calls field — see configure()); real HSET errors already
        // propagate as `Err` from the pipelined `cmd.query` above.
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
        let ef = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.ef)
            .unwrap_or(64);
        let parallel = params.parallel.unwrap_or(1) as usize;
        let query_timeout: i64 = std::env::var("DRAGONFLY_QUERY_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(60_000);

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        // KNN-only: filter conditions are ignored (never built into the query).
        let (queries, neighbors, _conditions) = dataset.read_queries()?;

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        // Precompute client-side request construction BEFORE the timed region so
        // the per-query window wraps ONLY the RPC round-trip + reply parse
        // (matching redis.rs/valkey.rs). Encoding the FLOAT32 blob is client
        // work, not server latency. The query string is unfiltered and identical
        // across all queries. Shared read-only across workers.
        let encoded_queries: Vec<Vec<u8>> =
            queries.iter().map(|q| encode_query_vector(q)).collect();
        let algorithm = self.config.algorithm.clone();
        let query_str = build_knn_query_str(&algorithm);

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
                let neighbors = &neighbors;
                let encoded_queries = &encoded_queries;
                let query_str = query_str.as_str();
                let algorithm = algorithm.as_str();
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                handles.push(s.spawn(move || {
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();
                    let mut pb_pending: u64 = 0;

                    let mut conn = match DragonflyEngine::connect(&host, port) {
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

                        // Timed window: precomputed blob + query string are passed
                        // in, so this wraps only the RPC round-trip and reply parse.
                        let query_start = Instant::now();
                        let results = ft_search_knn(
                            &mut conn,
                            &encoded_queries[idx],
                            query_str,
                            top,
                            ef,
                            algorithm,
                            query_timeout,
                        );
                        let query_time = query_start.elapsed().as_secs_f64();

                        match &results {
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
                        pb_pending += 1;
                        if pb_pending >= 256 {
                            pb.inc(pb_pending);
                            pb_pending = 0;
                        }
                    }
                    if pb_pending > 0 {
                        pb.inc(pb_pending);
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

        // Best-effort FT.SEARCH failure guard. Inert on Dragonfly (no
        // failed_calls in commandstats — see configure()); a failing FT.SEARCH
        // already surfaces as `Err` from `ft_search_knn` and is logged +
        // excluded from the stats (num_to_run minus successes).
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

    fn delete(&mut self) -> Result<(), String> {
        let mut conn = self.get_connection()?;
        let _ = redis::cmd("FT.DROPINDEX").arg("idx").query::<()>(&mut conn);
        let _ = redis::cmd("FLUSHALL").query::<()>(&mut conn);
        Ok(())
    }

    fn get_memory_usage(&mut self) -> Option<serde_json::Value> {
        let mut conn = self.get_connection().ok()?;

        let info_str: String = redis::cmd("INFO").arg("memory").query(&mut conn).ok()?;
        let used_memory: i64 = parse_used_memory(&info_str);

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
    use super::*;
    use redis::Value;

    fn bulk(s: &str) -> Value {
        Value::BulkString(s.as_bytes().to_vec())
    }

    #[test]
    fn encode_query_vector_is_float32_le_bytes() {
        let v = vec![1.0f32, -2.5, 3.25];
        let expected: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
        assert_eq!(encode_query_vector(&v), expected);
        // 3 f32 => 12 bytes.
        assert_eq!(encode_query_vector(&v).len(), 12);
    }

    #[test]
    fn encode_query_vector_pins_exact_bytes() {
        // 1.0f32 = 0x3F800000 little-endian => 00 00 80 3F.
        assert_eq!(encode_query_vector(&[1.0]), vec![0x00, 0x00, 0x80, 0x3F]);
    }

    #[test]
    fn map_distance_metric_covers_all_aliases() {
        assert_eq!(map_distance_metric("cosine"), "COSINE");
        assert_eq!(map_distance_metric("angular"), "COSINE");
        assert_eq!(map_distance_metric("euclidean"), "L2");
        assert_eq!(map_distance_metric("l2"), "L2");
        assert_eq!(map_distance_metric("dot"), "IP");
        assert_eq!(map_distance_metric("ip"), "IP");
        assert_eq!(map_distance_metric("L2"), "L2"); // case-insensitive
        assert_eq!(map_distance_metric("unknown"), "COSINE"); // default
    }

    #[test]
    fn build_knn_query_str_is_unfiltered_with_ef_runtime() {
        // HNSW emits EF_RUNTIME (per-query ef sweep); FLAT must NOT (it would be
        // a syntax error on a FLAT index).
        assert_eq!(
            build_knn_query_str("hnsw"),
            "*=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]"
        );
        assert_eq!(
            build_knn_query_str("HNSW"),
            "*=>[KNN $K @vector $vec_param EF_RUNTIME $EF AS vector_score]"
        );
        assert_eq!(
            build_knn_query_str("flat"),
            "*=>[KNN $K @vector $vec_param AS vector_score]"
        );
        assert!(uses_ef_runtime("hnsw") && !uses_ef_runtime("flat"));
    }

    #[test]
    fn parse_ft_search_response_resp2_reads_id_score_pairs() {
        // [count, id, [vector_score, val], id, [vector_score, val]]
        let resp = Value::Array(vec![
            Value::Int(2),
            bulk("7"),
            Value::Array(vec![bulk("vector_score"), bulk("0.5")]),
            bulk("42"),
            Value::Array(vec![bulk("vector_score"), bulk("0.75")]),
        ]);
        let out = parse_ft_search_response(&resp).unwrap();
        assert_eq!(out, vec![(7, 0.5), (42, 0.75)]);
    }

    #[test]
    fn parse_ft_search_response_resp3_map_reads_results() {
        let doc = Value::Map(vec![
            (bulk("id"), bulk("9")),
            (
                bulk("extra_attributes"),
                Value::Map(vec![(bulk("vector_score"), bulk("0.125"))]),
            ),
        ]);
        let resp = Value::Map(vec![(bulk("results"), Value::Array(vec![doc]))]);
        let out = parse_ft_search_response(&resp).unwrap();
        assert_eq!(out, vec![(9, 0.125)]);
    }

    #[test]
    fn parse_ft_search_response_empty_and_unknown_variants() {
        assert!(parse_ft_search_response(&Value::Nil).unwrap().is_empty());
        assert!(parse_ft_search_response(&Value::Int(0)).unwrap().is_empty());
    }

    #[test]
    fn parse_ft_search_resp2_drops_trailing_id_without_fields() {
        // Odd-length body: a trailing id with no field array is dropped.
        let resp = Value::Array(vec![Value::Int(1), bulk("5")]);
        let out = parse_ft_search_response(&resp).unwrap();
        assert!(out.is_empty(), "trailing id without fields must be dropped");
    }

    #[test]
    fn parse_ft_search_resp3_skips_doc_with_unparseable_id() {
        let doc = Value::Map(vec![
            (bulk("id"), bulk("not-a-number")),
            (
                bulk("extra_attributes"),
                Value::Map(vec![(bulk("vector_score"), bulk("0.1"))]),
            ),
        ]);
        let resp = Value::Map(vec![(bulk("results"), Value::Array(vec![doc]))]);
        assert!(parse_ft_search_response(&resp).unwrap().is_empty());
    }

    #[test]
    fn extract_vector_score_finds_field_or_defaults_zero() {
        let fields = vec![bulk("vector_score"), bulk("0.25")];
        assert_eq!(extract_vector_score(&fields), 0.25);
        let none = vec![bulk("other"), bulk("1.0")];
        assert_eq!(extract_vector_score(&none), 0.0);
    }

    #[test]
    fn value_as_i64_reads_variants() {
        assert_eq!(value_as_i64(&bulk("13")), 13);
        assert_eq!(value_as_i64(&Value::Int(-4)), -4);
        assert_eq!(value_as_i64(&Value::SimpleString("8".into())), 8);
        assert_eq!(value_as_i64(&Value::Nil), 0);
    }

    #[test]
    fn parse_used_memory_reads_exact_prefix_only() {
        let info = "# Memory\r\nused_memory:1048576\r\nused_memory_rss:2097152\r\n";
        assert_eq!(parse_used_memory(info), 1048576);
        assert_eq!(parse_used_memory("no memory line here"), 0);
    }
}
