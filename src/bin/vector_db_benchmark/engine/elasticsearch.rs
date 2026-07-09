//! Elasticsearch engine implementation.
//!
//! Uses the official `elasticsearch` crate (async, wrapped with tokio block_on).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use elasticsearch::http::request::JsonBody;
use elasticsearch::http::transport::{SingleNodeConnectionPool, TransportBuilder};
use elasticsearch::indices::{IndicesCreateParts, IndicesDeleteParts, IndicesForcemergeParts};
use elasticsearch::params::WaitForStatus;
use elasticsearch::{BulkParts, Elasticsearch, SearchParts};
use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use uuid::Uuid;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use vector_db_benchmark::readers::metadata::MetadataItem;

/// Elasticsearch engine configuration parsed from JSON
#[derive(Clone)]
struct ElasticsearchConfig {
    m: i64,
    ef_construction: i64,
    batch_size: usize,
    parallel: usize,
}

pub struct ElasticsearchEngine {
    name: String,
    index_name: String,
    #[allow(dead_code)]
    timeout: u64,
    config: ElasticsearchConfig,
    search_params: Vec<SearchParams>,
    /// Base URL for constructing per-thread clients
    base_url: String,
    /// Tokio runtime for async operations
    rt: tokio::runtime::Runtime,
    /// Shared Elasticsearch client
    client: Arc<Elasticsearch>,
}

impl ElasticsearchEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("ELASTIC_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(9200);

        let index_name = std::env::var("ELASTIC_INDEX").unwrap_or_else(|_| "bench".to_string());
        let timeout: u64 = std::env::var("ELASTIC_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300);

        // Extract HNSW config from collection_params.index_options (ES-specific)
        let (m, ef_construction) = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.index_options.as_ref())
            .map(|io| (io.m.unwrap_or(16), io.ef_construction.unwrap_or(100)))
            .unwrap_or((16, 100));

        let parallel = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("parallel"))
            .and_then(|v| v.as_i64())
            .unwrap_or(16) as usize;

        let batch_size = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("batch_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(500) as usize;

        let base_url = build_base_url(host, port);

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

        let client = create_es_client(&base_url, timeout)?;

        Ok(Self {
            name: engine_config.name.clone(),
            index_name,
            timeout,
            config: ElasticsearchConfig {
                m,
                ef_construction,
                batch_size,
                parallel,
            },
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            base_url,
            rt,
            client: Arc::new(client),
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

    fn delete_index(&self) -> Result<(), String> {
        let resp = self
            .rt
            .block_on(
                self.client
                    .indices()
                    .delete(IndicesDeleteParts::Index(&[&self.index_name]))
                    .send(),
            )
            .map_err(|e| format!("Failed to delete index: {}", e))?;

        let status = resp.status_code().as_u16();
        if status == 200 || status == 404 {
            Ok(())
        } else {
            Err(format!("Failed to delete index: status {}", status))
        }
    }

    fn create_index(&self, dataset: &Dataset) -> Result<(), String> {
        let distance = dataset.distance();
        let vector_size = dataset.vector_size();

        let dist_lower = distance.to_lowercase();
        if dist_lower == "dot" || dist_lower == "ip" {
            return Err(
                "Elasticsearch does not support DOT product distance for benchmarking".to_string(),
            );
        }
        if vector_size > 2048 {
            return Err(format!(
                "Elasticsearch does not support vector_size > 2048 (got {})",
                vector_size
            ));
        }

        let similarity = match dist_lower.as_str() {
            "l2" | "euclidean" => "l2_norm",
            "cosine" | "angular" => "cosine",
            other => {
                return Err(format!(
                    "Unsupported distance metric for Elasticsearch: {}",
                    other
                ))
            }
        };

        let mut properties = serde_json::json!({
            "vector": {
                "type": "dense_vector",
                "dims": vector_size,
                "index": true,
                "similarity": similarity,
                "index_options": {
                    "type": "hnsw",
                    "m": self.config.m,
                    "ef_construction": self.config.ef_construction,
                }
            }
        });

        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                let props = properties.as_object_mut().unwrap();
                for (field_name, field_type) in schema_obj {
                    let ft = field_type.as_str().unwrap_or("");
                    let es_type = match ft {
                        "int" => "long",
                        "geo" => "geo_point",
                        other => other,
                    };
                    props.insert(
                        field_name.clone(),
                        serde_json::json!({
                            "type": es_type,
                            "index": true,
                        }),
                    );
                }
            }
        }

        let body = serde_json::json!({
            "settings": {
                "index": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "refresh_interval": "10s",
                }
            },
            "mappings": {
                "_source": { "excludes": ["vector"] },
                "properties": properties,
            }
        });

        let resp = self
            .rt
            .block_on(
                self.client
                    .indices()
                    .create(IndicesCreateParts::Index(&self.index_name))
                    .body(body)
                    .send(),
            )
            .map_err(|e| format!("Failed to create index: {}", e))?;

        if !resp.status_code().is_success() {
            let body = self.rt.block_on(resp.text()).unwrap_or_default();
            return Err(format!("Failed to create index: {}", body));
        }

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
        let base_url = self.base_url.clone();
        let timeout = self.timeout;
        let index_name = self.index_name.clone();

        std::thread::scope(|s| {
            for _ in 0..self.config.parallel {
                let base_url = base_url.clone();
                let index_name = index_name.clone();
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let rt = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(rt) => rt,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e.to_string());
                            return;
                        }
                    };

                    let client = match create_es_client(&base_url, timeout) {
                        Ok(c) => c,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e);
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
                        if let Err(e) = upload_bulk_batch(
                            &rt,
                            &client,
                            &index_name,
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

    fn force_merge(&self) -> Result<(), String> {
        println!("Forcing merge into 1 segment...");

        let max_retries = 30;
        for attempt in 0..=max_retries {
            let result = self.rt.block_on(
                self.client
                    .indices()
                    .forcemerge(IndicesForcemergeParts::Index(&[&self.index_name]))
                    .max_num_segments(1)
                    .send(),
            );

            match result {
                Ok(resp) if resp.status_code().is_success() => {
                    self.wait_for_cluster_health()?;
                    return Ok(());
                }
                Ok(resp) => {
                    if attempt < max_retries {
                        println!(
                            "Force merge retry {}/{}: status {}",
                            attempt,
                            max_retries,
                            resp.status_code()
                        );
                        continue;
                    }
                    return Err(format!(
                        "Force merge failed after {} retries: {}",
                        max_retries,
                        resp.status_code()
                    ));
                }
                Err(e) => {
                    if attempt < max_retries {
                        println!("Force merge retry {}/{}: {}", attempt, max_retries, e);
                        continue;
                    }
                    return Err(format!(
                        "Force merge failed after {} retries: {}",
                        max_retries, e
                    ));
                }
            }
        }
        Ok(())
    }

    fn wait_for_cluster_health(&self) -> Result<(), String> {
        println!("Waiting for ES yellow status...");

        for _ in 0..100 {
            let result = self.rt.block_on(
                self.client
                    .cluster()
                    .health(elasticsearch::cluster::ClusterHealthParts::None)
                    .wait_for_status(WaitForStatus::Yellow)
                    .timeout("10s")
                    .send(),
            );

            match result {
                Ok(resp) if resp.status_code().is_success() => return Ok(()),
                _ => std::thread::sleep(std::time::Duration::from_millis(100)),
            }
        }
        Err("Elasticsearch cluster did not reach yellow status in time".to_string())
    }
}

/// Build the base URL with authentication from env vars.
fn build_base_url(host: &str, port: u16) -> String {
    let api_key = std::env::var("ELASTIC_API_KEY").ok();
    let user = std::env::var("ELASTIC_USER").unwrap_or_else(|_| "elastic".to_string());
    let password = std::env::var("ELASTIC_PASSWORD").unwrap_or_else(|_| "passwd".to_string());

    let scheme_host = if host.starts_with("http") {
        host.to_string()
    } else {
        format!("http://{}", host)
    };

    if api_key.is_some() {
        format!("{}:{}", scheme_host, port)
    } else if let Some(rest) = scheme_host.strip_prefix("http://") {
        format!("http://{}:{}@{}:{}", user, password, rest, port)
    } else if let Some(rest) = scheme_host.strip_prefix("https://") {
        format!("https://{}:{}@{}:{}", user, password, rest, port)
    } else {
        format!("http://{}:{}@{}:{}", user, password, scheme_host, port)
    }
}

/// Create an Elasticsearch client from a base URL.
fn create_es_client(base_url: &str, timeout: u64) -> Result<Elasticsearch, String> {
    let url = elasticsearch::http::Url::parse(base_url)
        .map_err(|e| format!("Invalid base URL '{}': {}", base_url, e))?;
    let pool = SingleNodeConnectionPool::new(url);
    let transport = TransportBuilder::new(pool)
        .timeout(std::time::Duration::from_secs(timeout))
        .disable_proxy()
        .cert_validation(elasticsearch::cert::CertificateValidation::None)
        .build()
        .map_err(|e| format!("Failed to build transport: {}", e))?;
    Ok(Elasticsearch::new(transport))
}

/// Convert integer ID to UUID hex string (matches Python uuid.UUID(int=idx).hex)
fn id_to_uuid_hex(id: i64) -> String {
    Uuid::from_u128(id as u128).as_simple().to_string()
}

/// Convert UUID hex string back to integer ID
fn uuid_hex_to_int(hex: &str) -> Result<i64, String> {
    let uuid = Uuid::parse_str(hex).map_err(|e| format!("Invalid UUID hex '{}': {}", hex, e))?;
    Ok(uuid.as_u128() as i64)
}

// ── Elasticsearch condition parser ─────────────────────────────────────

fn parse_es_conditions(conditions: &serde_json::Value) -> Option<serde_json::Value> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    let and_filters = obj
        .get("and")
        .and_then(|v| v.as_array())
        .map(|entries| build_subfilters(entries));
    let or_filters = obj
        .get("or")
        .and_then(|v| v.as_array())
        .map(|entries| build_subfilters(entries));

    if and_filters.is_none() && or_filters.is_none() {
        return None;
    }

    Some(serde_json::json!({
        "bool": {
            "must": and_filters.unwrap_or_default(),
            "should": or_filters.unwrap_or_default(),
        }
    }))
}

fn build_subfilters(entries: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut filters = Vec::new();
    for entry in entries {
        if let Some(entry_obj) = entry.as_object() {
            for (field_name, field_filters) in entry_obj {
                if let Some(filter_obj) = field_filters.as_object() {
                    for (condition_type, criteria) in filter_obj {
                        if let Some(filter) = build_filter(field_name, condition_type, criteria) {
                            filters.push(filter);
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
) -> Option<serde_json::Value> {
    match condition_type {
        "match" => {
            // match_any: field value in a list (keywords or integers). Emit a
            // `terms` query — the exact/case-sensitive OR-of-values semantics of
            // qdrant's Condition::matches(field, Vec). An empty IN-set matches
            // NOTHING, so we emit `terms: []` (a valid match-nothing query)
            // rather than dropping the clause: dropping the sole clause would
            // leave `bool.must:[]`, which ES treats as match-ALL — silently
            // returning unfiltered results, the inverse of the intended filter.
            if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                return Some(serde_json::json!({"terms": {field_name: any}}));
            }
            let value = criteria.get("value")?;
            Some(serde_json::json!({"match": {field_name: value}}))
        }
        "range" => {
            let criteria_obj = criteria.as_object()?;
            let mut range = serde_json::Map::new();
            for key in &["lt", "gt", "lte", "gte"] {
                if let Some(val) = criteria_obj.get(*key) {
                    if !val.is_null() {
                        range.insert(key.to_string(), val.clone());
                    }
                }
            }
            Some(serde_json::json!({"range": {field_name: range}}))
        }
        "geo" => {
            let lat = criteria.get("lat")?;
            let lon = criteria.get("lon")?;
            let radius = criteria
                .get("radius")
                .and_then(|r| r.as_f64())
                .unwrap_or(1000.0);
            Some(serde_json::json!({
                "geo_distance": {
                    "distance": format!("{}m", radius),
                    field_name: {"lat": lat, "lon": lon},
                }
            }))
        }
        _ => None,
    }
}

/// Upload a batch using the official Elasticsearch bulk API.
fn upload_bulk_batch(
    rt: &tokio::runtime::Runtime,
    client: &Elasticsearch,
    index_name: &str,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
) -> Result<(), String> {
    use vector_db_benchmark::readers::metadata::MetadataValue;

    let mut body: Vec<JsonBody<serde_json::Value>> = Vec::with_capacity(ids.len() * 2);

    for i in 0..ids.len() {
        let uuid_hex = id_to_uuid_hex(ids[i]);

        // Action line
        body.push(JsonBody::new(
            serde_json::json!({"index": {"_id": uuid_hex}}),
        ));

        // Document line
        let mut doc = serde_json::Map::new();
        let vec_json: Vec<serde_json::Value> = vectors[i]
            .iter()
            .map(|&f| serde_json::Value::from(f))
            .collect();
        doc.insert("vector".to_string(), serde_json::Value::Array(vec_json));

        if let Some(meta) = &metadata[i] {
            for (k, v) in &meta.fields {
                let val = match v {
                    MetadataValue::String(s) => serde_json::Value::String(s.clone()),
                    MetadataValue::Labels(labels) => serde_json::Value::Array(
                        labels
                            .iter()
                            .map(|l| serde_json::Value::String(l.clone()))
                            .collect(),
                    ),
                    MetadataValue::Geo { lon, lat } => {
                        serde_json::json!({ "lon": lon, "lat": lat })
                    }
                };
                doc.insert(k.clone(), val);
            }
        }

        body.push(JsonBody::new(serde_json::Value::Object(doc)));
    }

    let resp = rt
        .block_on(client.bulk(BulkParts::Index(index_name)).body(body).send())
        .map_err(|e| format!("Bulk upload failed: {}", e))?;

    if !resp.status_code().is_success() {
        let text = rt.block_on(resp.text()).unwrap_or_default();
        return Err(format!("Bulk upload error: {}", text));
    }

    let resp_body: serde_json::Value = rt
        .block_on(resp.json())
        .map_err(|e| format!("Failed to parse bulk response: {}", e))?;

    if resp_body
        .get("errors")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        let items = resp_body.get("items").and_then(|v| v.as_array());
        let error_count = items
            .map(|arr| {
                arr.iter()
                    .filter(|item| item.get("index").and_then(|idx| idx.get("error")).is_some())
                    .count()
            })
            .unwrap_or(0);
        return Err(format!(
            "Bulk upload had {} errors out of {} documents",
            error_count,
            ids.len()
        ));
    }

    Ok(())
}

/// Execute a KNN search using the official client.
fn knn_search(
    rt: &tokio::runtime::Runtime,
    client: &Elasticsearch,
    index_name: &str,
    query_vector: &[f32],
    top: usize,
    num_candidates: i64,
    filter: Option<&serde_json::Value>,
) -> Result<Vec<(i64, f64)>, String> {
    let mut knn = serde_json::json!({
        "field": "vector",
        "query_vector": query_vector,
        "k": top,
        "num_candidates": num_candidates,
    });
    if let Some(f) = filter {
        knn.as_object_mut()
            .unwrap()
            .insert("filter".to_string(), f.clone());
    }

    // Response trimming: the benchmark only needs each hit's id (always returned
    // as `_id` metadata) and score, so don't ship the document `_source`. This
    // trims the response for a fairer QPS/latency measurement — matching the
    // OpenSearch engine's trimming.
    let body = serde_json::json!({
        "knn": knn,
        "size": top,
        "_source": false,
    });

    let resp = rt
        .block_on(
            client
                .search(SearchParts::Index(&[index_name]))
                .body(body)
                .send(),
        )
        .map_err(|e| format!("KNN search failed: {}", e))?;

    if !resp.status_code().is_success() {
        let text = rt.block_on(resp.text()).unwrap_or_default();
        return Err(format!("KNN search error: {}", text));
    }

    let resp_body: serde_json::Value = rt
        .block_on(resp.json())
        .map_err(|e| format!("Failed to parse search response: {}", e))?;

    let hits = resp_body
        .get("hits")
        .and_then(|h| h.get("hits"))
        .and_then(|h| h.as_array())
        .ok_or_else(|| "Missing hits.hits in search response".to_string())?;

    let mut results = Vec::with_capacity(hits.len());
    for hit in hits {
        let id_hex = hit
            .get("_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "Missing _id in hit".to_string())?;
        let score = hit.get("_score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let id = uuid_hex_to_int(id_hex)?;
        results.push((id, score));
    }

    Ok(results)
}

// ── Engine trait implementation ──────────────────────────────────────────

impl Engine for ElasticsearchEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        println!(
            "Elasticsearch: index_options {{ m: {}, ef_construction: {} }}",
            self.config.m, self.config.ef_construction
        );

        println!("Ensuring index does not exist...");
        self.delete_index()?;

        println!("Creating index '{}'...", self.index_name);
        self.create_index(dataset)?;
        println!("Index '{}' created successfully.", self.index_name);

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

        self.upload_parallel(&ids, &vectors, &metadata)?;

        let upload_time = upload_start.elapsed().as_secs_f64();

        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        // Force merge post-upload. Include the merge/index-settle time in
        // total_time for cross-engine comparability (mirrors mongodb; matches
        // v0's post_upload() timing).
        let index_start = Instant::now();
        self.force_merge()?;
        let index_time = index_start.elapsed().as_secs_f64();

        let total_time = read_time + upload_time + index_time;
        println!(
            "Index time: {:.3}s, Total time (read+upload+index): {:.3}s",
            index_time, total_time
        );

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
        let parallel = params.parallel.unwrap_or(1) as usize;
        let num_candidates = params.num_candidates.unwrap_or(100);

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<serde_json::Value>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_es_conditions))
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
        let base_url = self.base_url.clone();
        let timeout = self.timeout;
        let index_name = self.index_name.clone();

        std::thread::scope(|s| {
            for _ in 0..parallel {
                let base_url = base_url.clone();
                let index_name = index_name.clone();
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
                    let rt = match tokio::runtime::Builder::new_current_thread()
                        .enable_all()
                        .build()
                    {
                        Ok(rt) => rt,
                        Err(_) => return,
                    };
                    let client = match create_es_client(&base_url, timeout) {
                        Ok(c) => c,
                        Err(_) => return,
                    };

                    loop {
                        let idx = query_idx.fetch_add(1, Ordering::SeqCst);
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
                        let results = knn_search(
                            &rt,
                            &client,
                            &index_name,
                            &queries[idx],
                            top,
                            num_candidates,
                            parsed_filters[idx].as_ref(),
                        );
                        let query_time = query_start.elapsed().as_secs_f64();

                        match results {
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

        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));
        crate::engine::compute_search_stats(
            &times, &precs, &recs, &mrr_vals, &ndcg_vals, total_time, top, parallel, num_to_run,
        )
    }

    fn delete(&mut self) -> Result<(), String> {
        self.delete_index()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_id_to_uuid_hex_zero() {
        assert_eq!(id_to_uuid_hex(0), "00000000000000000000000000000000");
    }

    #[test]
    fn test_match_any_string_list_emits_terms() {
        let c = build_filter(
            "color",
            "match",
            &serde_json::json!({"any": ["red", "blue"]}),
        )
        .unwrap();
        assert_eq!(c, serde_json::json!({"terms": {"color": ["red", "blue"]}}));
    }

    #[test]
    fn test_match_any_int_list_emits_terms() {
        let c = build_filter("size", "match", &serde_json::json!({"any": [1, 2, 3]})).unwrap();
        assert_eq!(c, serde_json::json!({"terms": {"size": [1, 2, 3]}}));
    }

    #[test]
    fn test_match_any_empty_list_matches_nothing() {
        // Empty IN-set must match NOTHING (never invert to match-all): `terms: []`.
        let c = build_filter("color", "match", &serde_json::json!({"any": []})).unwrap();
        assert_eq!(c, serde_json::json!({"terms": {"color": []}}));
    }

    #[test]
    fn test_match_exact_value_still_works() {
        let c = build_filter("color", "match", &serde_json::json!({"value": "red"})).unwrap();
        assert_eq!(c, serde_json::json!({"match": {"color": "red"}}));
    }

    #[test]
    fn test_id_to_uuid_hex_one() {
        assert_eq!(id_to_uuid_hex(1), "00000000000000000000000000000001");
    }

    #[test]
    fn test_id_to_uuid_hex_large() {
        assert_eq!(id_to_uuid_hex(255), "000000000000000000000000000000ff");
    }

    #[test]
    fn test_id_to_uuid_hex_typical_id() {
        assert_eq!(id_to_uuid_hex(12345), "00000000000000000000000000003039");
    }

    #[test]
    fn test_build_base_url_includes_credentials() {
        std::env::remove_var("ELASTIC_API_KEY");
        let url = build_base_url("myhost", 9200);
        assert!(
            url.contains("@myhost:9200"),
            "URL should contain auth@host:port"
        );
        assert!(url.starts_with("http://"), "URL should start with http://");
    }

    #[test]
    fn test_build_base_url_with_http_scheme() {
        std::env::remove_var("ELASTIC_API_KEY");
        let url = build_base_url("http://myhost", 9200);
        assert!(!url.contains("http://http://"));
        assert!(url.contains("@myhost:9200"));
    }

    #[test]
    fn test_build_base_url_with_https_scheme() {
        std::env::remove_var("ELASTIC_API_KEY");
        let url = build_base_url("https://myhost", 9200);
        assert!(url.starts_with("https://"));
        assert!(url.contains("@myhost:9200"));
    }

    #[test]
    fn test_config_parsing_defaults() {
        let config = EngineConfig {
            name: "test-es".to_string(),
            engine: Some("elasticsearch".to_string()),
            algorithm: None,
            connection_params: None,
            collection_params: None,
            search_params: None,
            upload_params: None,
            skip_vector_index: false,
        };
        let engine = ElasticsearchEngine::new(&config, "localhost").unwrap();
        assert_eq!(engine.name, "test-es");
        assert_eq!(engine.config.m, 16);
        assert_eq!(engine.config.ef_construction, 100);
        assert_eq!(engine.config.batch_size, 500);
        assert_eq!(engine.config.parallel, 16);
    }

    #[test]
    fn test_config_parsing_custom_values() {
        let config = EngineConfig {
            name: "test-es-custom".to_string(),
            engine: Some("elasticsearch".to_string()),
            algorithm: None,
            connection_params: None,
            collection_params: Some(crate::config::CollectionParams {
                hnsw_config: None,
                index_options: Some(crate::config::IndexOptions {
                    m: Some(32),
                    ef_construction: Some(256),
                }),
                extra: None,
            }),
            search_params: None,
            upload_params: Some(serde_json::json!({
                "parallel": 8,
                "batch_size": 1000
            })),
            skip_vector_index: false,
        };
        let engine = ElasticsearchEngine::new(&config, "localhost").unwrap();
        assert_eq!(engine.config.m, 32);
        assert_eq!(engine.config.ef_construction, 256);
        assert_eq!(engine.config.batch_size, 1000);
        assert_eq!(engine.config.parallel, 8);
    }
}
