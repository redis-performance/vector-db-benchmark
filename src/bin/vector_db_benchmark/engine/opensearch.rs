//! OpenSearch engine implementation.
//!
//! Very similar to Elasticsearch but uses knn_vector type and different query format.
//! Uses reqwest::blocking for HTTP calls against the OpenSearch REST API.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use uuid::Uuid;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use vector_db_benchmark::readers::metadata::MetadataItem;

#[derive(Clone)]
struct OpenSearchConfig {
    m: i64,
    ef_construction: i64,
    batch_size: usize,
    parallel: usize,
}

pub struct OpenSearchEngine {
    name: String,
    index_name: String,
    timeout: u64,
    config: OpenSearchConfig,
    search_params: Vec<SearchParams>,
    base_url: String,
}

impl OpenSearchEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("OPENSEARCH_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(9200);

        let index_name =
            std::env::var("OPENSEARCH_INDEX").unwrap_or_else(|_| "bench".to_string());
        let timeout: u64 = std::env::var("OPENSEARCH_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(300);

        // Extract HNSW config from collection_params.method.parameters (OpenSearch format)
        // or fall back to index_options (ES format) or defaults
        let (m, ef_construction) = extract_hnsw_params(engine_config);

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

        Ok(Self {
            name: engine_config.name.clone(),
            index_name,
            timeout,
            config: OpenSearchConfig {
                m,
                ef_construction,
                batch_size,
                parallel,
            },
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            base_url,
        })
    }

    fn create_client(&self) -> Result<reqwest::blocking::Client, String> {
        reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout))
            .danger_accept_invalid_certs(true)
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))
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

    fn delete_index(&self, client: &reqwest::blocking::Client) -> Result<(), String> {
        let url = format!("{}/{}", self.base_url, self.index_name);
        let resp = client.delete(&url).send().map_err(|e| e.to_string())?;
        if resp.status().is_success() || resp.status().as_u16() == 404 {
            Ok(())
        } else {
            Err(format!(
                "Failed to delete index: {} {}",
                resp.status(),
                resp.text().unwrap_or_default()
            ))
        }
    }

    fn create_index(
        &self,
        client: &reqwest::blocking::Client,
        dataset: &Dataset,
    ) -> Result<(), String> {
        let distance = dataset.distance();
        let vector_size = dataset.vector_size();

        let dist_lower = distance.to_lowercase();
        if dist_lower == "dot" || dist_lower == "ip" {
            return Err("OpenSearch does not support DOT product distance".to_string());
        }
        if vector_size > 2048 {
            return Err(format!(
                "OpenSearch does not support vector_size > 2048 (got {})",
                vector_size
            ));
        }

        // Map distance metric (OpenSearch uses different names than ES)
        let space_type = match dist_lower.as_str() {
            "l2" | "euclidean" => "l2",
            "cosine" | "angular" => "cosinesimil",
            other => {
                return Err(format!(
                    "Unsupported distance metric for OpenSearch: {}",
                    other
                ))
            }
        };

        // Build properties with knn_vector type
        let mut properties = serde_json::json!({
            "vector": {
                "type": "knn_vector",
                "dimension": vector_size,
                "method": {
                    "name": "hnsw",
                    "engine": "lucene",
                    "space_type": space_type,
                    "parameters": {
                        "m": self.config.m,
                        "ef_construction": self.config.ef_construction,
                    }
                }
            }
        });

        // Add schema fields from dataset config
        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                let props = properties.as_object_mut().unwrap();
                for (field_name, field_type) in schema_obj {
                    let ft = field_type.as_str().unwrap_or("");
                    let os_type = match ft {
                        "int" => "long",
                        "geo" => "geo_point",
                        other => other,
                    };
                    props.insert(
                        field_name.clone(),
                        serde_json::json!({
                            "type": os_type,
                            "index": true,
                        }),
                    );
                }
            }
        }

        let body = serde_json::json!({
            "settings": {
                "index": {
                    "knn": true,
                }
            },
            "mappings": {
                "properties": properties,
            }
        });

        let url = format!("{}/{}", self.base_url, self.index_name);
        let resp = client
            .put(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| format!("Failed to create index: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!(
                "Failed to create index: {} {}",
                resp.status(),
                resp.text().unwrap_or_default()
            ));
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

        std::thread::scope(|s| {
            for _ in 0..self.config.parallel {
                let base_url = self.base_url.clone();
                let index_name = self.index_name.clone();
                let timeout = self.timeout;
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let client = match reqwest::blocking::Client::builder()
                        .timeout(std::time::Duration::from_secs(timeout))
                        .danger_accept_invalid_certs(true)
                        .build()
                    {
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
                        if let Err(e) = upload_bulk_batch(
                            &client,
                            &base_url,
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

    fn force_merge(&self, client: &reqwest::blocking::Client) -> Result<(), String> {
        println!("Forcing merge...");
        let url = format!(
            "{}/{}/_forcemerge?wait_for_completion=true",
            self.base_url, self.index_name
        );

        let resp = client
            .post(&url)
            .send()
            .map_err(|e| format!("Force merge failed: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!(
                "Force merge error: {} {}",
                resp.status(),
                resp.text().unwrap_or_default()
            ));
        }
        Ok(())
    }

    /// Apply search-time settings (e.g., knn.algo_param.ef_search)
    fn setup_search(
        &self,
        client: &reqwest::blocking::Client,
        params: &SearchParams,
    ) -> Result<(), String> {
        // Extract knn.algo_param.ef_search from the extra params
        let ef_search = params
            .extra
            .as_ref()
            .and_then(|e| e.get("knn.algo_param.ef_search"))
            .and_then(|v| v.as_i64());

        if let Some(ef) = ef_search {
            let body = serde_json::json!({
                "index": {
                    "knn.algo_param.ef_search": ef,
                }
            });

            let url = format!("{}/{}/_settings", self.base_url, self.index_name);
            let resp = client
                .put(&url)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
                .map_err(|e| format!("Failed to apply search settings: {}", e))?;

            if !resp.status().is_success() {
                eprintln!(
                    "Warning: failed to set ef_search={}: {} {}",
                    ef,
                    resp.status(),
                    resp.text().unwrap_or_default()
                );
            }
        }
        Ok(())
    }
}

/// Extract m and ef_construction from OpenSearch collection_params.
/// Supports: collection_params.method.parameters.{m, ef_construction}
/// Falls back to: collection_params.index_options.{m, ef_construction}
fn extract_hnsw_params(engine_config: &EngineConfig) -> (i64, i64) {
    if let Some(cp) = &engine_config.collection_params {
        // Try OpenSearch format: method.parameters
        if let Some(extra) = &cp.extra {
            if let Some(method) = extra.get("method") {
                if let Some(params) = method.get("parameters") {
                    let m = params.get("m").and_then(|v| v.as_i64()).unwrap_or(16);
                    let ef = params
                        .get("ef_construction")
                        .and_then(|v| v.as_i64())
                        .unwrap_or(100);
                    return (m, ef);
                }
            }
        }
        // Try ES format: index_options
        if let Some(io) = &cp.index_options {
            return (io.m.unwrap_or(16), io.ef_construction.unwrap_or(100));
        }
    }
    (16, 100)
}

fn build_base_url(host: &str, port: u16) -> String {
    let user =
        std::env::var("OPENSEARCH_USER").unwrap_or_else(|_| "admin".to_string());
    let password =
        std::env::var("OPENSEARCH_PASSWORD").unwrap_or_else(|_| "admin".to_string());

    let scheme_host = if host.starts_with("http") {
        host.to_string()
    } else {
        format!("https://{}", host)
    };

    if let Some(rest) = scheme_host.strip_prefix("http://") {
        format!("http://{}:{}@{}:{}", user, password, rest, port)
    } else if let Some(rest) = scheme_host.strip_prefix("https://") {
        format!("https://{}:{}@{}:{}", user, password, rest, port)
    } else {
        format!("https://{}:{}@{}:{}", user, password, scheme_host, port)
    }
}

fn id_to_uuid_hex(id: i64) -> String {
    Uuid::from_u128(id as u128).as_simple().to_string()
}

fn uuid_hex_to_int(hex: &str) -> Result<i64, String> {
    let uuid = Uuid::parse_str(hex).map_err(|e| format!("Invalid UUID hex '{}': {}", hex, e))?;
    Ok(uuid.as_u128() as i64)
}

/// Parse conditions into OpenSearch bool query (same DSL as Elasticsearch).
fn parse_os_conditions(conditions: &serde_json::Value) -> Option<serde_json::Value> {
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

fn upload_bulk_batch(
    client: &reqwest::blocking::Client,
    base_url: &str,
    index_name: &str,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
) -> Result<(), String> {
    use vector_db_benchmark::readers::metadata::MetadataValue;

    let mut body = String::new();

    for i in 0..ids.len() {
        let uuid_hex = id_to_uuid_hex(ids[i]);
        body.push_str(&format!("{{\"index\":{{\"_id\":\"{}\"}}}}\n", uuid_hex));

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

        body.push_str(&serde_json::to_string(&doc).map_err(|e| e.to_string())?);
        body.push('\n');
    }

    let url = format!("{}/{}/_bulk", base_url, index_name);
    let resp = client
        .post(&url)
        .header("Content-Type", "application/x-ndjson")
        .body(body)
        .send()
        .map_err(|e| format!("Bulk upload failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Bulk upload error: {} {}",
            resp.status(),
            resp.text().unwrap_or_default()
        ));
    }

    let resp_body: serde_json::Value = resp
        .json()
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

/// OpenSearch KNN search (different format from Elasticsearch).
/// Uses {"knn": {"vector": {"vector": [...], "k": top}}} format.
fn knn_search(
    client: &reqwest::blocking::Client,
    base_url: &str,
    index_name: &str,
    query_vector: &[f32],
    top: usize,
    filter: Option<&serde_json::Value>,
) -> Result<Vec<(i64, f64)>, String> {
    let mut query = serde_json::json!({
        "knn": {
            "vector": {
                "vector": query_vector,
                "k": top,
            }
        }
    });

    if let Some(f) = filter {
        query = serde_json::json!({
            "bool": {
                "must": [query],
                "filter": f,
            }
        });
    }

    let body = serde_json::json!({
        "query": query,
        "size": top,
    });

    let url = format!("{}/{}/_search", base_url, index_name);
    let resp = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .map_err(|e| format!("KNN search failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "KNN search error: {} {}",
            resp.status(),
            resp.text().unwrap_or_default()
        ));
    }

    let resp_body: serde_json::Value = resp
        .json()
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

impl Engine for OpenSearchEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        let client = self.create_client()?;

        println!(
            "OpenSearch: HNSW {{ m: {}, ef_construction: {} }}",
            self.config.m, self.config.ef_construction
        );

        println!("Ensuring index does not exist...");
        self.delete_index(&client)?;

        println!("Creating index '{}'...", self.index_name);
        self.create_index(&client, dataset)?;
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
            "Read {} vectors ({}d) in {:.3}s",
            vectors.len(),
            vectors.first().map(|v| v.len()).unwrap_or(0),
            read_time,
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

        let client = self.create_client()?;
        self.force_merge(&client)?;

        let total_time = read_time + upload_time;

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

        // Apply search-time settings (ef_search)
        let client = self.create_client()?;
        self.setup_search(&client, params)?;

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<serde_json::Value>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_os_conditions))
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
        let query_idx = Arc::new(AtomicUsize::new(0));

        let start_time = Instant::now();

        std::thread::scope(|s| {
            for _ in 0..parallel {
                let base_url = self.base_url.clone();
                let index_name = self.index_name.clone();
                let timeout = self.timeout;
                let queries = &queries;
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
                let search_times = Arc::clone(&search_times);
                let precisions = Arc::clone(&precisions);
                let query_idx = Arc::clone(&query_idx);

                s.spawn(move || {
                    let client = match reqwest::blocking::Client::builder()
                        .timeout(std::time::Duration::from_secs(timeout))
                        .danger_accept_invalid_certs(true)
                        .build()
                    {
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
                            if n > 0 { n } else { 10 }
                        });

                        let query_start = Instant::now();
                        let results = knn_search(
                            &client,
                            &base_url,
                            &index_name,
                            &queries[idx],
                            top,
                            parsed_filters[idx].as_ref(),
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
        let client = self.create_client()?;
        self.delete_index(&client)
    }
}
