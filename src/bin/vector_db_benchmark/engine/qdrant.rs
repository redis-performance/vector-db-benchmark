//! Qdrant engine implementation.
//!
//! Uses Qdrant's REST API via reqwest::blocking.
//! Supports HNSW index with configurable M/ef_construct, payload indexing,
//! and filter conditions.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use vector_db_benchmark::readers::metadata::MetadataItem;

const DEFAULT_COLLECTION: &str = "benchmark";

pub struct QdrantEngine {
    name: String,
    collection_name: String,
    timeout: u64,
    batch_size: usize,
    parallel: usize,
    base_url: String,
    api_key: Option<String>,
    search_params: Vec<SearchParams>,
    /// Raw collection_params JSON to pass through to Qdrant
    collection_params_extra: serde_json::Value,
}

impl QdrantEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("QDRANT_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6333);

        let collection_name = std::env::var("QDRANT_COLLECTION_NAME")
            .unwrap_or_else(|_| DEFAULT_COLLECTION.to_string());

        let api_key = std::env::var("QDRANT_API_KEY").ok();

        let timeout = engine_config
            .connection_params
            .as_ref()
            .and_then(|p| p.get("timeout"))
            .and_then(|v| v.as_u64())
            .unwrap_or(300);

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
            .unwrap_or(1024) as usize;

        let base_url = if let Some(url) = std::env::var("QDRANT_URL").ok() {
            url
        } else if host.starts_with("http") {
            format!("{}:{}", host, port)
        } else {
            format!("http://{}:{}", host, port)
        };

        // Extract collection params extra (optimizers_config, hnsw_config, quantization_config, etc.)
        // These are passed through to Qdrant's create collection API
        let collection_params_extra = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.extra.as_ref())
            .map(|e| serde_json::to_value(e).unwrap_or_default())
            .unwrap_or(serde_json::json!({}));

        Ok(Self {
            name: engine_config.name.clone(),
            collection_name,
            timeout,
            batch_size,
            parallel,
            base_url,
            api_key,
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            collection_params_extra,
        })
    }

    fn create_client(&self) -> Result<reqwest::blocking::Client, String> {
        reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout))
            .build()
            .map_err(|e| format!("Failed to create HTTP client: {}", e))
    }

    fn add_auth(
        &self,
        req: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        if let Some(key) = &self.api_key {
            req.header("api-key", key)
        } else {
            req
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

    fn delete_collection(&self, client: &reqwest::blocking::Client) -> Result<(), String> {
        let url = format!("{}/collections/{}", self.base_url, self.collection_name);
        let req = client.delete(&url);
        let resp = self.add_auth(req).send().map_err(|e| e.to_string())?;
        // Ignore 404 (collection doesn't exist)
        if resp.status().is_success() || resp.status().as_u16() == 404 {
            Ok(())
        } else {
            Err(format!(
                "Failed to delete collection: {} {}",
                resp.status(),
                resp.text().unwrap_or_default()
            ))
        }
    }

    fn create_collection(
        &self,
        client: &reqwest::blocking::Client,
        dataset: &Dataset,
    ) -> Result<(), String> {
        let distance = dataset.distance();
        let vector_size = dataset.vector_size();

        let qdrant_distance = match distance.to_lowercase().as_str() {
            "l2" | "euclidean" => "Euclid",
            "cosine" | "angular" => "Cosine",
            "dot" | "ip" => "Dot",
            other => {
                return Err(format!("Unsupported distance metric for Qdrant: {}", other))
            }
        };

        // Build create collection body
        let mut body = serde_json::json!({
            "vectors": {
                "size": vector_size,
                "distance": qdrant_distance,
            }
        });

        // Merge extra collection params (hnsw_config, optimizers_config, etc.)
        if let Some(extra_obj) = self.collection_params_extra.as_object() {
            let body_obj = body.as_object_mut().unwrap();
            for (k, v) in extra_obj {
                // Skip "timeout" as it's a connection param, not collection config
                if k == "timeout" {
                    continue;
                }
                body_obj.insert(k.clone(), v.clone());
            }
        }

        let url = format!("{}/collections/{}", self.base_url, self.collection_name);
        let req = client
            .put(&url)
            .header("Content-Type", "application/json")
            .json(&body);
        let resp = self.add_auth(req).send().map_err(|e| e.to_string())?;

        if !resp.status().is_success() {
            return Err(format!(
                "Failed to create collection: {} {}",
                resp.status(),
                resp.text().unwrap_or_default()
            ));
        }

        // Disable optimization during indexing
        let update_body = serde_json::json!({
            "optimizers_config": {
                "max_optimization_threads": 0,
            }
        });
        let url = format!("{}/collections/{}", self.base_url, self.collection_name);
        let req = client
            .patch(&url)
            .header("Content-Type", "application/json")
            .json(&update_body);
        let _ = self.add_auth(req).send();

        // Create payload indexes for schema fields
        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                for (field_name, field_type) in schema_obj {
                    let ft = field_type.as_str().unwrap_or("");
                    let qdrant_type = match ft {
                        "int" => "integer",
                        "keyword" => "keyword",
                        "text" => "text",
                        "float" => "float",
                        "geo" => "geo",
                        _ => continue,
                    };
                    let index_body = serde_json::json!({
                        "field_name": field_name,
                        "field_schema": qdrant_type,
                    });
                    let url = format!(
                        "{}/collections/{}/index",
                        self.base_url, self.collection_name
                    );
                    let req = client
                        .put(&url)
                        .header("Content-Type", "application/json")
                        .json(&index_body);
                    let _ = self.add_auth(req).send();
                }
            }
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
            .step_by(self.batch_size)
            .map(|start| (start, (start + self.batch_size).min(ids.len())))
            .collect();

        let total_batches = batches.len();
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        std::thread::scope(|s| {
            for _ in 0..self.parallel {
                let base_url = self.base_url.clone();
                let collection_name = self.collection_name.clone();
                let api_key = self.api_key.clone();
                let timeout = self.timeout;
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let client = match reqwest::blocking::Client::builder()
                        .timeout(std::time::Duration::from_secs(timeout))
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
                        if let Err(e) = upsert_points(
                            &client,
                            &base_url,
                            &collection_name,
                            api_key.as_deref(),
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

    fn wait_collection_green(&self, client: &reqwest::blocking::Client) -> Result<(), String> {
        println!("Waiting for collection to be GREEN...");
        let url = format!("{}/collections/{}", self.base_url, self.collection_name);

        // Re-enable optimization
        let update_body = serde_json::json!({
            "optimizers_config": {
                "max_optimization_threads": null,
            }
        });
        let patch_url = format!("{}/collections/{}", self.base_url, self.collection_name);
        let req = client
            .patch(&patch_url)
            .header("Content-Type", "application/json")
            .json(&update_body);
        let _ = self.add_auth(req).send();

        for _ in 0..600 {
            std::thread::sleep(std::time::Duration::from_secs(5));

            let req = client.get(&url);
            if let Ok(resp) = self.add_auth(req).send() {
                if let Ok(body) = resp.json::<serde_json::Value>() {
                    if let Some(status) = body
                        .get("result")
                        .and_then(|r| r.get("status"))
                        .and_then(|s| s.as_str())
                    {
                        if status == "green" {
                            // Double-check
                            std::thread::sleep(std::time::Duration::from_secs(5));
                            let req2 = client.get(&url);
                            if let Ok(resp2) = self.add_auth(req2).send() {
                                if let Ok(body2) = resp2.json::<serde_json::Value>() {
                                    if body2
                                        .get("result")
                                        .and_then(|r| r.get("status"))
                                        .and_then(|s| s.as_str())
                                        == Some("green")
                                    {
                                        println!("Collection is GREEN.");
                                        return Ok(());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Err("Timed out waiting for collection to reach GREEN status".to_string())
    }
}

/// Upsert a batch of points to Qdrant.
fn upsert_points(
    client: &reqwest::blocking::Client,
    base_url: &str,
    collection_name: &str,
    api_key: Option<&str>,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
) -> Result<(), String> {
    use vector_db_benchmark::readers::metadata::MetadataValue;

    let mut points = Vec::with_capacity(ids.len());
    for i in 0..ids.len() {
        let mut payload = serde_json::Map::new();
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
                        serde_json::json!({"lon": lon, "lat": lat})
                    }
                };
                payload.insert(k.clone(), val);
            }
        }

        points.push(serde_json::json!({
            "id": ids[i],
            "vector": vectors[i],
            "payload": payload,
        }));
    }

    let body = serde_json::json!({
        "points": points,
    });

    let url = format!(
        "{}/collections/{}/points?wait=false",
        base_url, collection_name
    );
    let mut req = client
        .put(&url)
        .header("Content-Type", "application/json")
        .json(&body);

    if let Some(key) = api_key {
        req = req.header("api-key", key);
    }

    let resp = req.send().map_err(|e| format!("Upsert failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Upsert error: {} {}",
            resp.status(),
            resp.text().unwrap_or_default()
        ));
    }

    Ok(())
}

/// Search Qdrant via REST API.
fn search_points(
    client: &reqwest::blocking::Client,
    base_url: &str,
    collection_name: &str,
    api_key: Option<&str>,
    query_vector: &[f32],
    top: usize,
    search_params: Option<&serde_json::Value>,
    filter: Option<&serde_json::Value>,
) -> Result<Vec<(i64, f64)>, String> {
    let mut body = serde_json::json!({
        "vector": query_vector,
        "limit": top,
        "with_payload": false,
    });

    if let Some(params) = search_params {
        body.as_object_mut()
            .unwrap()
            .insert("params".to_string(), params.clone());
    }

    if let Some(f) = filter {
        body.as_object_mut()
            .unwrap()
            .insert("filter".to_string(), f.clone());
    }

    let url = format!(
        "{}/collections/{}/points/search",
        base_url, collection_name
    );
    let mut req = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body);

    if let Some(key) = api_key {
        req = req.header("api-key", key);
    }

    let resp = req
        .send()
        .map_err(|e| format!("Search failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Search error: {} {}",
            resp.status(),
            resp.text().unwrap_or_default()
        ));
    }

    let resp_body: serde_json::Value = resp
        .json()
        .map_err(|e| format!("Failed to parse search response: {}", e))?;

    let results = resp_body
        .get("result")
        .and_then(|r| r.as_array())
        .ok_or_else(|| "Missing result array in search response".to_string())?;

    let mut hits = Vec::with_capacity(results.len());
    for hit in results {
        let id = hit.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
        let score = hit.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
        hits.push((id, score));
    }

    Ok(hits)
}

/// Parse conditions into Qdrant filter format.
fn parse_qdrant_conditions(conditions: &serde_json::Value) -> Option<serde_json::Value> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    let must = obj
        .get("and")
        .and_then(|v| v.as_array())
        .map(|entries| build_qdrant_subfilters(entries));
    let should = obj
        .get("or")
        .and_then(|v| v.as_array())
        .map(|entries| build_qdrant_subfilters(entries));

    if must.is_none() && should.is_none() {
        return None;
    }

    let mut filter = serde_json::Map::new();
    if let Some(m) = must {
        filter.insert("must".to_string(), serde_json::Value::Array(m));
    }
    if let Some(s) = should {
        filter.insert("should".to_string(), serde_json::Value::Array(s));
    }

    Some(serde_json::Value::Object(filter))
}

fn build_qdrant_subfilters(entries: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut filters = Vec::new();
    for entry in entries {
        if let Some(entry_obj) = entry.as_object() {
            for (field_name, field_filters) in entry_obj {
                if let Some(filter_obj) = field_filters.as_object() {
                    for (cond_type, criteria) in filter_obj {
                        if let Some(f) = build_qdrant_filter(field_name, cond_type, criteria) {
                            filters.push(f);
                        }
                    }
                }
            }
        }
    }
    filters
}

fn build_qdrant_filter(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
) -> Option<serde_json::Value> {
    match condition_type {
        "match" => {
            let value = criteria.get("value")?;
            Some(serde_json::json!({
                "key": field_name,
                "match": {"value": value},
            }))
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
            Some(serde_json::json!({
                "key": field_name,
                "range": range,
            }))
        }
        "geo" => {
            let lat = criteria.get("lat")?.as_f64()?;
            let lon = criteria.get("lon")?.as_f64()?;
            let radius = criteria
                .get("radius")
                .and_then(|r| r.as_f64())
                .unwrap_or(1000.0);
            Some(serde_json::json!({
                "key": field_name,
                "geo_radius": {
                    "center": {"lon": lon, "lat": lat},
                    "radius": radius,
                },
            }))
        }
        _ => None,
    }
}

impl Engine for QdrantEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        let client = self.create_client()?;

        println!("Deleting existing collection...");
        self.delete_collection(&client)?;

        println!("Creating collection '{}'...", self.collection_name);
        self.create_collection(&client, dataset)?;
        println!("Collection '{}' created.", self.collection_name);

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
            self.parallel, self.batch_size
        );
        let upload_start = Instant::now();
        self.upload_parallel(&ids, &vectors, &metadata)?;
        let upload_time = upload_start.elapsed().as_secs_f64();

        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        // Wait for indexing to complete
        let client = self.create_client()?;
        self.wait_collection_green(&client)?;

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

        // Build Qdrant search params from config
        // Qdrant uses search_params.hnsw_ef or search_params.search_params
        let qdrant_search_params: Option<serde_json::Value> = params
            .search_params
            .as_ref()
            .map(|sp| {
                let mut p = serde_json::Map::new();
                if let Some(ef) = sp.ef {
                    // Qdrant uses "hnsw_ef" in search params
                    p.insert("hnsw_ef".to_string(), serde_json::json!(ef));
                }
                // Also check for hnsw_ef in the inner search_params extra
                if let Some(extra) = &sp.extra {
                    if let Some(hnsw_ef) = extra.get("hnsw_ef") {
                        p.insert("hnsw_ef".to_string(), hnsw_ef.clone());
                    }
                    // Pass through quantization params etc.
                    if let Some(q) = extra.get("quantization") {
                        p.insert("quantization".to_string(), q.clone());
                    }
                }
                serde_json::Value::Object(p)
            });

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<serde_json::Value>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_qdrant_conditions))
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
                let collection_name = self.collection_name.clone();
                let api_key = self.api_key.clone();
                let timeout = self.timeout;
                let qdrant_search_params = &qdrant_search_params;
                let queries = &queries;
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
                let search_times = Arc::clone(&search_times);
                let precisions = Arc::clone(&precisions);
                let query_idx = Arc::clone(&query_idx);

                s.spawn(move || {
                    let client = match reqwest::blocking::Client::builder()
                        .timeout(std::time::Duration::from_secs(timeout))
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
                        let results = search_points(
                            &client,
                            &base_url,
                            &collection_name,
                            api_key.as_deref(),
                            &queries[idx],
                            top,
                            qdrant_search_params.as_ref(),
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
        self.delete_collection(&client)
    }

    fn get_memory_usage(&mut self) -> Option<serde_json::Value> {
        let client = self.create_client().ok()?;
        let url = format!("{}/collections/{}", self.base_url, self.collection_name);
        let req = client.get(&url);
        let resp = self.add_auth(req).send().ok()?;
        let body: serde_json::Value = resp.json().ok()?;
        Some(serde_json::json!({
            "collection_info": body.get("result"),
        }))
    }
}
