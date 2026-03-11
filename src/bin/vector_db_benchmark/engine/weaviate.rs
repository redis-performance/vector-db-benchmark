//! Weaviate engine implementation.
//!
//! Uses Weaviate's REST API (v1) via reqwest::blocking.
//! Supports HNSW vector index with configurable efConstruction/maxConnections,
//! schema-based properties, and near_vector search.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use uuid::Uuid;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use vector_db_benchmark::readers::metadata::MetadataItem;

const DEFAULT_CLASS_NAME: &str = "Benchmark";

pub struct WeaviateEngine {
    name: String,
    class_name: String,
    timeout: u64,
    batch_size: usize,
    parallel: usize,
    base_url: String,
    api_key: Option<String>,
    search_params: Vec<SearchParams>,
    /// vectorIndexConfig from collection_params
    vector_index_config: serde_json::Value,
}

impl WeaviateEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("WEAVIATE_HTTP_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8080);

        let class_name =
            std::env::var("WEAVIATE_CLASS_NAME").unwrap_or_else(|_| DEFAULT_CLASS_NAME.to_string());

        let api_key = std::env::var("WEAVIATE_API_KEY").ok();

        let timeout = engine_config
            .connection_params
            .as_ref()
            .and_then(|p| p.get("timeout_config"))
            .and_then(|v| v.as_u64())
            .unwrap_or(90);

        let parallel = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("parallel"))
            .and_then(|v| v.as_i64())
            .unwrap_or(8) as usize;

        let batch_size = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("batch_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(1024) as usize;

        let base_url = if host.starts_with("http") {
            format!("{}:{}", host, port)
        } else {
            format!("http://{}:{}", host, port)
        };

        // Extract vectorIndexConfig from collection_params.extra
        let vector_index_config = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.extra.as_ref())
            .and_then(|e| e.get("vectorIndexConfig"))
            .cloned()
            .unwrap_or(serde_json::json!({}));

        Ok(Self {
            name: engine_config.name.clone(),
            class_name,
            timeout,
            batch_size,
            parallel,
            base_url,
            api_key,
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            vector_index_config,
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
            req.header("Authorization", format!("Bearer {}", key))
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

    fn delete_class(&self, client: &reqwest::blocking::Client) -> Result<(), String> {
        let url = format!("{}/v1/schema/{}", self.base_url, self.class_name);
        let req = client.delete(&url);
        let resp = self.add_auth(req).send().map_err(|e| e.to_string())?;
        if resp.status().is_success() || resp.status().as_u16() == 404 {
            Ok(())
        } else {
            Err(format!(
                "Failed to delete class: {} {}",
                resp.status(),
                resp.text().unwrap_or_default()
            ))
        }
    }

    fn create_class(
        &self,
        client: &reqwest::blocking::Client,
        dataset: &Dataset,
    ) -> Result<(), String> {
        let distance = dataset.distance();

        let weaviate_distance = match distance.to_lowercase().as_str() {
            "l2" | "euclidean" => "l2-squared",
            "cosine" | "angular" => "cosine",
            "dot" | "ip" => "dot",
            other => {
                return Err(format!(
                    "Unsupported distance metric for Weaviate: {}",
                    other
                ))
            }
        };

        // Build properties from schema
        let mut properties = Vec::new();
        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                for (field_name, field_type) in schema_obj {
                    let ft = field_type.as_str().unwrap_or("");
                    let wv_type = match ft {
                        "int" => "int",
                        "keyword" => "string",
                        "text" => "string",
                        "float" => "number",
                        "geo" => "geoCoordinates",
                        _ => continue,
                    };
                    properties.push(serde_json::json!({
                        "name": field_name,
                        "dataType": [wv_type],
                        "indexInverted": true,
                    }));
                }
            }
        }

        // Merge vectorIndexConfig with distance and cache settings
        let mut vic = serde_json::json!({
            "vectorCacheMaxObjects": 1_000_000_000i64,
            "distance": weaviate_distance,
        });
        if let Some(vic_obj) = self.vector_index_config.as_object() {
            let merged = vic.as_object_mut().unwrap();
            for (k, v) in vic_obj {
                merged.insert(k.clone(), v.clone());
            }
        }

        let body = serde_json::json!({
            "class": self.class_name,
            "vectorizer": "none",
            "properties": properties,
            "vectorIndexConfig": vic,
        });

        let url = format!("{}/v1/schema", self.base_url);
        let req = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body);
        let resp = self
            .add_auth(req)
            .send()
            .map_err(|e| format!("Failed to create class: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!(
                "Failed to create class: {} {}",
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
            .step_by(self.batch_size)
            .map(|start| (start, (start + self.batch_size).min(ids.len())))
            .collect();

        let total_batches = batches.len();
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        std::thread::scope(|s| {
            for _ in 0..self.parallel {
                let base_url = self.base_url.clone();
                let class_name = self.class_name.clone();
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
                        if let Err(e) = upload_batch_objects(
                            &client,
                            &base_url,
                            &class_name,
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

    /// Update vectorIndexConfig ef for search-time tuning.
    fn setup_search(
        &self,
        client: &reqwest::blocking::Client,
        params: &SearchParams,
    ) -> Result<(), String> {
        // Extract ef from search_params extra (vectorIndexConfig.ef)
        let ef = params
            .extra
            .as_ref()
            .and_then(|e| e.get("vectorIndexConfig"))
            .and_then(|v| v.get("ef"))
            .and_then(|v| v.as_i64());

        if let Some(ef_val) = ef {
            let body = serde_json::json!({
                "vectorIndexConfig": {
                    "ef": ef_val,
                }
            });

            let url = format!("{}/v1/schema/{}", self.base_url, self.class_name);
            let req = client
                .put(&url)
                .header("Content-Type", "application/json")
                .json(&body);
            let resp = self.add_auth(req).send().map_err(|e| e.to_string())?;

            if !resp.status().is_success() {
                eprintln!(
                    "Warning: failed to update vectorIndexConfig ef={}: {} {}",
                    ef_val,
                    resp.status(),
                    resp.text().unwrap_or_default()
                );
            }
        }

        Ok(())
    }
}

/// Convert integer ID to UUID (matches Python uuid.UUID(int=id))
fn id_to_uuid(id: i64) -> String {
    Uuid::from_u128(id as u128).to_string()
}

/// Convert UUID string back to integer
fn uuid_to_int(uuid_str: &str) -> Result<i64, String> {
    let uuid =
        Uuid::parse_str(uuid_str).map_err(|e| format!("Invalid UUID '{}': {}", uuid_str, e))?;
    Ok(uuid.as_u128() as i64)
}

/// Upload a batch of objects via Weaviate's batch API.
fn upload_batch_objects(
    client: &reqwest::blocking::Client,
    base_url: &str,
    class_name: &str,
    api_key: Option<&str>,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
) -> Result<(), String> {
    use vector_db_benchmark::readers::metadata::MetadataValue;

    let mut objects = Vec::with_capacity(ids.len());
    for i in 0..ids.len() {
        let uuid = id_to_uuid(ids[i]);

        let mut properties = serde_json::Map::new();
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
                        serde_json::json!({"latitude": lat, "longitude": lon})
                    }
                };
                properties.insert(k.clone(), val);
            }
        }

        objects.push(serde_json::json!({
            "class": class_name,
            "id": uuid,
            "properties": properties,
            "vector": vectors[i],
        }));
    }

    let body = serde_json::json!({
        "objects": objects,
    });

    let url = format!("{}/v1/batch/objects", base_url);
    let mut req = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body);

    if let Some(key) = api_key {
        req = req.header("Authorization", format!("Bearer {}", key));
    }

    let resp = req
        .send()
        .map_err(|e| format!("Batch upload failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Batch upload error: {} {}",
            resp.status(),
            resp.text().unwrap_or_default()
        ));
    }

    Ok(())
}

/// Search Weaviate using GraphQL near_vector query.
fn near_vector_search(
    client: &reqwest::blocking::Client,
    base_url: &str,
    class_name: &str,
    api_key: Option<&str>,
    query_vector: &[f32],
    top: usize,
    filter: Option<&serde_json::Value>,
) -> Result<Vec<(i64, f64)>, String> {
    // Build GraphQL query
    let vector_str: String = query_vector
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");

    let where_clause = if let Some(f) = filter {
        format!(", where: {}", serde_json::to_string(f).unwrap_or_default())
    } else {
        String::new()
    };

    let graphql = format!(
        r#"{{
            Get {{
                {class_name}(
                    nearVector: {{ vector: [{vector_str}] }},
                    limit: {top}
                    {where_clause}
                ) {{
                    _additional {{
                        id
                        distance
                    }}
                }}
            }}
        }}"#,
        class_name = class_name,
        vector_str = vector_str,
        top = top,
        where_clause = where_clause,
    );

    let body = serde_json::json!({"query": graphql});

    let url = format!("{}/v1/graphql", base_url);
    let mut req = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body);

    if let Some(key) = api_key {
        req = req.header("Authorization", format!("Bearer {}", key));
    }

    let resp = req
        .send()
        .map_err(|e| format!("GraphQL search failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "GraphQL search error: {} {}",
            resp.status(),
            resp.text().unwrap_or_default()
        ));
    }

    let resp_body: serde_json::Value = resp
        .json()
        .map_err(|e| format!("Failed to parse GraphQL response: {}", e))?;

    // Check for errors
    if let Some(errors) = resp_body.get("errors").and_then(|e| e.as_array()) {
        if !errors.is_empty() {
            return Err(format!("GraphQL errors: {:?}", errors));
        }
    }

    let results = resp_body
        .get("data")
        .and_then(|d| d.get("Get"))
        .and_then(|g| g.get(class_name))
        .and_then(|c| c.as_array())
        .ok_or_else(|| "Missing results in GraphQL response".to_string())?;

    let mut hits = Vec::with_capacity(results.len());
    for result in results {
        let additional = result.get("_additional").ok_or("Missing _additional")?;
        let id_str = additional
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or("Missing id")?;
        let distance = additional
            .get("distance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let id = uuid_to_int(id_str)?;
        hits.push((id, distance));
    }

    Ok(hits)
}

/// Parse conditions into Weaviate where filter format.
fn parse_weaviate_conditions(conditions: &serde_json::Value) -> Option<serde_json::Value> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    let mut operands = Vec::new();

    if let Some(and_items) = obj.get("and").and_then(|v| v.as_array()) {
        let and_filters: Vec<serde_json::Value> = and_items
            .iter()
            .filter_map(|entry| build_weaviate_entry_filter(entry))
            .collect();
        if !and_filters.is_empty() {
            if and_filters.len() == 1 {
                operands.push(and_filters.into_iter().next().unwrap());
            } else {
                operands.push(serde_json::json!({
                    "operator": "And",
                    "operands": and_filters,
                }));
            }
        }
    }

    if let Some(or_items) = obj.get("or").and_then(|v| v.as_array()) {
        let or_filters: Vec<serde_json::Value> = or_items
            .iter()
            .filter_map(|entry| build_weaviate_entry_filter(entry))
            .collect();
        if !or_filters.is_empty() {
            if or_filters.len() == 1 {
                operands.push(or_filters.into_iter().next().unwrap());
            } else {
                operands.push(serde_json::json!({
                    "operator": "Or",
                    "operands": or_filters,
                }));
            }
        }
    }

    if operands.is_empty() {
        return None;
    }
    if operands.len() == 1 {
        return Some(operands.into_iter().next().unwrap());
    }

    Some(serde_json::json!({
        "operator": "And",
        "operands": operands,
    }))
}

fn build_weaviate_entry_filter(entry: &serde_json::Value) -> Option<serde_json::Value> {
    let entry_obj = entry.as_object()?;
    let mut filters = Vec::new();

    for (field_name, field_filters) in entry_obj {
        if let Some(filter_obj) = field_filters.as_object() {
            for (cond_type, criteria) in filter_obj {
                if let Some(f) = build_weaviate_filter(field_name, cond_type, criteria) {
                    filters.push(f);
                }
            }
        }
    }

    if filters.is_empty() {
        None
    } else if filters.len() == 1 {
        Some(filters.into_iter().next().unwrap())
    } else {
        Some(serde_json::json!({
            "operator": "And",
            "operands": filters,
        }))
    }
}

fn build_weaviate_filter(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
) -> Option<serde_json::Value> {
    match condition_type {
        "match" => {
            let value = criteria.get("value")?;
            let value_key = if value.is_string() {
                "valueString"
            } else if value.is_number() {
                if value.is_i64() {
                    "valueInt"
                } else {
                    "valueNumber"
                }
            } else if value.is_boolean() {
                "valueBoolean"
            } else {
                "valueString"
            };
            Some(serde_json::json!({
                "path": [field_name],
                "operator": "Equal",
                value_key: value,
            }))
        }
        "range" => {
            let criteria_obj = criteria.as_object()?;
            let mut operands = Vec::new();

            let value_key = |v: &serde_json::Value| -> &str {
                if v.is_i64() {
                    "valueInt"
                } else {
                    "valueNumber"
                }
            };

            if let Some(lt) = criteria_obj.get("lt") {
                if !lt.is_null() {
                    operands.push(serde_json::json!({
                        "path": [field_name],
                        "operator": "LessThan",
                        value_key(lt): lt,
                    }));
                }
            }
            if let Some(gt) = criteria_obj.get("gt") {
                if !gt.is_null() {
                    operands.push(serde_json::json!({
                        "path": [field_name],
                        "operator": "GreaterThan",
                        value_key(gt): gt,
                    }));
                }
            }
            if let Some(lte) = criteria_obj.get("lte") {
                if !lte.is_null() {
                    operands.push(serde_json::json!({
                        "path": [field_name],
                        "operator": "LessThanEqual",
                        value_key(lte): lte,
                    }));
                }
            }
            if let Some(gte) = criteria_obj.get("gte") {
                if !gte.is_null() {
                    operands.push(serde_json::json!({
                        "path": [field_name],
                        "operator": "GreaterThanEqual",
                        value_key(gte): gte,
                    }));
                }
            }

            if operands.is_empty() {
                None
            } else if operands.len() == 1 {
                Some(operands.into_iter().next().unwrap())
            } else {
                Some(serde_json::json!({
                    "operator": "And",
                    "operands": operands,
                }))
            }
        }
        "geo" => {
            let lat = criteria.get("lat")?.as_f64()?;
            let lon = criteria.get("lon")?.as_f64()?;
            let radius = criteria
                .get("radius")
                .and_then(|r| r.as_f64())
                .unwrap_or(1000.0);
            Some(serde_json::json!({
                "path": [field_name],
                "operator": "WithinGeoRange",
                "valueGeoRange": {
                    "geoCoordinates": {
                        "latitude": lat,
                        "longitude": lon,
                    },
                    "distance": {
                        "max": radius,
                    }
                }
            }))
        }
        _ => None,
    }
}

impl Engine for WeaviateEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        let client = self.create_client()?;

        println!("Deleting existing class...");
        self.delete_class(&client)?;

        println!("Creating class '{}'...", self.class_name);
        self.create_class(&client, dataset)?;
        println!("Class '{}' created.", self.class_name);

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

        // Apply search-time vectorIndexConfig
        let client = self.create_client()?;
        self.setup_search(&client, params)?;

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<serde_json::Value>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_weaviate_conditions))
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

        std::thread::scope(|s| {
            for _ in 0..parallel {
                let base_url = self.base_url.clone();
                let class_name = self.class_name.clone();
                let api_key = self.api_key.clone();
                let timeout = self.timeout;
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
                            if n > 0 {
                                n
                            } else {
                                10
                            }
                        });

                        let query_start = Instant::now();
                        let results = near_vector_search(
                            &client,
                            &base_url,
                            &class_name,
                            api_key.as_deref(),
                            &queries[idx],
                            top,
                            parsed_filters[idx].as_ref(),
                        );
                        let query_time = query_start.elapsed().as_secs_f64();

                        search_times.lock().unwrap().push(query_time);

                        if let Ok(result_ids) = results {
                            let ordered_ids: Vec<i64> = result_ids.iter().map(|(id, _)| *id).collect();
                            let m = crate::metrics::compute_metrics(&ordered_ids, &neighbors[idx], top);
                            precisions.lock().unwrap().push(m.precision);
                            recalls.lock().unwrap().push(m.recall);
                            mrrs.lock().unwrap().push(m.mrr);
                            ndcgs.lock().unwrap().push(m.ndcg);
                        } else {
                            precisions.lock().unwrap().push(0.0);
                            recalls.lock().unwrap().push(0.0);
                            mrrs.lock().unwrap().push(0.0);
                            ndcgs.lock().unwrap().push(0.0);
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
        let client = self.create_client()?;
        self.delete_class(&client)
    }
}
