//! Milvus engine implementation.
//!
//! Uses the Milvus RESTful API (v2) via reqwest::blocking.
//! Supports HNSW index with configurable M/efConstruction,
//! multiple distance metrics (L2, IP, COSINE), and schema-based collections.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use vector_db_benchmark::readers::metadata::MetadataItem;

const DEFAULT_COLLECTION: &str = "Benchmark";

pub struct MilvusEngine {
    name: String,
    collection_name: String,
    timeout: u64,
    batch_size: usize,
    parallel: usize,
    base_url: String,
    search_params: Vec<SearchParams>,
    /// M and efConstruction from upload_params.index_params
    index_m: i64,
    index_ef_construction: i64,
    /// Distance metric type (L2, IP, COSINE)
    metric_type: String,
    /// Index type (HNSW, IVF_FLAT, etc.)
    index_type: String,
}

impl MilvusEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("MILVUS_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(19530);

        let collection_name = std::env::var("MILVUS_COLLECTION_NAME")
            .unwrap_or_else(|_| DEFAULT_COLLECTION.to_string());

        let timeout: u64 = std::env::var("MILVUS_TIMEOUT")
            .ok()
            .and_then(|v| v.parse().ok())
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

        // Extract index params from upload_params
        let index_params = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("index_params"));

        let index_m = index_params
            .and_then(|p| p.get("M"))
            .and_then(|v| v.as_i64())
            .unwrap_or(16);

        let index_ef_construction = index_params
            .and_then(|p| p.get("efConstruction"))
            .and_then(|v| v.as_i64())
            .unwrap_or(128);

        let index_type = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("index_type"))
            .and_then(|v| v.as_str())
            .unwrap_or("HNSW")
            .to_string();

        // Build base URL - Milvus REST API uses port 19530 (same as gRPC in newer versions)
        // or a dedicated REST port (9091 in some setups)
        let base_url = if host.starts_with("http") {
            host.to_string()
        } else {
            format!("http://{}:{}", host, port)
        };

        Ok(Self {
            name: engine_config.name.clone(),
            collection_name,
            timeout,
            batch_size,
            parallel,
            base_url,
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            index_m,
            index_ef_construction,
            metric_type: String::new(), // Set during configure
            index_type,
        })
    }

    fn create_client(&self) -> Result<reqwest::blocking::Client, String> {
        reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(self.timeout))
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

    fn drop_collection(&self, client: &reqwest::blocking::Client) -> Result<(), String> {
        let body = serde_json::json!({
            "collectionName": self.collection_name,
        });

        let url = format!("{}/v2/vectordb/collections/drop", self.base_url);
        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| e.to_string())?;

        // Ignore errors (collection might not exist)
        let _ = resp;
        Ok(())
    }

    fn create_collection(
        &self,
        client: &reqwest::blocking::Client,
        dataset: &Dataset,
    ) -> Result<(), String> {
        let vector_size = dataset.vector_size();

        // Build schema
        let mut fields = vec![
            serde_json::json!({
                "fieldName": "id",
                "dataType": "Int64",
                "isPrimary": true,
            }),
            serde_json::json!({
                "fieldName": "vector",
                "dataType": "FloatVector",
                "elementTypeParams": {
                    "dim": vector_size.to_string(),
                }
            }),
        ];

        // Add schema fields from dataset config
        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                for (field_name, field_type) in schema_obj {
                    let ft = field_type.as_str().unwrap_or("");
                    let milvus_type = match ft {
                        "int" => "Int64",
                        "keyword" | "text" => "VarChar",
                        "float" => "Double",
                        _ => continue,
                    };
                    let mut field = serde_json::json!({
                        "fieldName": field_name,
                        "dataType": milvus_type,
                    });
                    if milvus_type == "VarChar" {
                        field.as_object_mut().unwrap().insert(
                            "elementTypeParams".to_string(),
                            serde_json::json!({"max_length": "500"}),
                        );
                    }
                    fields.push(field);
                }
            }
        }

        let body = serde_json::json!({
            "collectionName": self.collection_name,
            "schema": {
                "fields": fields,
                "enableDynamicField": false,
            }
        });

        let url = format!("{}/v2/vectordb/collections/create", self.base_url);
        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| format!("Failed to create collection: {}", e))?;

        if !resp.status().is_success() {
            return Err(format!(
                "Failed to create collection: {} {}",
                resp.status(),
                resp.text().unwrap_or_default()
            ));
        }

        let resp_body: serde_json::Value = resp.json().unwrap_or_default();
        let code = resp_body.get("code").and_then(|c| c.as_i64()).unwrap_or(0);
        if code != 0 {
            let msg = resp_body
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error");
            return Err(format!("Failed to create collection: {}", msg));
        }

        Ok(())
    }

    fn create_index(&self, client: &reqwest::blocking::Client) -> Result<(), String> {
        let body = serde_json::json!({
            "collectionName": self.collection_name,
            "indexParams": [{
                "fieldName": "vector",
                "indexName": "vector_index",
                "metricType": self.metric_type,
                "indexType": self.index_type,
                "params": {
                    "M": self.index_m,
                    "efConstruction": self.index_ef_construction,
                }
            }]
        });

        let url = format!("{}/v2/vectordb/indexes/create", self.base_url);
        let resp = client
            .post(&url)
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

    fn load_collection(&self, client: &reqwest::blocking::Client) -> Result<(), String> {
        let body = serde_json::json!({
            "collectionName": self.collection_name,
        });

        let url = format!("{}/v2/vectordb/collections/load", self.base_url);
        let resp = client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| format!("Failed to load collection: {}", e))?;

        if !resp.status().is_success() {
            eprintln!(
                "Warning: load collection returned: {} {}",
                resp.status(),
                resp.text().unwrap_or_default()
            );
        }

        // Wait for loading to complete (exponential backoff: 1s, 2s, 4s, ... capped at 16s)
        println!("Waiting for collection to be loaded...");
        let mut backoff_secs = 1u64;
        let max_backoff = 16u64;
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(240);
        loop {
            std::thread::sleep(std::time::Duration::from_secs(backoff_secs));
            let body = serde_json::json!({
                "collectionName": self.collection_name,
            });
            let url = format!("{}/v2/vectordb/collections/get_load_state", self.base_url);
            if let Ok(resp) = client
                .post(&url)
                .header("Content-Type", "application/json")
                .json(&body)
                .send()
            {
                if let Ok(body) = resp.json::<serde_json::Value>() {
                    if let Some(data) = body.get("data") {
                        if let Some(state) = data.get("loadState").and_then(|s| s.as_str()) {
                            if state == "LoadStateLoaded" {
                                println!("Collection loaded.");
                                return Ok(());
                            }
                        }
                    }
                }
            }
            if std::time::Instant::now() > deadline {
                return Err("Timed out waiting for collection to load".to_string());
            }
            backoff_secs = (backoff_secs * 2).min(max_backoff);
        }
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
                        if let Err(e) = insert_batch(
                            &client,
                            &base_url,
                            &collection_name,
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
}

/// Insert a batch of vectors using Milvus REST API.
fn insert_batch(
    client: &reqwest::blocking::Client,
    base_url: &str,
    collection_name: &str,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
) -> Result<(), String> {
    use vector_db_benchmark::readers::metadata::MetadataValue;

    let mut data = Vec::with_capacity(ids.len());
    for i in 0..ids.len() {
        let mut row = serde_json::json!({
            "id": ids[i],
            "vector": vectors[i],
        });

        if let Some(meta) = &metadata[i] {
            let row_obj = row.as_object_mut().unwrap();
            for (k, v) in &meta.fields {
                let val = match v {
                    MetadataValue::String(s) => serde_json::Value::String(s.clone()),
                    MetadataValue::Labels(labels) => {
                        // For VarChar fields, join labels into single string
                        serde_json::Value::String(labels.join(","))
                    }
                    MetadataValue::Geo { lon, lat } => {
                        // Milvus doesn't support geo natively; store as string
                        serde_json::Value::String(format!("{},{}", lat, lon))
                    }
                };
                row_obj.insert(k.clone(), val);
            }
        }

        data.push(row);
    }

    let body = serde_json::json!({
        "collectionName": collection_name,
        "data": data,
    });

    let url = format!("{}/v2/vectordb/entities/insert", base_url);
    let resp = client
        .post(&url)
        .header("Content-Type", "application/json")
        .json(&body)
        .send()
        .map_err(|e| format!("Insert failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Insert error: {} {}",
            resp.status(),
            resp.text().unwrap_or_default()
        ));
    }

    let resp_body: serde_json::Value = resp.json().unwrap_or_default();
    let code = resp_body.get("code").and_then(|c| c.as_i64()).unwrap_or(0);
    if code != 0 {
        let msg = resp_body
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error");
        return Err(format!("Insert failed: {}", msg));
    }

    Ok(())
}

/// Search Milvus via REST API.
#[allow(clippy::too_many_arguments)]
/// Build and serialize one search request body to JSON bytes. Done OUTSIDE the
/// per-query timed window: serializing the query vector to JSON decimal text
/// (ryu formatting over every dimension) is client CPU work, not server latency.
/// Pre-serializing means the timed send only copies the finished bytes onto the
/// socket, matching the reference engines (pgvector/qdrant). The bytes are
/// identical to what `.json(&body)` would have sent inline.
fn build_search_body(
    collection_name: &str,
    query_vector: &[f32],
    top: usize,
    metric_type: &str,
    ef: Option<i64>,
    filter: Option<&str>,
) -> Vec<u8> {
    let mut body = serde_json::json!({
        "collectionName": collection_name,
        "data": [query_vector],
        "annsField": "vector",
        "limit": top,
        "outputFields": ["id"],
    });

    let mut search_params = serde_json::json!({
        "metric_type": metric_type,
    });
    if let Some(ef_val) = ef {
        search_params
            .as_object_mut()
            .unwrap()
            .insert("params".to_string(), serde_json::json!({"ef": ef_val}));
    }
    body.as_object_mut()
        .unwrap()
        .insert("searchParams".to_string(), search_params);

    if let Some(f) = filter {
        body.as_object_mut().unwrap().insert(
            "filter".to_string(),
            serde_json::Value::String(f.to_string()),
        );
    }

    serde_json::to_vec(&body).expect("serialize search body")
}

/// Send a pre-serialized search request and return the raw response text. Only
/// the network send + wire read live here (the caller wraps this in the timed
/// window); the response is parsed by `parse_search_response` AFTER `elapsed`.
fn send_search(
    client: &reqwest::blocking::Client,
    base_url: &str,
    body: &[u8],
) -> Result<String, String> {
    let url = format!("{}/v2/vectordb/entities/search", base_url);
    let resp = client
        .post(&url)
        .header("Content-Type", "application/json")
        .body(body.to_vec())
        .send()
        .map_err(|e| format!("Search failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!(
            "Search error: {} {}",
            resp.status(),
            resp.text().unwrap_or_default()
        ));
    }

    resp.text()
        .map_err(|e| format!("Failed to read search response: {}", e))
}

/// Parse a search response body (done AFTER the timed window — DOM
/// deserialization is client work, mirroring pgvector/qdrant).
fn parse_search_response(text: &str) -> Result<Vec<(i64, f64)>, String> {
    let resp_body: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| format!("Failed to parse search response: {}", e))?;

    let code = resp_body.get("code").and_then(|c| c.as_i64()).unwrap_or(0);
    if code != 0 {
        let msg = resp_body
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("unknown error");
        return Err(format!("Search failed: {}", msg));
    }

    let results = resp_body
        .get("data")
        .and_then(|d| d.as_array())
        .ok_or_else(|| "Missing data array in search response".to_string())?;

    let mut hits = Vec::with_capacity(results.len());
    for result in results {
        let id = result.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
        let distance = result
            .get("distance")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        hits.push((id, distance));
    }

    Ok(hits)
}

/// Parse conditions into Milvus filter expression.
/// Milvus uses string-based filter expressions like "field == value && field > 10"
fn parse_milvus_conditions(conditions: &serde_json::Value) -> Option<String> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    let mut clauses = Vec::new();

    if let Some(and_items) = obj.get("and").and_then(|v| v.as_array()) {
        let and_filters: Vec<String> = and_items
            .iter()
            .filter_map(build_milvus_entry_filter)
            .collect();
        if !and_filters.is_empty() {
            clauses.push(format!("({})", and_filters.join(" && ")));
        }
    }

    if let Some(or_items) = obj.get("or").and_then(|v| v.as_array()) {
        let or_filters: Vec<String> = or_items
            .iter()
            .filter_map(build_milvus_entry_filter)
            .collect();
        if !or_filters.is_empty() {
            clauses.push(format!("({})", or_filters.join(" || ")));
        }
    }

    if clauses.is_empty() {
        None
    } else {
        Some(clauses.join(" && "))
    }
}

fn build_milvus_entry_filter(entry: &serde_json::Value) -> Option<String> {
    let entry_obj = entry.as_object()?;
    let mut filters = Vec::new();

    for (field_name, field_filters) in entry_obj {
        if let Some(filter_obj) = field_filters.as_object() {
            for (cond_type, criteria) in filter_obj {
                if let Some(f) = build_milvus_filter(field_name, cond_type, criteria) {
                    filters.push(f);
                }
            }
        }
    }

    if filters.is_empty() {
        None
    } else {
        Some(filters.join(" && "))
    }
}

/// Quote a string value for a Milvus filter expression.
///
/// Milvus expressions use double-quoted string literals. Backslashes and
/// double-quotes inside the value must be escaped (backslash first, so we don't
/// double-escape the backslashes we just introduced), otherwise a value such as
/// `15"laptop` would terminate the literal early and produce a malformed /
/// injectable expression that errors the whole search. Returns the value WITH
/// its surrounding double-quotes so callers can inline it directly.
fn quote_milvus_string(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

fn build_milvus_filter(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
) -> Option<String> {
    match condition_type {
        "match" => {
            // match_any: field value in a list -> Milvus `in [...]` expression,
            // the OR-of-values semantics that mirror qdrant's
            // Condition::matches(field, Vec). Strings are quoted/escaped,
            // numbers inlined; bool/null/nested items are skipped so an invalid
            // expression is never produced. An empty (or all-skipped) IN-set
            // matches NOTHING: we still emit `field in []` (a valid
            // match-nothing expression) rather than dropping the clause, since
            // dropping the sole clause would leave no filter and silently
            // return every row (the inverse of the filter).
            if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                // NOTE: match_any here supports SINGLE-VALUED keyword fields only.
                // Milvus stores a multi-valued keyword field (MetadataValue::Labels,
                // only for a field literally named `labels`) as a single
                // comma-joined VarChar (see insert_batch: `labels.join(",")`), so
                // `field in ["red","blue"]` tests whole-string set membership and
                // cannot match a doc stored as `"red,green"`. True contains-any
                // semantics (as pgvector's array-overlap / mongodb's array $in) would
                // require changing storage to ARRAY(VarChar) + array_contains_any,
                // which is out of scope here (no dataset to test against) and tracked
                // as a follow-up. Strings are quoted/escaped, numbers inlined;
                // bool/null/nested items are skipped so an invalid expression is
                // never produced. An empty (or all-skipped) IN-set matches NOTHING:
                // we still emit `field in []` (a valid match-nothing expression)
                // rather than dropping the clause, since dropping the sole clause
                // would leave no filter and silently return every row.
                let items: Vec<String> = any
                    .iter()
                    .filter_map(|v| {
                        if let Some(s) = v.as_str() {
                            Some(quote_milvus_string(s))
                        } else if v.is_number() {
                            Some(v.to_string())
                        } else {
                            None
                        }
                    })
                    .collect();
                return Some(format!("{} in [{}]", field_name, items.join(", ")));
            }
            let value = criteria.get("value")?;
            if let Some(s) = value.as_str() {
                Some(format!("{} == {}", field_name, quote_milvus_string(s)))
            } else {
                Some(format!("{} == {}", field_name, value))
            }
        }
        "range" => {
            let criteria_obj = criteria.as_object()?;
            let mut clauses = Vec::new();
            if let Some(lt) = criteria_obj.get("lt") {
                if !lt.is_null() {
                    clauses.push(format!("{} < {}", field_name, lt));
                }
            }
            if let Some(gt) = criteria_obj.get("gt") {
                if !gt.is_null() {
                    clauses.push(format!("{} > {}", field_name, gt));
                }
            }
            if let Some(lte) = criteria_obj.get("lte") {
                if !lte.is_null() {
                    clauses.push(format!("{} <= {}", field_name, lte));
                }
            }
            if let Some(gte) = criteria_obj.get("gte") {
                if !gte.is_null() {
                    clauses.push(format!("{} >= {}", field_name, gte));
                }
            }
            if clauses.is_empty() {
                None
            } else {
                Some(format!("({})", clauses.join(" && ")))
            }
        }
        "geo" => {
            // Milvus doesn't support geo natively
            None
        }
        _ => None,
    }
}

impl Engine for MilvusEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        let distance = dataset.distance();
        let dist_lower = distance.to_lowercase();

        // Map distance metric
        self.metric_type = match dist_lower.as_str() {
            "l2" | "euclidean" => "L2".to_string(),
            "dot" | "ip" => "IP".to_string(),
            "cosine" | "angular" => "IP".to_string(), // Milvus uses IP for cosine (normalized vectors)
            other => return Err(format!("Unsupported distance metric for Milvus: {}", other)),
        };

        let client = self.create_client()?;

        println!("Dropping existing collection...");
        self.drop_collection(&client)?;

        println!("Creating collection '{}'...", self.collection_name);
        self.create_collection(&client, dataset)?;

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

        // Create index after upload, then load the collection into memory. Both
        // are part of the ingest cost and are included in total_time for
        // cross-engine comparability (mirrors mongodb; matches v0's post_upload()
        // timing).
        let client = self.create_client()?;
        println!(
            "Creating {} index (M={}, efConstruction={}, metric={})...",
            self.index_type, self.index_m, self.index_ef_construction, self.metric_type
        );
        let index_start = Instant::now();
        self.create_index(&client)?;

        // Load collection into memory
        self.load_collection(&client)?;
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

        // Extract ef from search params
        let ef = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.ef)
            .or_else(|| {
                params
                    .extra
                    .as_ref()
                    .and_then(|e| e.get("params"))
                    .and_then(|p| p.get("ef"))
                    .and_then(|v| v.as_i64())
            });

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<String>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_milvus_conditions))
            .collect();

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        // Precompute per-query `top` and the fully serialized request bodies
        // BEFORE the parallel region so the timed window wraps only the RPC
        // round-trip (see build_search_body). `tops[idx]` reproduces the same k
        // the request embeds, so recall is computed against an identical result
        // set.
        let tops: Vec<usize> = (0..num_to_run)
            .map(|idx| {
                explicit_top.unwrap_or_else(|| {
                    let n = neighbors[idx].len();
                    if n > 0 {
                        n
                    } else {
                        10
                    }
                })
            })
            .collect();
        let bodies: Vec<Vec<u8>> = (0..num_to_run)
            .map(|idx| {
                build_search_body(
                    &self.collection_name,
                    &queries[idx],
                    tops[idx],
                    &self.metric_type,
                    ef,
                    parsed_filters[idx].as_deref(),
                )
            })
            .collect();

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
                let base_url = self.base_url.clone();
                let timeout = self.timeout;
                let neighbors = &neighbors;
                let tops = &tops;
                let bodies = &bodies;
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                handles.push(s.spawn(move || {
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();

                    let client = match reqwest::blocking::Client::builder()
                        .timeout(std::time::Duration::from_secs(timeout))
                        .build()
                    {
                        Ok(c) => c,
                        Err(_) => return (t, p, r, mr, nd),
                    };

                    loop {
                        let idx = query_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_to_run {
                            break;
                        }

                        let top = tops[idx];

                        // Timed window: only the network send + wire read of the
                        // raw response. Body is pre-serialized; response is parsed
                        // after `elapsed`.
                        let query_start = Instant::now();
                        let response = send_search(&client, &base_url, &bodies[idx]);
                        let query_time = query_start.elapsed().as_secs_f64();

                        match response.and_then(|text| parse_search_response(&text)) {
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

        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));
        crate::engine::compute_search_stats(
            &times, &precs, &recs, &mrr_vals, &ndcg_vals, total_time, top, parallel, num_to_run,
        )
    }

    fn delete(&mut self) -> Result<(), String> {
        let client = self.create_client()?;
        self.drop_collection(&client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn match_any_string_list_emits_in() {
        let e = json!({"and": [{"color": {"match": {"any": ["red", "blue"]}}}]});
        let expr = parse_milvus_conditions(&e).unwrap();
        assert!(
            expr.contains(r#"color in ["red", "blue"]"#),
            "expr={}",
            expr
        );
    }

    #[test]
    fn match_any_int_list_emits_in() {
        let e = json!({"and": [{"size": {"match": {"any": [1, 2, 3]}}}]});
        let expr = parse_milvus_conditions(&e).unwrap();
        assert!(expr.contains("size in [1, 2, 3]"), "expr={}", expr);
    }

    #[test]
    fn match_any_empty_list_matches_nothing() {
        // Empty IN-set must match NOTHING (never invert to returning all rows):
        // `field in []`, not a dropped clause.
        let e = json!({"and": [{"color": {"match": {"any": []}}}]});
        let expr = parse_milvus_conditions(&e).unwrap();
        assert!(expr.contains("color in []"), "expr={}", expr);
    }

    #[test]
    fn match_exact_value_still_works() {
        assert_eq!(
            build_milvus_filter("color", "match", &json!({"value": "red"})).unwrap(),
            r#"color == "red""#
        );
    }

    #[test]
    fn match_exact_value_escapes_quotes_and_backslashes() {
        // Exact-value string branch must escape through the shared helper:
        // backslash first (so introduced backslashes aren't doubled), then quote.
        assert_eq!(
            build_milvus_filter("color", "match", &json!({"value": r#"a"b\c"#})).unwrap(),
            r#"color == "a\"b\\c""#
        );
    }

    #[test]
    fn match_any_escapes_quotes_and_backslashes() {
        // A keyword value containing BOTH a double-quote and a backslash must be
        // escaped identically to the exact-value branch (shared helper). Input
        // `a"b\c` -> literal `"a\"b\\c"`.
        let e = json!({"and": [{"color": {"match": {"any": [r#"a"b\c"#]}}}]});
        let expr = parse_milvus_conditions(&e).unwrap();
        assert_eq!(expr, r#"(color in ["a\"b\\c"])"#, "expr={}", expr);
    }
}
