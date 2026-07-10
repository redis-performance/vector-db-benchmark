//! Weaviate engine implementation.
//!
//! Uses Weaviate's REST API (v1) via reqwest::blocking.
//! Supports HNSW vector index with configurable efConstruction/maxConnections,
//! schema-based properties, and near_vector search.

use std::collections::HashMap;
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
                    if let Some(prop) = weaviate_property(field_name, ft) {
                        properties.push(prop);
                    }
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
        schema_types: &HashMap<String, String>,
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
                let schema_types = schema_types.clone();
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
                            &schema_types,
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

    /// Update the class's vectorIndexConfig `ef` for search-time tuning.
    ///
    /// In Weaviate, query-time `ef` is a class-level schema setting, not a
    /// per-query parameter. Updating it requires `PUT /v1/schema/{class}` with the
    /// *complete* class object: a partial body (just `vectorIndexConfig`) is
    /// rejected with 422 ("class name is immutable: attempted change from
    /// \"<class>\" to \"\""), which would silently leave every query running at the
    /// default dynamic ef (-1) and flatten the recall/QPS sweep. So we fetch the
    /// current class, merge the new `ef` into its `vectorIndexConfig`, and PUT the
    /// whole object back — immutable fields (efConstruction, maxConnections) are
    /// sent back unchanged, so they don't trip the immutability check. A failure
    /// here returns Err (rather than a swallowed warning) so a broken ef sweep
    /// surfaces immediately instead of producing misleading results.
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

        let Some(ef_val) = ef else {
            return Ok(());
        };

        let url = format!("{}/v1/schema/{}", self.base_url, self.class_name);

        // 1. Fetch the current class definition.
        let get_req = client.get(&url);
        let get_resp = self
            .add_auth(get_req)
            .send()
            .map_err(|e| format!("Failed to GET class for ef update: {}", e))?;
        if !get_resp.status().is_success() {
            return Err(format!(
                "Failed to GET class {} for ef update: {} {}",
                self.class_name,
                get_resp.status(),
                get_resp.text().unwrap_or_default()
            ));
        }
        let mut class_obj: serde_json::Value = get_resp
            .json()
            .map_err(|e| format!("Failed to parse class JSON for ef update: {}", e))?;

        // 2. Merge the new ef into vectorIndexConfig (creating it if absent).
        let obj = class_obj
            .as_object_mut()
            .ok_or_else(|| "class definition is not a JSON object".to_string())?;
        let vic = obj
            .entry("vectorIndexConfig")
            .or_insert_with(|| serde_json::json!({}));
        let vic_obj = vic
            .as_object_mut()
            .ok_or_else(|| "vectorIndexConfig is not a JSON object".to_string())?;
        vic_obj.insert("ef".to_string(), serde_json::json!(ef_val));

        // 3. PUT the full class object back.
        let put_req = client
            .put(&url)
            .header("Content-Type", "application/json")
            .json(&class_obj);
        let put_resp = self
            .add_auth(put_req)
            .send()
            .map_err(|e| format!("Failed to PUT class for ef update: {}", e))?;
        if !put_resp.status().is_success() {
            return Err(format!(
                "Failed to update vectorIndexConfig ef={}: {} {}",
                ef_val,
                put_resp.status(),
                put_resp.text().unwrap_or_default()
            ));
        }

        Ok(())
    }
}

/// Build a Weaviate schema property from a dataset schema `field_type`.
///
/// Returns `None` for unsupported field types (skipped). Filtering requires an
/// inverted index: modern Weaviate (>= 1.19, incl. 1.38) replaced the
/// deprecated `indexInverted` flag with `indexFilterable` (roaring-bitmap
/// filter index) and `indexSearchable` (BM25). `Equal` filters read the
/// FILTERABLE index, so it must be enabled explicitly. `indexSearchable` is
/// only valid on text-backed properties, so it is set only for those.
fn weaviate_property(field_name: &str, field_type: &str) -> Option<serde_json::Value> {
    let wv_type = match field_type {
        "int" => "int",
        "keyword" | "text" => "text",
        "float" => "number",
        "geo" => "geoCoordinates",
        _ => return None,
    };
    let mut prop = serde_json::json!({
        "name": field_name,
        "dataType": [wv_type],
        "indexFilterable": true,
    });
    let obj = prop.as_object_mut().unwrap();
    // Keyword fields must match on the WHOLE value (exact keyword equality,
    // like qdrant). Weaviate's default `word` tokenization turns `Equal` into
    // token-containment, so `Equal "Blue"` would also match "Dark Blue". Force
    // `field` tokenization for keyword; keep `word` for full-text. Tokenization
    // and `indexSearchable` apply only to text-backed properties.
    if let Some(tok) = match field_type {
        "keyword" => Some("field"),
        "text" => Some("word"),
        _ => None,
    } {
        obj.insert("tokenization".to_string(), serde_json::json!(tok));
        obj.insert("indexSearchable".to_string(), serde_json::json!(true));
    }
    Some(prop)
}

/// Convert a dataset metadata value into a JSON value typed for Weaviate's
/// strict schema. Numbers arrive from the reader as strings (see
/// `parse_metadata_from_json`); Weaviate rejects a whole object if e.g. an
/// `int` property receives a string, so numeric fields must be coerced back to
/// JSON numbers using the dataset `schema_type`. Non-numeric fields pass
/// through unchanged.
fn coerce_metadata_value(
    schema_type: Option<&str>,
    value: &vector_db_benchmark::readers::metadata::MetadataValue,
) -> serde_json::Value {
    use vector_db_benchmark::readers::metadata::MetadataValue;
    match value {
        MetadataValue::String(s) => match schema_type {
            Some("int") => s
                .parse::<i64>()
                .map(serde_json::Value::from)
                .unwrap_or_else(|_| serde_json::Value::String(s.clone())),
            Some("float") => s
                .parse::<f64>()
                .map(serde_json::Value::from)
                .unwrap_or_else(|_| serde_json::Value::String(s.clone())),
            _ => serde_json::Value::String(s.clone()),
        },
        MetadataValue::Labels(labels) => serde_json::Value::Array(
            labels
                .iter()
                .map(|l| serde_json::Value::String(l.clone()))
                .collect(),
        ),
        MetadataValue::Geo { lon, lat } => {
            serde_json::json!({"latitude": lat, "longitude": lon})
        }
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
#[allow(clippy::too_many_arguments)]
fn upload_batch_objects(
    client: &reqwest::blocking::Client,
    base_url: &str,
    class_name: &str,
    api_key: Option<&str>,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
    schema_types: &HashMap<String, String>,
) -> Result<(), String> {
    let mut objects = Vec::with_capacity(ids.len());
    for i in 0..ids.len() {
        let uuid = id_to_uuid(ids[i]);

        let mut properties = serde_json::Map::new();
        if let Some(meta) = &metadata[i] {
            for (k, v) in &meta.fields {
                let val = coerce_metadata_value(schema_types.get(k).map(|s| s.as_str()), v);
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

    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(format!("Batch upload error: {} {}", status, text));
    }

    // The batch API returns HTTP 200 even when individual objects fail schema
    // validation (e.g. a value whose type does not match the property). Those
    // objects are silently NOT stored, which would leave the collection empty
    // and make every filtered search return zero hits. Inspect each object's
    // per-item `result.status` and surface the first failure.
    if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&text) {
        if let Some(arr) = parsed.as_array() {
            for obj in arr {
                let status = obj
                    .get("result")
                    .and_then(|r| r.get("status"))
                    .and_then(|s| s.as_str());
                if let Some(st) = status {
                    if st != "SUCCESS" {
                        let msg = obj
                            .pointer("/result/errors/error/0/message")
                            .and_then(|m| m.as_str())
                            .unwrap_or("unknown error");
                        return Err(format!(
                            "Batch object import failed (status {}): {}",
                            st, msg
                        ));
                    }
                }
            }
        }
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
        format!(", where: {}", json_to_graphql_literal(f))
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
            .filter_map(build_weaviate_entry_filter)
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
            .filter_map(build_weaviate_entry_filter)
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

/// Serialize a where-filter `Value` as a **GraphQL object literal** (not JSON).
///
/// Weaviate's GraphQL `where` argument requires object keys as bare names and
/// the `operator` value as an unquoted enum (`operator: Equal`), whereas
/// `serde_json::to_string` emits quoted keys/values (`"operator":"Equal"`),
/// which the GraphQL parser rejects with a syntax error. Object keys are
/// emitted unquoted, the `operator` field's value is emitted as an unquoted
/// enum, and every other scalar keeps normal JSON quoting/formatting.
fn json_to_graphql_literal(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Object(map) => {
            let fields: Vec<String> = map
                .iter()
                .map(|(k, val)| {
                    if k == "operator" {
                        // GraphQL enum value: unquoted.
                        format!("{}: {}", k, val.as_str().unwrap_or_default())
                    } else {
                        format!("{}: {}", k, json_to_graphql_literal(val))
                    }
                })
                .collect();
            format!("{{{}}}", fields.join(", "))
        }
        serde_json::Value::Array(arr) => {
            let items: Vec<String> = arr.iter().map(json_to_graphql_literal).collect();
            format!("[{}]", items.join(", "))
        }
        serde_json::Value::String(s) => {
            format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
        }
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Null => "null".to_string(),
    }
}

fn build_weaviate_filter(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
) -> Option<serde_json::Value> {
    match condition_type {
        "match" => {
            // Value typing shared by exact match and match_any.
            let value_key = |v: &serde_json::Value| -> &'static str {
                if v.is_string() {
                    "valueText"
                } else if v.is_i64() {
                    "valueInt"
                } else if v.is_number() {
                    "valueNumber"
                } else if v.is_boolean() {
                    "valueBoolean"
                } else {
                    "valueText"
                }
            };

            // match_any: field value in a list -> OR of `Equal` conditions,
            // reusing the engine's proven exact-match path (same OR-of-values
            // semantics as qdrant's Condition::matches(field, Vec)). An empty
            // IN-set matches NOTHING: emit an unsatisfiable `Equal(x) AND
            // NotEqual(x)` rather than dropping the clause, since dropping the
            // sole clause would return every object (the inverse of the filter).
            if let Some(any) = criteria.get("any").and_then(|v| v.as_array()) {
                let operands: Vec<serde_json::Value> = any
                    .iter()
                    .map(|v| {
                        let vk = value_key(v);
                        serde_json::json!({
                            "path": [field_name],
                            "operator": "Equal",
                            vk: v,
                        })
                    })
                    .collect();
                return Some(match operands.len() {
                    0 => {
                        const NEVER: &str = "__match_any_never_match__";
                        serde_json::json!({
                            "operator": "And",
                            "operands": [
                                {"path": [field_name], "operator": "Equal", "valueText": NEVER},
                                {"path": [field_name], "operator": "NotEqual", "valueText": NEVER},
                            ]
                        })
                    }
                    1 => operands.into_iter().next().unwrap(),
                    _ => serde_json::json!({"operator": "Or", "operands": operands}),
                });
            }

            let value = criteria.get("value")?;
            let vk = value_key(value);
            Some(serde_json::json!({
                "path": [field_name],
                "operator": "Equal",
                vk: value,
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
        // Map field name -> dataset schema type so uploads can coerce values
        // to the types Weaviate's strict schema expects (e.g. int, not string).
        let schema_types: HashMap<String, String> = dataset
            .config
            .schema
            .as_ref()
            .and_then(|s| s.as_object())
            .map(|o| {
                o.iter()
                    .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                    .collect()
            })
            .unwrap_or_default();

        let upload_start = Instant::now();
        self.upload_parallel(&ids, &vectors, &metadata, &schema_types)?;
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
                let class_name = self.class_name.clone();
                let api_key = self.api_key.clone();
                let timeout = self.timeout;
                let queries = &queries;
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
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

                        match results {
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
        self.delete_class(&client)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn graphql_literal_uses_bare_keys_and_enum_operator() {
        let f = build_weaviate_filter("color", "match", &json!({"any": ["red", "blue"]})).unwrap();
        let g = json_to_graphql_literal(&f);
        assert!(g.contains("operator: Or"), "g={}", g);
        assert!(g.contains("operator: Equal"), "g={}", g);
        assert!(g.contains("path: [\"color\"]"), "g={}", g);
        assert!(g.contains("valueText: \"red\""), "g={}", g);
        assert!(!g.contains("\"operator\""), "g={}", g);
        assert!(!g.contains("\"path\""), "g={}", g);
    }

    #[test]
    fn graphql_literal_numbers_unquoted() {
        let f = json!({"path": ["size"], "operator": "Equal", "valueInt": 3});
        let g = json_to_graphql_literal(&f);
        assert!(g.contains("valueInt: 3"), "g={}", g);
        assert!(!g.contains("\"valueInt\""), "g={}", g);
    }

    #[test]
    fn match_any_string_list_emits_or_of_equal() {
        let c = build_weaviate_filter("color", "match", &json!({"any": ["red", "blue"]})).unwrap();
        assert_eq!(c["operator"], "Or");
        let ops = c["operands"].as_array().unwrap();
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0]["operator"], "Equal");
        assert_eq!(ops[0]["path"], json!(["color"]));
        assert_eq!(ops[0]["valueText"], "red");
        assert_eq!(ops[1]["valueText"], "blue");
    }

    #[test]
    fn match_any_int_list_emits_or_of_equal() {
        let c = build_weaviate_filter("size", "match", &json!({"any": [1, 2, 3]})).unwrap();
        assert_eq!(c["operator"], "Or");
        let ops = c["operands"].as_array().unwrap();
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0]["operator"], "Equal");
        assert_eq!(ops[0]["valueInt"], 1);
    }

    #[test]
    fn match_any_single_element_is_bare_equal() {
        let c = build_weaviate_filter("color", "match", &json!({"any": ["red"]})).unwrap();
        assert_eq!(c["operator"], "Equal");
        assert_eq!(c["valueText"], "red");
    }

    #[test]
    fn match_any_empty_list_matches_nothing() {
        // Empty IN-set -> unsatisfiable And(Equal(x), NotEqual(x)): matches
        // nothing (clause not dropped, never inverted to match-all).
        let c = build_weaviate_filter("color", "match", &json!({"any": []})).unwrap();
        assert_eq!(c["operator"], "And");
        let ops = c["operands"].as_array().unwrap();
        assert_eq!(ops[0]["operator"], "Equal");
        assert_eq!(ops[1]["operator"], "NotEqual");
        assert_eq!(ops[0]["valueText"], ops[1]["valueText"]);
    }

    #[test]
    fn match_exact_value_still_works() {
        let c = build_weaviate_filter("color", "match", &json!({"value": "red"})).unwrap();
        assert_eq!(c["operator"], "Equal");
        assert_eq!(c["valueText"], "red");
    }

    #[test]
    fn keyword_property_is_filterable_with_field_tokenization() {
        // Modern Weaviate needs `indexFilterable` (not the deprecated
        // `indexInverted`) for `Equal` filters to match; keyword uses `field`
        // tokenization so `Equal "red"` matches the whole value exactly.
        let p = weaviate_property("color", "keyword").unwrap();
        assert_eq!(p["dataType"], json!(["text"]));
        assert_eq!(p["indexFilterable"], true);
        assert_eq!(p["tokenization"], "field");
        assert_eq!(p["indexSearchable"], true);
        assert!(p.get("indexInverted").is_none(), "p={}", p);
    }

    #[test]
    fn int_property_is_filterable_without_searchable() {
        // `indexSearchable` is invalid on non-text properties and would make
        // Weaviate reject the whole schema; it must be omitted for int.
        let p = weaviate_property("size", "int").unwrap();
        assert_eq!(p["dataType"], json!(["int"]));
        assert_eq!(p["indexFilterable"], true);
        assert!(p.get("indexSearchable").is_none(), "p={}", p);
        assert!(p.get("tokenization").is_none(), "p={}", p);
    }

    #[test]
    fn unsupported_property_type_is_skipped() {
        assert!(weaviate_property("x", "bogus").is_none());
    }

    #[test]
    fn coerce_int_string_to_json_number() {
        use vector_db_benchmark::readers::metadata::MetadataValue;
        // The reader hands numbers over as strings; an `int` schema field must
        // become a JSON number or Weaviate rejects the whole object.
        let v = MetadataValue::String("1".to_string());
        assert_eq!(coerce_metadata_value(Some("int"), &v), json!(1));
        let f = MetadataValue::String("1.5".to_string());
        assert_eq!(coerce_metadata_value(Some("float"), &f), json!(1.5));
    }

    #[test]
    fn coerce_keyword_stays_string() {
        use vector_db_benchmark::readers::metadata::MetadataValue;
        let v = MetadataValue::String("red".to_string());
        assert_eq!(coerce_metadata_value(Some("keyword"), &v), json!("red"));
        // Unknown/unmapped fields pass through unchanged.
        assert_eq!(coerce_metadata_value(None, &v), json!("red"));
    }
}
