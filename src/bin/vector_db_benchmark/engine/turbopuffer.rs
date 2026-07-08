//! Turbopuffer engine implementation.
//!
//! Uses the `turbopuffer-client` crate (async) with a tokio runtime.
//! Turbopuffer is a cloud-only vector database. Requires an API key
//! set via the TURBOPUFFER_API_KEY environment variable.
//!
//! Supports filtering on metadata attributes using Turbopuffer's
//! filter operators (Eq, In, Gt, Lt, etc.) and logical combinators.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use turbopuffer_client::Client;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use vector_db_benchmark::readers::metadata::MetadataItem;

/// One upload batch: ids, their vectors, and optional per-item metadata.
type UploadBatch = (Vec<i64>, Vec<Vec<f32>>, Vec<Option<MetadataItem>>);

const DEFAULT_NAMESPACE: &str = "benchmark";

pub struct TurbopufferEngine {
    name: String,
    namespace: String,
    batch_size: usize,
    parallel: usize,
    search_params: Vec<SearchParams>,
    distance_metric: String,
    /// Tokio runtime for async operations
    rt: tokio::runtime::Runtime,
    /// Shared client (wrapped in Arc for thread-safe sharing)
    client: Arc<Client>,
}

impl TurbopufferEngine {
    pub fn new(engine_config: &EngineConfig, _host: &str) -> Result<Self, String> {
        let api_key = std::env::var("TURBOPUFFER_API_KEY").map_err(|_| {
            "TURBOPUFFER_API_KEY environment variable is required for turbopuffer engine"
                .to_string()
        })?;

        let namespace = std::env::var("TURBOPUFFER_NAMESPACE")
            .unwrap_or_else(|_| DEFAULT_NAMESPACE.to_string());

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
            .unwrap_or(1000) as usize;

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

        let client = Arc::new(Client::new(&api_key));

        Ok(Self {
            name: engine_config.name.clone(),
            namespace,
            batch_size,
            parallel,
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            distance_metric: "cosine_distance".to_string(),
            rt,
            client,
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

    /// Upload a batch of vectors with metadata
    fn upload_batch(
        client: &Client,
        namespace: &str,
        ids: &[i64],
        vectors: &[Vec<f32>],
        metadata: &[Option<MetadataItem>],
        rt: &tokio::runtime::Runtime,
    ) -> Result<(), String> {
        let ids_json: Vec<serde_json::Value> = ids.iter().map(|id| serde_json::json!(id)).collect();
        let vecs_json: Vec<Vec<f64>> = vectors
            .iter()
            .map(|v| v.iter().map(|f| *f as f64).collect())
            .collect();

        let mut body = serde_json::json!({
            "ids": ids_json,
            "vectors": vecs_json,
        });

        // Build attributes from metadata
        if metadata.iter().any(|m| m.is_some()) {
            let mut attributes: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
            // Collect all unique field names from all metadata items
            let mut field_names: Vec<String> = Vec::new();
            for item in metadata.iter().flatten() {
                for (key, _) in &item.fields {
                    if !field_names.contains(key) {
                        field_names.push(key.clone());
                    }
                }
            }

            for field_name in &field_names {
                let values: Vec<serde_json::Value> = metadata
                    .iter()
                    .map(|m| {
                        m.as_ref()
                            .and_then(|item| {
                                item.fields
                                    .iter()
                                    .find(|(k, _)| k == field_name)
                                    .map(|(_, v)| metadata_value_to_json(v))
                            })
                            .unwrap_or(serde_json::Value::Null)
                    })
                    .collect();
                attributes.insert(field_name.clone(), serde_json::Value::Array(values));
            }

            body["attributes"] = serde_json::Value::Object(attributes);
        }

        let ns = client.namespace(namespace);
        rt.block_on(async { ns.upsert(&body).await })
            .map_err(|e| format!("Turbopuffer upsert failed: {}", e))?;

        Ok(())
    }
}

/// Convert a MetadataValue to a JSON value for Turbopuffer attributes
fn metadata_value_to_json(val: &MetadataValue) -> serde_json::Value {
    match val {
        MetadataValue::String(s) => {
            // Try to parse as number for numeric attributes
            if let Ok(n) = s.parse::<i64>() {
                serde_json::json!(n)
            } else if let Ok(f) = s.parse::<f64>() {
                serde_json::json!(f)
            } else {
                serde_json::json!(s)
            }
        }
        MetadataValue::Labels(labels) => serde_json::json!(labels),
        MetadataValue::Geo { lon, lat } => {
            serde_json::json!({"lon": lon, "lat": lat})
        }
    }
}

use vector_db_benchmark::readers::metadata::MetadataValue;

/// Convert benchmark distance metric to Turbopuffer distance_metric name
fn to_turbopuffer_metric(distance: &str) -> &str {
    match distance {
        "cosine" | "angular" => "cosine_distance",
        "l2" | "euclidean" => "euclidean_squared",
        "dot" | "ip" => "cosine_distance", // turbopuffer doesn't have dot product, cosine is closest
        _ => "cosine_distance",
    }
}

/// Parse benchmark condition JSON into Turbopuffer filter format.
///
/// Benchmark conditions look like:
///   { "and": [{"field_name": {"match": {"value": "..."}}}] }
/// or:
///   { "and": [{"field_name": {"range": {"gt": 5}}}] }
///
/// Turbopuffer filters look like:
///   ["And", [["field_name", "Eq", value], ...]]
fn parse_turbopuffer_filter(conditions: &serde_json::Value) -> Option<serde_json::Value> {
    let obj = conditions.as_object()?;

    // Handle "and" / "or" top-level keys
    for (key, value) in obj {
        match key.as_str() {
            "and" => {
                let clauses = value.as_array()?;
                let mut tpf_clauses = Vec::new();
                for clause in clauses {
                    if let Some(f) = parse_single_condition(clause) {
                        tpf_clauses.push(f);
                    }
                }
                if tpf_clauses.is_empty() {
                    return None;
                }
                return Some(serde_json::json!(["And", tpf_clauses]));
            }
            "or" => {
                let clauses = value.as_array()?;
                let mut tpf_clauses = Vec::new();
                for clause in clauses {
                    if let Some(f) = parse_single_condition(clause) {
                        tpf_clauses.push(f);
                    }
                }
                if tpf_clauses.is_empty() {
                    return None;
                }
                return Some(serde_json::json!(["Or", tpf_clauses]));
            }
            _ => {
                // Single condition at top level
                if let Some(f) = parse_single_condition(conditions) {
                    return Some(f);
                }
            }
        }
    }
    None
}

/// Parse a single condition like {"field_name": {"match": {"value": "..."}} }
fn parse_single_condition(condition: &serde_json::Value) -> Option<serde_json::Value> {
    let obj = condition.as_object()?;
    for (field_name, constraint) in obj {
        let constraint_obj = constraint.as_object()?;
        for (op, operand) in constraint_obj {
            match op.as_str() {
                "match" => {
                    // {"match": {"value": x}} => ["field_name", "Eq", x]
                    if let Some(val) = operand.get("value") {
                        if let Some(arr) = val.as_array() {
                            // Multiple values => In
                            return Some(serde_json::json!([field_name, "In", arr]));
                        }
                        return Some(serde_json::json!([field_name, "Eq", val]));
                    }
                }
                "range" => {
                    let range_obj = operand.as_object()?;
                    let mut clauses = Vec::new();
                    if let Some(gt) = range_obj.get("gt") {
                        clauses.push(serde_json::json!([field_name, "Gt", gt]));
                    }
                    if let Some(gte) = range_obj.get("gte") {
                        clauses.push(serde_json::json!([field_name, "Gte", gte]));
                    }
                    if let Some(lt) = range_obj.get("lt") {
                        clauses.push(serde_json::json!([field_name, "Lt", lt]));
                    }
                    if let Some(lte) = range_obj.get("lte") {
                        clauses.push(serde_json::json!([field_name, "Lte", lte]));
                    }
                    if clauses.len() == 1 {
                        return Some(clauses.into_iter().next().unwrap());
                    } else if clauses.len() > 1 {
                        return Some(serde_json::json!(["And", clauses]));
                    }
                }
                "geo" => {
                    // Turbopuffer doesn't support geo filters natively; skip
                    return None;
                }
                _ => {}
            }
        }
    }
    None
}

impl Engine for TurbopufferEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        // Set distance metric based on dataset
        self.distance_metric =
            to_turbopuffer_metric(dataset.config.distance.as_deref().unwrap_or("cosine"))
                .to_string();

        println!(
            "Turbopuffer namespace: {}, metric: {}",
            self.namespace, self.distance_metric
        );

        // Delete existing namespace if it exists
        let ns = self.client.namespace(&self.namespace);
        let _ = self.rt.block_on(async { ns.delete().await });

        println!("Namespace configured (will be created on first upsert)");
        Ok(())
    }

    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        let dataset_path = dataset.get_path()?;
        let normalize = dataset.needs_normalization();
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
            self.parallel, self.batch_size
        );
        let upload_start = Instant::now();

        let total = vectors.len();
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta}) {msg}"
            )
            .unwrap()
            .with_key("eta", |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                write!(w, "{:.1}s", state.eta().as_secs_f64()).unwrap()
            })
            .progress_chars("#>-")
        );

        if self.parallel <= 1 {
            // Sequential upload
            for start in (0..total).step_by(self.batch_size) {
                let end = (start + self.batch_size).min(total);
                let batch_ids = &ids[start..end];
                let batch_vecs = &vectors[start..end];
                let batch_meta = &metadata[start..end];

                Self::upload_batch(
                    &self.client,
                    &self.namespace,
                    batch_ids,
                    batch_vecs,
                    batch_meta,
                    &self.rt,
                )?;
                pb.set_position(end as u64);
            }
        } else {
            // Parallel upload
            let client = Arc::clone(&self.client);
            let namespace = self.namespace.clone();
            let batch_size = self.batch_size;
            let counter = Arc::new(AtomicUsize::new(0));
            let errors = Arc::new(Mutex::new(Vec::new()));

            let batches: Vec<UploadBatch> = (0..total)
                .step_by(batch_size)
                .map(|start| {
                    let end = (start + batch_size).min(total);
                    (
                        ids[start..end].to_vec(),
                        vectors[start..end].to_vec(),
                        metadata[start..end].to_vec(),
                    )
                })
                .collect();

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(self.parallel)
                .build()
                .map_err(|e| format!("Failed to create thread pool: {}", e))?;

            pool.scope(|s| {
                for (batch_ids, batch_vecs, batch_meta) in &batches {
                    let client = Arc::clone(&client);
                    let namespace = namespace.clone();
                    let counter = Arc::clone(&counter);
                    let errors = Arc::clone(&errors);
                    let pb = pb.clone();

                    s.spawn(move |_| {
                        let rt = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .unwrap();
                        if let Err(e) = Self::upload_batch(
                            &client, &namespace, batch_ids, batch_vecs, batch_meta, &rt,
                        ) {
                            errors.lock().unwrap().push(e);
                            return;
                        }
                        let done =
                            counter.fetch_add(batch_ids.len(), Ordering::Relaxed) + batch_ids.len();
                        pb.set_position(done as u64);
                    });
                }
            });

            let errs = errors.lock().unwrap();
            if !errs.is_empty() {
                return Err(format!("Upload errors: {}", errs.join("; ")));
            }
        }

        pb.finish_with_message("done");

        let upload_time = upload_start.elapsed().as_secs_f64();
        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            total as f64 / upload_time
        );

        let total_time = read_time + upload_time;
        println!("Total time: {:.3}s", total_time);

        Ok(UploadStats {
            upload_time,
            total_time,
            upload_count: total,
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

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<serde_json::Value>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_turbopuffer_filter))
            .collect();

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));

        println!(
            "\tRunning {} queries (top={}, parallel={})...",
            HumanCount(num_to_run as u64),
            top,
            parallel
        );

        // Push-based collection: only successful queries contribute latency +
        // quality samples, so a failure is never scored as a 0-recall / 0-latency
        // sample. Order is completion order (fine for aggregates).
        let latencies = Arc::new(Mutex::new(Vec::<f64>::new()));
        let precisions = Arc::new(Mutex::new(Vec::<f64>::new()));
        let recalls = Arc::new(Mutex::new(Vec::<f64>::new()));
        let mrrs = Arc::new(Mutex::new(Vec::<f64>::new()));
        let ndcgs = Arc::new(Mutex::new(Vec::<f64>::new()));

        let pb = self.create_progress_bar(num_to_run);
        let total_start = Instant::now();

        if parallel <= 1 {
            for i in 0..num_to_run {
                let query = &queries[i];
                let filter = parsed_filters[i].as_ref();
                let start = Instant::now();
                let results = single_query(
                    &self.client,
                    &self.namespace,
                    query,
                    top,
                    &self.distance_metric,
                    filter,
                    &self.rt,
                );
                let elapsed = start.elapsed().as_secs_f64();

                match results {
                    Ok(result_ids) => {
                        let m = crate::metrics::compute_metrics(&result_ids, &neighbors[i], top);
                        latencies.lock().unwrap().push(elapsed);
                        precisions.lock().unwrap().push(m.precision);
                        recalls.lock().unwrap().push(m.recall);
                        mrrs.lock().unwrap().push(m.mrr);
                        ndcgs.lock().unwrap().push(m.ndcg);
                    }
                    Err(e) => {
                        eprintln!("Search query {} failed: {}", i, e);
                    }
                }
                pb.inc(1);
            }
        } else {
            let client = Arc::clone(&self.client);
            let namespace = self.namespace.clone();
            let distance_metric = self.distance_metric.clone();

            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(parallel)
                .build()
                .map_err(|e| format!("Failed to create search thread pool: {}", e))?;

            pool.scope(|s| {
                for i in 0..num_to_run {
                    let client = Arc::clone(&client);
                    let namespace = namespace.clone();
                    let distance_metric = distance_metric.clone();
                    let query = queries[i].clone();
                    let filter = parsed_filters[i].clone();
                    let neighbor = neighbors[i].clone();
                    let latencies = Arc::clone(&latencies);
                    let precisions = Arc::clone(&precisions);
                    let recalls = Arc::clone(&recalls);
                    let mrrs = Arc::clone(&mrrs);
                    let ndcgs = Arc::clone(&ndcgs);
                    let pb = &pb;

                    s.spawn(move |_| {
                        let rt = tokio::runtime::Builder::new_current_thread()
                            .enable_all()
                            .build()
                            .unwrap();
                        let start = Instant::now();
                        let results = single_query(
                            &client,
                            &namespace,
                            &query,
                            top,
                            &distance_metric,
                            filter.as_ref(),
                            &rt,
                        );
                        let elapsed = start.elapsed().as_secs_f64();

                        match results {
                            Ok(result_ids) => {
                                let m =
                                    crate::metrics::compute_metrics(&result_ids, &neighbor, top);
                                latencies.lock().unwrap().push(elapsed);
                                precisions.lock().unwrap().push(m.precision);
                                recalls.lock().unwrap().push(m.recall);
                                mrrs.lock().unwrap().push(m.mrr);
                                ndcgs.lock().unwrap().push(m.ndcg);
                            }
                            Err(e) => {
                                eprintln!("Search query {} failed: {}", i, e);
                            }
                        }
                        pb.inc(1);
                    });
                }
            });
        }

        pb.finish_and_clear();
        let total_time = total_start.elapsed().as_secs_f64();

        let latencies = latencies.lock().unwrap().clone();
        let precisions = precisions.lock().unwrap().clone();
        let recalls = recalls.lock().unwrap().clone();
        let mrrs = mrrs.lock().unwrap().clone();
        let ndcgs = ndcgs.lock().unwrap().clone();

        // Aggregate over successful queries only.
        let succeeded = latencies.len();
        if succeeded == 0 {
            return Err("No searches completed (all queries failed)".to_string());
        }
        let failed = num_to_run - succeeded;
        if failed > 0 {
            eprintln!("WARNING: {} of {} queries failed", failed, num_to_run);
        }

        let mean_time = latencies.iter().sum::<f64>() / succeeded as f64;
        let mean_precision = precisions.iter().sum::<f64>() / succeeded as f64;
        let mean_recall = recalls.iter().sum::<f64>() / succeeded as f64;
        let mean_mrr = mrrs.iter().sum::<f64>() / succeeded as f64;
        let mean_ndcg = ndcgs.iter().sum::<f64>() / succeeded as f64;

        let variance = latencies
            .iter()
            .map(|t| (t - mean_time).powi(2))
            .sum::<f64>()
            / succeeded as f64;
        let std_time = variance.sqrt();

        let min_time = latencies.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_time = latencies.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let mut sorted_latencies = latencies.clone();
        sorted_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let pct = |q: f64| sorted_latencies[((succeeded as f64 * q) as usize).min(succeeded - 1)];
        let p50_time = pct(0.50);
        let p95_time = pct(0.95);
        let p99_time = pct(0.99);

        let rps = succeeded as f64 / total_time;

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
            precisions,
            recalls,
            mrrs,
            ndcgs,
            latencies,
            top,
            num_queries: num_to_run,
            parallel,
            ..Default::default()
        })
    }

    fn delete(&mut self) -> Result<(), String> {
        let ns = self.client.namespace(&self.namespace);
        self.rt
            .block_on(async { ns.delete().await })
            .map_err(|e| format!("Turbopuffer delete failed: {}", e))?;
        println!("Namespace '{}' deleted", self.namespace);
        Ok(())
    }
}

/// Execute a single query against Turbopuffer
fn single_query(
    client: &Client,
    namespace: &str,
    query: &[f32],
    top_k: usize,
    distance_metric: &str,
    filter: Option<&serde_json::Value>,
    rt: &tokio::runtime::Runtime,
) -> Result<Vec<i64>, String> {
    let query_vec: Vec<f64> = query.iter().map(|f| *f as f64).collect();

    let mut body = serde_json::json!({
        "vector": query_vec,
        "distance_metric": distance_metric,
        "top_k": top_k,
    });

    if let Some(f) = filter {
        body["filters"] = f.clone();
    }

    let ns = client.namespace(namespace);
    let response = rt
        .block_on(async { ns.query(&body).await })
        .map_err(|e| format!("Turbopuffer query failed: {}", e))?;

    let ids: Vec<i64> = response
        .vectors
        .iter()
        .map(|v| match &v.id {
            turbopuffer_client::response::Id::Int(i) => *i as i64,
            turbopuffer_client::response::Id::String(s) => s.parse::<i64>().unwrap_or(-1),
        })
        .collect();

    Ok(ids)
}
