//! Vertex AI Vector Search engine (Google Cloud), pure KNN.
//!
//! Cloud-only, like Turbopuffer — there is no local server. Talks to the
//! Vertex AI REST API (`{region}-aiplatform.googleapis.com`) with a bearer
//! token, using `reqwest::blocking`.
//!
//! Lifecycle (issue: "starting simple by pure KNN"):
//! - `configure`: create a STREAM_UPDATE Index (tree-AH), an IndexEndpoint with
//!   a public endpoint, and deploy the index. Deploying is SLOW (tens of
//!   minutes) — set `VERTEX_DEPLOY_TIMEOUT_SECS` accordingly. To skip the slow
//!   create+deploy, point the engine at an already-deployed index by setting
//!   `VERTEX_INDEX`, `VERTEX_INDEX_ENDPOINT`, and `VERTEX_DEPLOYED_INDEX_ID`.
//! - `upload`: `upsertDatapoints` in batches (streaming index).
//! - `search`: `findNeighbors` against the public endpoint, one persistent
//!   worker per `parallel`, timing only the RPC + reply parse.
//!
//! No metadata filters, no mixed workload, no quantization — pure vector KNN.
//!
//! Auth: `VERTEX_ACCESS_TOKEN` if set, otherwise `gcloud auth
//! print-access-token`. Tokens are short-lived; the token is re-fetched at the
//! start of each phase (and once before the timed search region).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};

const DEFAULT_REGION: &str = "us-central1";
const DEFAULT_MACHINE_TYPE: &str = "e2-standard-16";
const DEFAULT_DISPLAY_NAME: &str = "vdb_benchmark";
const DEFAULT_APPROX_NEIGHBORS: i64 = 150;
const DEFAULT_LEAF_EMBEDDING_COUNT: i64 = 500;
const DEFAULT_LEAF_SEARCH_PERCENT: i64 = 7;

/// The three env vars that point the engine at an ALREADY-deployed index, in
/// which case `configure` skips create+deploy and `delete` skips teardown (so
/// the caller-owned resources are left in place). Reuse requires all three.
fn reuse_index_ids() -> Option<(String, String, String)> {
    Some((
        std::env::var("VERTEX_INDEX").ok()?,
        std::env::var("VERTEX_INDEX_ENDPOINT").ok()?,
        std::env::var("VERTEX_DEPLOYED_INDEX_ID").ok()?,
    ))
}

/// Access tokens (GCP) live ~60 min; refresh a phase-long token before it can
/// expire mid-flight.
const TOKEN_REFRESH_AFTER: Duration = Duration::from_secs(45 * 60);

/// Map a dataset distance string to a Vertex `distanceMeasureType`.
fn vertex_distance_measure(distance: &str) -> &'static str {
    match distance {
        "cosine" | "angular" => "COSINE_DISTANCE",
        "l2" | "euclidean" => "SQUARED_L2_DISTANCE",
        "dot" | "ip" => "DOT_PRODUCT_DISTANCE",
        _ => "DOT_PRODUCT_DISTANCE",
    }
}

/// Body for `indexes.create` — a STREAM_UPDATE tree-AH index. `shard_size`
/// (e.g. `SHARD_SIZE_SMALL`) constrains which deploy machine types are valid:
/// the default `SHARD_SIZE_MEDIUM` requires `e2-standard-16`+, while
/// `SHARD_SIZE_SMALL` allows smaller machines. Omitted when `None`.
fn build_index_body(
    display_name: &str,
    dimensions: i64,
    distance_measure: &str,
    approx_neighbors: i64,
    leaf_embedding_count: i64,
    leaf_search_percent: i64,
    shard_size: Option<&str>,
) -> serde_json::Value {
    let mut config = serde_json::json!({
        "dimensions": dimensions,
        "approximateNeighborsCount": approx_neighbors,
        "distanceMeasureType": distance_measure,
        "algorithmConfig": {
            "treeAhConfig": {
                "leafNodeEmbeddingCount": leaf_embedding_count.to_string(),
                "leafNodesToSearchPercent": leaf_search_percent,
            }
        }
    });
    if let Some(shard) = shard_size {
        config["shardSize"] = serde_json::json!(shard);
    }
    serde_json::json!({
        "displayName": display_name,
        "indexUpdateMethod": "STREAM_UPDATE",
        "metadata": { "config": config },
    })
}

/// Body for `indexEndpoints.deployIndex`.
fn build_deploy_body(
    deployed_index_id: &str,
    index_name: &str,
    machine_type: &str,
) -> serde_json::Value {
    serde_json::json!({
        "deployedIndex": {
            "id": deployed_index_id,
            "index": index_name,
            "dedicatedResources": {
                "machineSpec": { "machineType": machine_type },
                "minReplicaCount": 1,
                "maxReplicaCount": 1,
            }
        }
    })
}

/// Body for `indexes.upsertDatapoints` — ids are stringified row indices.
fn build_upsert_body(ids: &[i64], vectors: &[Vec<f32>]) -> serde_json::Value {
    let datapoints: Vec<serde_json::Value> = ids
        .iter()
        .zip(vectors.iter())
        .map(|(id, v)| {
            serde_json::json!({
                "datapointId": id.to_string(),
                "featureVector": v,
            })
        })
        .collect();
    serde_json::json!({ "datapoints": datapoints })
}

/// Body for `indexEndpoints.findNeighbors` (single query). An optional
/// `fraction_leaf_nodes_to_search_override` (0..1) trades recall for latency.
fn build_find_neighbors_body(
    deployed_index_id: &str,
    query: &[f32],
    top: usize,
    fraction_leaf_override: Option<f64>,
) -> serde_json::Value {
    let mut datapoint = serde_json::json!({ "datapoint": { "featureVector": query } });
    if let Some(frac) = fraction_leaf_override {
        datapoint["fractionLeafNodesToSearchOverride"] = serde_json::json!(frac);
    }
    datapoint["neighborCount"] = serde_json::json!(top);
    serde_json::json!({
        "deployedIndexId": deployed_index_id,
        "queries": [datapoint],
        "returnFullDatapoint": false,
    })
}

/// Parse the neighbor ids of the FIRST query from a `findNeighbors` reply.
/// Datapoint ids are stringified integers; unparseable ids are skipped.
fn parse_find_neighbors_response(resp: &serde_json::Value) -> Vec<i64> {
    resp.get("nearestNeighbors")
        .and_then(|nn| nn.as_array())
        .and_then(|arr| arr.first())
        .and_then(|first| first.get("neighbors"))
        .and_then(|n| n.as_array())
        .map(|neighbors| {
            neighbors
                .iter()
                .filter_map(|nb| {
                    nb.get("datapoint")
                        .and_then(|dp| dp.get("datapointId"))
                        .and_then(|id| id.as_str())
                        .and_then(|s| s.parse::<i64>().ok())
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Inspect a long-running-operation reply: `Ok(Some(response))` when done,
/// `Ok(None)` while still running, `Err` when the operation reported an error.
fn parse_lro(resp: &serde_json::Value) -> Result<Option<serde_json::Value>, String> {
    if let Some(err) = resp.get("error") {
        if !err.is_null() {
            return Err(format!("operation failed: {}", err));
        }
    }
    if resp.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
        // A done op with no `response` (e.g. delete) still counts as complete.
        return Ok(Some(
            resp.get("response")
                .cloned()
                .unwrap_or(serde_json::json!({})),
        ));
    }
    Ok(None)
}

pub struct VertexEngine {
    name: String,
    project: String,
    region: String,
    machine_type: String,
    display_name: String,
    batch_size: usize,
    parallel: usize,
    search_params: Vec<SearchParams>,
    distance_measure: String,
    deploy_timeout: Duration,
    approx_neighbors: i64,
    leaf_embedding_count: i64,
    leaf_search_percent: i64,
    shard_size: Option<String>,
    // Populated during configure().
    index_name: String,
    index_endpoint_name: String,
    deployed_index_id: String,
    public_endpoint_domain: String,
    client: reqwest::blocking::Client,
}

impl VertexEngine {
    pub fn new(engine_config: &EngineConfig, _host: &str) -> Result<Self, String> {
        let project = std::env::var("VERTEX_PROJECT").map_err(|_| {
            "VERTEX_PROJECT environment variable is required for the vertex engine".to_string()
        })?;
        let region = std::env::var("VERTEX_REGION").unwrap_or_else(|_| DEFAULT_REGION.to_string());
        let machine_type = std::env::var("VERTEX_MACHINE_TYPE")
            .unwrap_or_else(|_| DEFAULT_MACHINE_TYPE.to_string());
        let display_name = std::env::var("VERTEX_INDEX_DISPLAY_NAME")
            .unwrap_or_else(|_| DEFAULT_DISPLAY_NAME.to_string());

        let deploy_timeout = Duration::from_secs(
            std::env::var("VERTEX_DEPLOY_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(3600),
        );

        let env_i64 = |k: &str, default: i64| -> i64 {
            std::env::var(k)
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(default)
        };

        let parallel = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("parallel"))
            .and_then(|v| v.as_i64())
            .unwrap_or(4) as usize;
        let batch_size = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("batch_size"))
            .and_then(|v| v.as_i64())
            .unwrap_or(1000) as usize;

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| format!("Failed to build HTTP client: {}", e))?;

        Ok(Self {
            name: engine_config.name.clone(),
            project,
            region,
            machine_type,
            display_name,
            batch_size,
            parallel,
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            distance_measure: String::new(),
            deploy_timeout,
            approx_neighbors: env_i64("VERTEX_APPROX_NEIGHBORS", DEFAULT_APPROX_NEIGHBORS),
            leaf_embedding_count: env_i64(
                "VERTEX_LEAF_EMBEDDING_COUNT",
                DEFAULT_LEAF_EMBEDDING_COUNT,
            ),
            leaf_search_percent: env_i64("VERTEX_LEAF_SEARCH_PERCENT", DEFAULT_LEAF_SEARCH_PERCENT),
            shard_size: std::env::var("VERTEX_SHARD_SIZE")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            index_name: String::new(),
            index_endpoint_name: String::new(),
            deployed_index_id: String::new(),
            public_endpoint_domain: String::new(),
            client,
        })
    }

    fn base_url(&self) -> String {
        format!("https://{}-aiplatform.googleapis.com/v1", self.region)
    }

    fn parent(&self) -> String {
        format!("projects/{}/locations/{}", self.project, self.region)
    }

    /// A fresh bearer token: `VERTEX_ACCESS_TOKEN` if set, else `gcloud auth
    /// print-access-token`. Re-fetched per phase so a long deploy doesn't run on
    /// an expired token.
    fn access_token(&self) -> Result<String, String> {
        if let Ok(t) = std::env::var("VERTEX_ACCESS_TOKEN") {
            let t = t.trim().to_string();
            if !t.is_empty() {
                return Ok(t);
            }
        }
        let out = std::process::Command::new("gcloud")
            .args(["auth", "print-access-token"])
            .output()
            .map_err(|e| {
                format!("VERTEX_ACCESS_TOKEN unset and running `gcloud auth print-access-token` failed: {}", e)
            })?;
        if !out.status.success() {
            return Err(format!(
                "`gcloud auth print-access-token` failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
    }

    /// POST JSON to `url`, returning the parsed reply. Non-2xx is an error with
    /// the response body (Vertex returns a helpful `error.message`).
    fn post_json(
        &self,
        url: &str,
        token: &str,
        body: &serde_json::Value,
    ) -> Result<serde_json::Value, String> {
        let resp = self
            .client
            .post(url)
            .bearer_auth(token)
            .json(body)
            .send()
            .map_err(|e| format!("POST {} failed: {}", url, e))?;
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            return Err(format!("POST {} -> {}: {}", url, status, text));
        }
        serde_json::from_str(&text).map_err(|e| format!("invalid JSON from {}: {}", url, e))
    }

    fn get_json(&self, url: &str, token: &str) -> Result<serde_json::Value, String> {
        let resp = self
            .client
            .get(url)
            .bearer_auth(token)
            .send()
            .map_err(|e| format!("GET {} failed: {}", url, e))?;
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            return Err(format!("GET {} -> {}: {}", url, status, text));
        }
        serde_json::from_str(&text).map_err(|e| format!("invalid JSON from {}: {}", url, e))
    }

    /// Poll a long-running operation until done, returning its `response`.
    fn poll_operation(
        &self,
        operation_name: &str,
        timeout: Duration,
        label: &str,
    ) -> Result<serde_json::Value, String> {
        let url = format!("{}/{}", self.base_url(), operation_name);
        let start = Instant::now();
        let mut logged = false;
        loop {
            // Re-fetch the token each poll so a multi-minute wait can't expire it.
            let token = self.access_token()?;
            let resp = self.get_json(&url, &token)?;
            match parse_lro(&resp)? {
                Some(response) => return Ok(response),
                None => {
                    if start.elapsed() > timeout {
                        return Err(format!(
                            "{} did not finish within {}s",
                            label,
                            timeout.as_secs()
                        ));
                    }
                    if !logged {
                        println!(
                            "\tWaiting for {} (polling, timeout {}s)...",
                            label,
                            timeout.as_secs()
                        );
                        logged = true;
                    }
                    std::thread::sleep(Duration::from_secs(15));
                }
            }
        }
    }
}

impl Engine for VertexEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        self.distance_measure = vertex_distance_measure(dataset.distance()).to_string();
        let token = self.access_token()?;

        // Reuse path: skip the slow create+deploy when the caller points at an
        // already-deployed index.
        if let Some((index, endpoint, deployed)) = reuse_index_ids() {
            self.index_name = index;
            self.index_endpoint_name = endpoint;
            self.deployed_index_id = deployed;
            println!("Reusing deployed Vertex index {}", self.index_name);
        } else {
            // 1. Create the index.
            println!(
                "Creating Vertex index (dim={}, {})...",
                dataset.vector_size(),
                self.distance_measure
            );
            let index_body = build_index_body(
                &self.display_name,
                dataset.vector_size(),
                &self.distance_measure,
                self.approx_neighbors,
                self.leaf_embedding_count,
                self.leaf_search_percent,
                self.shard_size.as_deref(),
            );
            let op = self.post_json(
                &format!("{}/{}/indexes", self.base_url(), self.parent()),
                &token,
                &index_body,
            )?;
            let op_name = op
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or("index create returned no operation name")?
                .to_string();
            let created = self.poll_operation(&op_name, self.deploy_timeout, "index creation")?;
            self.index_name = created
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or("index create response missing name")?
                .to_string();

            // 2. Create the public index endpoint.
            println!("Creating Vertex index endpoint...");
            let ep_body = serde_json::json!({
                "displayName": self.display_name,
                "publicEndpointEnabled": true,
            });
            let op = self.post_json(
                &format!("{}/{}/indexEndpoints", self.base_url(), self.parent()),
                &token,
                &ep_body,
            )?;
            let op_name = op
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or("endpoint create returned no operation name")?
                .to_string();
            let created =
                self.poll_operation(&op_name, self.deploy_timeout, "endpoint creation")?;
            self.index_endpoint_name = created
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or("endpoint create response missing name")?
                .to_string();

            // 3. Deploy the index (SLOW).
            self.deployed_index_id = format!("{}_deployed", self.display_name);
            println!(
                "Deploying index to endpoint (id={}, machine={})...",
                self.deployed_index_id, self.machine_type
            );
            let deploy_body = build_deploy_body(
                &self.deployed_index_id,
                &self.index_name,
                &self.machine_type,
            );
            let op = self.post_json(
                &format!(
                    "{}/{}:deployIndex",
                    self.base_url(),
                    self.index_endpoint_name
                ),
                &token,
                &deploy_body,
            )?;
            let op_name = op
                .get("name")
                .and_then(|n| n.as_str())
                .ok_or("deployIndex returned no operation name")?
                .to_string();
            self.poll_operation(&op_name, self.deploy_timeout, "index deployment")?;
        }

        // Resolve the public endpoint domain used by findNeighbors.
        let token = self.access_token()?;
        let ep = self.get_json(
            &format!("{}/{}", self.base_url(), self.index_endpoint_name),
            &token,
        )?;
        self.public_endpoint_domain = ep
            .get("publicEndpointDomainName")
            .and_then(|d| d.as_str())
            .ok_or(
                "index endpoint has no publicEndpointDomainName (is a public endpoint enabled?)",
            )?
            .to_string();
        println!("Vertex endpoint ready at {}", self.public_endpoint_domain);
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
            "Read {} vectors ({}d) in {:.3}s",
            vectors.len(),
            vectors.first().map(|v| v.len()).unwrap_or(0),
            read_time
        );

        let url = format!("{}/{}:upsertDatapoints", self.base_url(), self.index_name);
        let batch_size = self.batch_size.max(1);
        let batches: Vec<(usize, usize)> = (0..ids.len())
            .step_by(batch_size)
            .map(|s| (s, (s + batch_size).min(ids.len())))
            .collect();
        // Honest parallelism: never spin up more workers than batches.
        let workers = self.parallel.max(1).min(batches.len().max(1));

        let pb = self.create_progress_bar(ids.len());
        let upload_start = Instant::now();

        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let engine = &*self;
        let ids = &ids;
        let vectors = &vectors;
        let batches = &batches;
        let url = url.as_str();

        std::thread::scope(|s| {
            for _ in 0..workers {
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;
                s.spawn(move || {
                    let client = match reqwest::blocking::Client::builder()
                        .timeout(Duration::from_secs(300))
                        .build()
                    {
                        Ok(c) => c,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e.to_string());
                            return;
                        }
                    };
                    // Each worker holds its own token and refreshes it before the
                    // ~60-min GCP token lifetime elapses, so a long streaming
                    // upload can't die on a 401 mid-flight.
                    let (mut token, mut token_at) = match engine.access_token() {
                        Ok(t) => (t, Instant::now()),
                        Err(e) => {
                            *error.lock().unwrap() = Some(e);
                            return;
                        }
                    };
                    loop {
                        let idx = batch_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= batches.len() || error.lock().unwrap().is_some() {
                            break;
                        }
                        if token_at.elapsed() > TOKEN_REFRESH_AFTER {
                            match engine.access_token() {
                                Ok(t) => {
                                    token = t;
                                    token_at = Instant::now();
                                }
                                Err(e) => {
                                    *error.lock().unwrap() = Some(e);
                                    break;
                                }
                            }
                        }
                        let (start, end) = batches[idx];
                        let body = build_upsert_body(&ids[start..end], &vectors[start..end]);
                        match client.post(url).bearer_auth(&token).json(&body).send() {
                            Ok(r) if r.status().is_success() => {}
                            Ok(r) => {
                                *error.lock().unwrap() = Some(format!(
                                    "{}: {}",
                                    r.status(),
                                    r.text().unwrap_or_default()
                                ));
                                break;
                            }
                            Err(e) => {
                                *error.lock().unwrap() = Some(e.to_string());
                                break;
                            }
                        }
                        pb.inc((end - start) as u64);
                    }
                });
            }
        });
        pb.finish_and_clear();

        if let Some(e) = error.lock().unwrap().take() {
            return Err(format!("upsertDatapoints failed: {}", e));
        }

        let upload_time = upload_start.elapsed().as_secs_f64();
        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            vectors.len() as f64 / upload_time
        );

        Ok(UploadStats {
            upload_time,
            total_time: read_time + upload_time,
            upload_count: vectors.len(),
            parallel: workers,
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
        let parallel = params.parallel.unwrap_or(1).max(1) as usize;

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, _conditions) = dataset.read_queries()?;

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };
        let top = explicit_top.unwrap_or_else(|| neighbors.first().map(|n| n.len()).unwrap_or(10));

        if neighbors.len() < num_to_run {
            return Err(format!(
                "dataset misaligned: {} neighbor lists for {} queries to run",
                neighbors.len(),
                num_to_run
            ));
        }

        // Optional per-query recall/latency knob (0..1 fraction of leaf nodes).
        let fraction_leaf_override: Option<f64> = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.extra.as_ref())
            .and_then(|e| e.get("fraction_leaf_nodes_to_search_override"))
            .and_then(|v| v.as_f64());

        println!(
            "\tRunning {} queries (top={}, parallel={})...",
            HumanCount(num_to_run as u64),
            top,
            parallel
        );

        // One access token for the whole timed region (avoids a gcloud shell-out
        // in the hot loop). Each worker builds its own blocking client so no
        // connection pool is shared across threads.
        let token = self.access_token()?;
        let url = format!(
            "https://{}/v1/{}:findNeighbors",
            self.public_endpoint_domain, self.index_endpoint_name
        );
        let deployed_index_id = self.deployed_index_id.as_str();

        let workers = parallel.min(num_to_run.max(1));
        let query_idx = Arc::new(AtomicUsize::new(0));
        let pb = self.create_progress_bar(num_to_run);
        let total_start = Instant::now();

        let queries = &queries;
        let neighbors = &neighbors;
        let url = url.as_str();
        let token = token.as_str();

        let mut latencies: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut precisions: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut recalls: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut mrrs: Vec<f64> = Vec::with_capacity(num_to_run);
        let mut ndcgs: Vec<f64> = Vec::with_capacity(num_to_run);

        std::thread::scope(|s| {
            let mut handles = Vec::with_capacity(workers);
            for _ in 0..workers {
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;
                handles.push(s.spawn(move || {
                    let mut t = Vec::new();
                    let mut p = Vec::new();
                    let mut r = Vec::new();
                    let mut mr = Vec::new();
                    let mut nd = Vec::new();

                    let client = match reqwest::blocking::Client::builder()
                        .timeout(Duration::from_secs(60))
                        .build()
                    {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("Vertex worker client build failed: {}", e);
                            return (t, p, r, mr, nd);
                        }
                    };

                    loop {
                        let idx = query_idx.fetch_add(1, Ordering::Relaxed);
                        if idx >= num_to_run {
                            break;
                        }
                        let body = build_find_neighbors_body(
                            deployed_index_id,
                            &queries[idx],
                            top,
                            fraction_leaf_override,
                        );

                        // Timed window: RPC round-trip + reply parse only. The
                        // request body is built above (client-side work), matching
                        // the other engines' boundary.
                        let start = Instant::now();
                        let outcome: Result<Vec<i64>, String> = (|| {
                            let resp = client
                                .post(url)
                                .bearer_auth(token)
                                .json(&body)
                                .send()
                                .map_err(|e| e.to_string())?;
                            if !resp.status().is_success() {
                                return Err(format!(
                                    "{}: {}",
                                    resp.status(),
                                    resp.text().unwrap_or_default()
                                ));
                            }
                            let json: serde_json::Value = resp.json().map_err(|e| e.to_string())?;
                            Ok(parse_find_neighbors_response(&json))
                        })();
                        let elapsed = start.elapsed().as_secs_f64();

                        match outcome {
                            Ok(result_ids) => {
                                let m = crate::metrics::compute_metrics(
                                    &result_ids,
                                    &neighbors[idx],
                                    top,
                                );
                                t.push(elapsed);
                                p.push(m.precision);
                                r.push(m.recall);
                                mr.push(m.mrr);
                                nd.push(m.ndcg);
                            }
                            Err(e) => eprintln!("Search query {} failed: {}", idx, e),
                        }
                        pb.inc(1);
                    }
                    (t, p, r, mr, nd)
                }));
            }
            for h in handles {
                let (t, p, r, mr, nd) = h.join().unwrap();
                latencies.extend(t);
                precisions.extend(p);
                recalls.extend(r);
                mrrs.extend(mr);
                ndcgs.extend(nd);
            }
        });

        pb.finish_and_clear();
        let total_time = total_start.elapsed().as_secs_f64();

        let succeeded = latencies.len();
        if succeeded == 0 {
            return Err("No searches completed (all queries failed)".to_string());
        }
        let failed = num_to_run - succeeded;
        if failed > 0 {
            eprintln!("WARNING: {} of {} queries failed", failed, num_to_run);
        }

        crate::engine::compute_search_stats(
            &latencies,
            &precisions,
            &recalls,
            &mrrs,
            &ndcgs,
            total_time,
            top,
            parallel,
            num_to_run,
        )
    }

    fn delete(&mut self) -> Result<(), String> {
        // Only tear down resources this run created. Use the SAME three-var
        // condition as configure()'s reuse path, so a run that set only
        // VERTEX_INDEX (and therefore created a fresh endpoint+deployment) still
        // cleans those up instead of leaking billable resources.
        if reuse_index_ids().is_some() {
            return Ok(());
        }
        let token = match self.access_token() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Vertex delete: cannot get token: {}", e);
                return Ok(());
            }
        };
        if !self.index_endpoint_name.is_empty() && !self.deployed_index_id.is_empty() {
            let body = serde_json::json!({ "deployedIndexId": self.deployed_index_id });
            let url = format!(
                "{}/{}:undeployIndex",
                self.base_url(),
                self.index_endpoint_name
            );
            if let Ok(op) = self.post_json(&url, &token, &body) {
                if let Some(name) = op.get("name").and_then(|n| n.as_str()) {
                    let _ = self.poll_operation(name, self.deploy_timeout, "undeploy");
                }
            }
        }
        if !self.index_endpoint_name.is_empty() {
            let _ = self
                .client
                .delete(format!("{}/{}", self.base_url(), self.index_endpoint_name))
                .bearer_auth(&token)
                .send();
        }
        if !self.index_name.is_empty() {
            let _ = self
                .client
                .delete(format!("{}/{}", self.base_url(), self.index_name))
                .bearer_auth(&token)
                .send();
        }
        println!("Vertex resources deleted");
        Ok(())
    }
}

impl VertexEngine {
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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Vertex AI is cloud-only (needs a GCP project, a bearer token, and a
    // ~tens-of-minutes index deploy), so there is no live integration test.
    // These pins cover the pure request-body builders and reply parsers.

    #[test]
    fn distance_measure_mapping() {
        assert_eq!(vertex_distance_measure("cosine"), "COSINE_DISTANCE");
        assert_eq!(vertex_distance_measure("angular"), "COSINE_DISTANCE");
        assert_eq!(vertex_distance_measure("l2"), "SQUARED_L2_DISTANCE");
        assert_eq!(vertex_distance_measure("euclidean"), "SQUARED_L2_DISTANCE");
        assert_eq!(vertex_distance_measure("dot"), "DOT_PRODUCT_DISTANCE");
        assert_eq!(vertex_distance_measure("ip"), "DOT_PRODUCT_DISTANCE");
        assert_eq!(vertex_distance_measure("mystery"), "DOT_PRODUCT_DISTANCE");
    }

    #[test]
    fn index_body_is_stream_update_tree_ah() {
        let b = build_index_body("bench", 768, "COSINE_DISTANCE", 150, 500, 7, None);
        assert_eq!(b["indexUpdateMethod"], "STREAM_UPDATE");
        let cfg = &b["metadata"]["config"];
        assert_eq!(cfg["dimensions"], 768);
        assert_eq!(cfg["distanceMeasureType"], "COSINE_DISTANCE");
        assert_eq!(cfg["approximateNeighborsCount"], 150);
        // leafNodeEmbeddingCount is a STRING per the API.
        assert_eq!(
            cfg["algorithmConfig"]["treeAhConfig"]["leafNodeEmbeddingCount"],
            "500"
        );
        assert_eq!(
            cfg["algorithmConfig"]["treeAhConfig"]["leafNodesToSearchPercent"],
            7
        );
        // shardSize omitted when None, set when Some.
        assert!(cfg.get("shardSize").is_none());
        let b2 = build_index_body(
            "bench",
            8,
            "COSINE_DISTANCE",
            150,
            500,
            7,
            Some("SHARD_SIZE_SMALL"),
        );
        assert_eq!(b2["metadata"]["config"]["shardSize"], "SHARD_SIZE_SMALL");
    }

    #[test]
    fn upsert_body_stringifies_ids_and_keeps_vectors() {
        let b = build_upsert_body(&[0, 42], &[vec![1.0, 2.0], vec![3.0, 4.0]]);
        let dps = b["datapoints"].as_array().unwrap();
        assert_eq!(dps.len(), 2);
        assert_eq!(dps[0]["datapointId"], "0");
        assert_eq!(dps[1]["datapointId"], "42");
        assert_eq!(dps[1]["featureVector"], json!([3.0, 4.0]));
    }

    #[test]
    fn find_neighbors_body_sets_count_and_optional_override() {
        let b = build_find_neighbors_body("dep", &[1.0, 2.0], 10, None);
        assert_eq!(b["deployedIndexId"], "dep");
        assert_eq!(b["returnFullDatapoint"], false);
        let q = &b["queries"][0];
        assert_eq!(q["neighborCount"], 10);
        assert_eq!(q["datapoint"]["featureVector"], json!([1.0, 2.0]));
        assert!(q.get("fractionLeafNodesToSearchOverride").is_none());

        let b2 = build_find_neighbors_body("dep", &[1.0], 5, Some(0.2));
        assert_eq!(b2["queries"][0]["fractionLeafNodesToSearchOverride"], 0.2);
    }

    #[test]
    fn parse_neighbors_extracts_first_query_ids_in_order() {
        let resp = json!({
            "nearestNeighbors": [{
                "id": "q0",
                "neighbors": [
                    {"datapoint": {"datapointId": "7"}, "distance": 0.1},
                    {"datapoint": {"datapointId": "3"}, "distance": 0.2},
                    {"datapoint": {"datapointId": "not-an-int"}, "distance": 0.3}
                ]
            }]
        });
        assert_eq!(parse_find_neighbors_response(&resp), vec![7, 3]);
    }

    #[test]
    fn parse_neighbors_empty_on_missing_fields() {
        assert_eq!(parse_find_neighbors_response(&json!({})), Vec::<i64>::new());
        assert_eq!(
            parse_find_neighbors_response(&json!({"nearestNeighbors": []})),
            Vec::<i64>::new()
        );
    }

    #[test]
    fn lro_states() {
        // Pending.
        assert!(parse_lro(&json!({"name": "op", "done": false}))
            .unwrap()
            .is_none());
        // Done with a response.
        let done = parse_lro(&json!({"done": true, "response": {"name": "idx"}}))
            .unwrap()
            .unwrap();
        assert_eq!(done["name"], "idx");
        // Done with no response (e.g. delete) -> empty object, still complete.
        assert!(parse_lro(&json!({"done": true})).unwrap().is_some());
        // Error.
        assert!(parse_lro(&json!({"error": {"code": 3, "message": "bad"}})).is_err());
    }

    #[test]
    fn deploy_body_shape() {
        let b = build_deploy_body(
            "dep_id",
            "projects/p/locations/r/indexes/1",
            "e2-standard-16",
        );
        assert_eq!(b["deployedIndex"]["id"], "dep_id");
        assert_eq!(
            b["deployedIndex"]["index"],
            "projects/p/locations/r/indexes/1"
        );
        assert_eq!(
            b["deployedIndex"]["dedicatedResources"]["machineSpec"]["machineType"],
            "e2-standard-16"
        );
        assert_eq!(
            b["deployedIndex"]["dedicatedResources"]["minReplicaCount"],
            1
        );
    }
}
