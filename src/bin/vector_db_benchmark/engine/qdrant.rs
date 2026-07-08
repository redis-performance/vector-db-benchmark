//! Qdrant engine implementation.
//!
//! Uses the official `qdrant-client` crate with gRPC transport.
//! Wraps async calls with a tokio runtime (block_on) since the
//! benchmark Engine trait is synchronous.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use qdrant_client::qdrant::quantization_config::Quantization;
use qdrant_client::qdrant::vectors_config::Config;
use qdrant_client::qdrant::{
    BinaryQuantization, Condition, CreateCollectionBuilder, DatetimeRange, DeleteCollectionBuilder,
    Distance, FieldType, Filter, HnswConfigDiff, MaxOptimizationThreads, NamedVectors,
    OptimizersConfigDiff, PointStruct, PrefetchQueryBuilder, QuantizationSearchParams,
    QuantizationType, QueryPointsBuilder, ScalarQuantization, SearchParams as QdrantSearchParams,
    SparseVectorParamsBuilder, SparseVectorsConfigBuilder, Timestamp, Vector, VectorInput,
    VectorParamsBuilder, VectorsConfig,
};
use qdrant_client::{Payload, Qdrant};

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UploadStats};
use vector_db_benchmark::readers::metadata::MetadataItem;

const DEFAULT_COLLECTION: &str = "benchmark";

pub struct QdrantEngine {
    name: String,
    collection_name: String,
    #[allow(dead_code)]
    timeout: u64,
    batch_size: usize,
    parallel: usize,
    #[allow(dead_code)]
    grpc_url: String,
    /// REST base URL (e.g. http://host:6333) for the /metrics and /telemetry endpoints
    rest_url: String,
    api_key: Option<String>,
    search_params: Vec<SearchParams>,
    /// Raw collection_params JSON to pass through to Qdrant
    collection_params_extra: serde_json::Value,
    hnsw_m: Option<u64>,
    hnsw_ef_construct: Option<u64>,
    /// Tokio runtime for async operations
    rt: tokio::runtime::Runtime,
    /// Shared Qdrant client (wrapped in Arc for thread-safe sharing)
    client: Arc<Qdrant>,
}

impl QdrantEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let grpc_port: u16 = std::env::var("QDRANT_GRPC_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(6334);

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

        // Determine the host without scheme
        let clean_host = host
            .trim_start_matches("http://")
            .trim_start_matches("https://");

        let grpc_url = if let Ok(url) = std::env::var("QDRANT_URL") {
            url
        } else {
            format!("http://{}:{}", clean_host, grpc_port)
        };

        // REST endpoint (default port 6333) for /metrics and /telemetry. Overridable
        // via QDRANT_REST_URL, or QDRANT_REST_PORT for just the port.
        let rest_url = if let Ok(url) = std::env::var("QDRANT_REST_URL") {
            url.trim_end_matches('/').to_string()
        } else {
            let rest_port: u16 = std::env::var("QDRANT_REST_PORT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(6333);
            format!("http://{}:{}", clean_host, rest_port)
        };

        let collection_params_extra = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.extra.as_ref())
            .map(|e| serde_json::to_value(e).unwrap_or_default())
            .unwrap_or(serde_json::json!({}));

        let typed_hnsw = engine_config
            .collection_params
            .as_ref()
            .and_then(|cp| cp.hnsw_config.as_ref());
        let hnsw_m = typed_hnsw.and_then(|h| h.m).map(|v| v as u64);
        let hnsw_ef_construct = typed_hnsw.and_then(|h| h.ef_construction).map(|v| v as u64);

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Failed to create tokio runtime: {}", e))?;

        let client = rt
            .block_on(async {
                let mut builder =
                    Qdrant::from_url(&grpc_url).timeout(std::time::Duration::from_secs(timeout));
                if let Some(key) = &api_key {
                    builder = builder.api_key(key.clone());
                }
                builder.build()
            })
            .map_err(|e| format!("Failed to create Qdrant client: {}", e))?;

        Ok(Self {
            name: engine_config.name.clone(),
            collection_name,
            timeout,
            batch_size,
            parallel,
            grpc_url,
            rest_url,
            api_key,
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            collection_params_extra,
            hnsw_m,
            hnsw_ef_construct,
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

    fn delete_collection(&self) -> Result<(), String> {
        let _ = self.rt.block_on(
            self.client
                .delete_collection(DeleteCollectionBuilder::new(&self.collection_name)),
        );
        Ok(())
    }

    fn create_collection(&self, dataset: &Dataset) -> Result<(), String> {
        if dataset.is_sparse() {
            return self.create_sparse_collection(dataset);
        }

        let distance = dataset.distance();
        let vector_size = dataset.vector_size();

        let qdrant_distance = match distance.to_lowercase().as_str() {
            "l2" | "euclidean" => Distance::Euclid,
            "cosine" | "angular" => Distance::Cosine,
            "dot" | "ip" => Distance::Dot,
            other => return Err(format!("Unsupported distance metric for Qdrant: {}", other)),
        };

        // HNSW params come from the TYPED collection_params.hnsw_config field
        // (serde captures "m"/"ef_construct" there via aliases; the flattened
        // `extra` map never contains hnsw_config since it is a declared field).
        let hnsw_m = self.hnsw_m;
        let hnsw_ef = self.hnsw_ef_construct;

        // Optionally store vectors on disk (mmap) — collection_params.vectors_config.on_disk.
        let vectors_on_disk = self
            .collection_params_extra
            .get("vectors_config")
            .and_then(|v| v.get("on_disk"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let vector_params =
            VectorParamsBuilder::new(vector_size as u64, qdrant_distance).on_disk(vectors_on_disk);

        let mut create_builder = CreateCollectionBuilder::new(&self.collection_name)
            .vectors_config(VectorsConfig {
                config: Some(Config::Params(vector_params.build())),
            });

        // Apply HNSW config if specified
        if hnsw_m.is_some() || hnsw_ef.is_some() {
            let mut hnsw_config = HnswConfigDiff::default();
            if let Some(m) = hnsw_m {
                hnsw_config.m = Some(m);
            }
            if let Some(ef) = hnsw_ef {
                hnsw_config.ef_construct = Some(ef);
            }
            create_builder = create_builder.hnsw_config(hnsw_config);
        }

        // Pass through optimizers_config (e.g. the rps-tuned default_segment_number /
        // max_segment_size / memmap_threshold) — mirrors the python engine, which
        // forwards collection_params verbatim.
        if let Some(opt) = self.collection_params_extra.get("optimizers_config") {
            let mut diff = OptimizersConfigDiff::default();
            if let Some(v) = opt.get("default_segment_number").and_then(|v| v.as_u64()) {
                diff.default_segment_number = Some(v);
            }
            if let Some(v) = opt.get("max_segment_size").and_then(|v| v.as_u64()) {
                diff.max_segment_size = Some(v);
            }
            if let Some(v) = opt.get("memmap_threshold").and_then(|v| v.as_u64()) {
                diff.memmap_threshold = Some(v);
            }
            create_builder = create_builder.optimizers_config(diff);
        }

        // Pass through quantization_config (sq/bq tuned setups).
        if let Some(q) = self.collection_params_extra.get("quantization_config") {
            let quantization = if let Some(s) = q.get("scalar") {
                let qtype = match s.get("type").and_then(|v| v.as_str()) {
                    Some("int8") | None => QuantizationType::Int8,
                    Some(other) => {
                        return Err(format!("Unsupported scalar quantization type: {}", other))
                    }
                };
                Some(Quantization::Scalar(ScalarQuantization {
                    r#type: qtype.into(),
                    quantile: s.get("quantile").and_then(|v| v.as_f64()).map(|v| v as f32),
                    always_ram: s.get("always_ram").and_then(|v| v.as_bool()),
                }))
            } else {
                q.get("binary").map(|b| {
                    Quantization::Binary(BinaryQuantization {
                        always_ram: b.get("always_ram").and_then(|v| v.as_bool()),
                        ..Default::default()
                    })
                })
            };
            if let Some(quantization) = quantization {
                create_builder = create_builder.quantization_config(quantization);
            }
        }

        self.rt
            .block_on(self.client.create_collection(create_builder))
            .map_err(|e| format!("Failed to create collection: {}", e))?;

        // Disable optimization during indexing
        let _ = self.rt.block_on(
            self.client.update_collection(
                qdrant_client::qdrant::UpdateCollectionBuilder::new(&self.collection_name)
                    .optimizers_config(OptimizersConfigDiff {
                        max_optimization_threads: Some(MaxOptimizationThreads {
                            variant: Some(
                                qdrant_client::qdrant::max_optimization_threads::Variant::Value(0),
                            ),
                        }),
                        ..Default::default()
                    }),
            ),
        );

        self.create_payload_indexes(dataset);

        Ok(())
    }

    /// Create Qdrant payload indexes for the dataset's schema fields.
    fn create_payload_indexes(&self, dataset: &Dataset) {
        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                for (field_name, field_type) in schema_obj {
                    let ft = field_type.as_str().unwrap_or("");
                    let qdrant_type = match ft {
                        "int" => FieldType::Integer,
                        "keyword" => FieldType::Keyword,
                        "text" => FieldType::Text,
                        "float" => FieldType::Float,
                        "geo" => FieldType::Geo,
                        "uuid" => FieldType::Uuid,
                        "bool" => FieldType::Bool,
                        "datetime" => FieldType::Datetime,
                        _ => continue,
                    };
                    let _ = self.rt.block_on(self.client.create_field_index(
                        qdrant_client::qdrant::CreateFieldIndexCollectionBuilder::new(
                            &self.collection_name,
                            field_name.clone(),
                            qdrant_type,
                        ),
                    ));
                }
            }
        }
    }

    /// Create a sparse-vector collection with a single named "sparse" vector.
    fn create_sparse_collection(&self, dataset: &Dataset) -> Result<(), String> {
        let mut sparse_cfg = SparseVectorsConfigBuilder::default();
        sparse_cfg.add_named_vector_params("sparse", SparseVectorParamsBuilder::default());
        let create_builder =
            CreateCollectionBuilder::new(&self.collection_name).sparse_vectors_config(sparse_cfg);
        self.rt
            .block_on(self.client.create_collection(create_builder))
            .map_err(|e| format!("Failed to create sparse collection: {}", e))?;
        self.create_payload_indexes(dataset);
        Ok(())
    }

    /// Upload sparse vectors under the named "sparse" vector, batched.
    fn upload_sparse(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        let dataset_path = dataset.get_path()?;
        println!("Reading sparse dataset from {}...", dataset_path.display());
        let read_start = Instant::now();
        let (ids, vectors) = dataset.read_sparse_data()?;
        let read_time = read_start.elapsed().as_secs_f64();
        println!("Read {} sparse vectors in {:.3}s", vectors.len(), read_time);

        println!("Starting sparse upload, batch size {}...", self.batch_size);
        let upload_start = Instant::now();
        let pb = self.create_progress_bar(ids.len());
        for start in (0..ids.len()).step_by(self.batch_size) {
            let end = (start + self.batch_size).min(ids.len());
            let points: Vec<PointStruct> = (start..end)
                .map(|i| {
                    PointStruct::new(
                        ids[i] as u64,
                        NamedVectors::default().add_vector(
                            "sparse",
                            Vector::new_sparse(
                                vectors[i].indices.clone(),
                                vectors[i].values.clone(),
                            ),
                        ),
                        Payload::new(),
                    )
                })
                .collect();
            self.rt
                .block_on(
                    self.client.upsert_points(
                        qdrant_client::qdrant::UpsertPointsBuilder::new(
                            &self.collection_name,
                            points,
                        )
                        .wait(true),
                    ),
                )
                .map_err(|e| format!("Sparse upsert failed: {}", e))?;
            pb.inc((end - start) as u64);
        }
        pb.finish_with_message("Upload complete");
        let upload_time = upload_start.elapsed().as_secs_f64();
        self.wait_collection_green()?;

        Ok(UploadStats {
            upload_time,
            total_time: read_time + upload_time,
            upload_count: vectors.len(),
            parallel: 1,
            batch_size: self.batch_size,
            memory_usage: None,
        })
    }

    fn upload_parallel(
        &self,
        ids: &[i64],
        vectors: &[Vec<f32>],
        metadata: &[Option<MetadataItem>],
    ) -> Result<(), String> {
        use vector_db_benchmark::readers::metadata::MetadataValue;

        let pb = self.create_progress_bar(ids.len());
        let batches: Vec<(usize, usize)> = (0..ids.len())
            .step_by(self.batch_size)
            .map(|start| (start, (start + self.batch_size).min(ids.len())))
            .collect();

        let total_batches = batches.len();
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let client = Arc::clone(&self.client);
        let collection_name = self.collection_name.clone();

        std::thread::scope(|s| {
            for _ in 0..self.parallel {
                let client = Arc::clone(&client);
                let collection_name = collection_name.clone();
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let rt = match tokio::runtime::Runtime::new() {
                        Ok(rt) => rt,
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
                        let mut points = Vec::with_capacity(batch_end - batch_start);

                        for i in batch_start..batch_end {
                            let mut payload = Payload::new();
                            if let Some(meta) = &metadata[i] {
                                for (k, v) in &meta.fields {
                                    match v {
                                        MetadataValue::String(s) => {
                                            payload.insert(k.clone(), s.clone());
                                        }
                                        MetadataValue::Labels(labels) => {
                                            let arr: Vec<qdrant_client::qdrant::Value> =
                                                labels.iter().map(|l| l.clone().into()).collect();
                                            payload.insert(
                                                k.clone(),
                                                qdrant_client::qdrant::Value {
                                                    kind: Some(
                                                        qdrant_client::qdrant::value::Kind::ListValue(
                                                            qdrant_client::qdrant::ListValue {
                                                                values: arr,
                                                            },
                                                        ),
                                                    ),
                                                },
                                            );
                                        }
                                        MetadataValue::Geo { lon, lat } => {
                                            let mut geo_payload = Payload::new();
                                            geo_payload.insert("lon", *lon);
                                            geo_payload.insert("lat", *lat);
                                            payload.insert(
                                                k.clone(),
                                                qdrant_client::qdrant::Value::from(
                                                    serde_json::json!({"lon": lon, "lat": lat}),
                                                ),
                                            );
                                        }
                                    };
                                }
                            }

                            points.push(PointStruct::new(
                                ids[i] as u64,
                                vectors[i].clone(),
                                payload,
                            ));
                        }

                        let result = rt.block_on(client.upsert_points(
                            qdrant_client::qdrant::UpsertPointsBuilder::new(
                                &collection_name,
                                points,
                            )
                            .wait(false),
                        ));

                        if let Err(e) = result {
                            *error.lock().unwrap() = Some(format!("Upsert failed: {}", e));
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

    fn wait_collection_green(&self) -> Result<(), String> {
        println!("Waiting for collection to be GREEN...");

        // Re-enable optimization (auto mode)
        let _ = self.rt.block_on(
            self.client.update_collection(
                qdrant_client::qdrant::UpdateCollectionBuilder::new(&self.collection_name)
                    .optimizers_config(OptimizersConfigDiff {
                        max_optimization_threads: Some(MaxOptimizationThreads {
                            variant: Some(
                                qdrant_client::qdrant::max_optimization_threads::Variant::Setting(
                                    qdrant_client::qdrant::max_optimization_threads::Setting::Auto
                                        as i32,
                                ),
                            ),
                        }),
                        ..Default::default()
                    }),
            ),
        );

        // Index build can take hours on large datasets (e.g. deep-image-96 has
        // ~10M vectors); the old fixed 50-min cap (600 * 5s) was too short for
        // high-M / high-ef_construct configs and silently aborted the whole run.
        // Make the budget configurable (QDRANT_GREEN_WAIT_SECS) with a generous
        // default, and log indexing progress so a slow build is visible and
        // distinguishable from a genuinely stuck one.
        let green_wait_secs: u64 = std::env::var("QDRANT_GREEN_WAIT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(14400); // 4h
        let iterations = (green_wait_secs / 5).max(1);
        for i in 0..iterations {
            std::thread::sleep(std::time::Duration::from_secs(5));

            if let Ok(info) = self
                .rt
                .block_on(self.client.collection_info(&self.collection_name))
            {
                // status: 1 = Green, 2 = Yellow, 3 = Red (from protobuf enum)
                if let Some(result) = info.result {
                    // Progress heartbeat every ~60s so long builds aren't opaque.
                    if i % 12 == 11 {
                        println!(
                            "  ...waiting for GREEN: status={} indexed={}/{} ({}s / {}s budget)",
                            result.status,
                            result.indexed_vectors_count.unwrap_or(0),
                            result.points_count.unwrap_or(0),
                            (i + 1) * 5,
                            green_wait_secs
                        );
                    }
                    if result.status == 1 {
                        // Double-check
                        std::thread::sleep(std::time::Duration::from_secs(5));
                        if let Ok(info2) = self
                            .rt
                            .block_on(self.client.collection_info(&self.collection_name))
                        {
                            if let Some(result2) = info2.result {
                                if result2.status == 1 {
                                    println!("Collection is GREEN.");
                                    return Ok(());
                                }
                            }
                        }
                    }
                }
            }
        }
        Err(format!(
            "Timed out waiting for collection to reach GREEN status after {}s \
             (override with QDRANT_GREEN_WAIT_SECS)",
            green_wait_secs
        ))
    }
}

/// Parse conditions into Qdrant filter format.
fn parse_qdrant_conditions(conditions: &serde_json::Value) -> Option<Filter> {
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

    let mut filter = Filter::default();
    if let Some(m) = must {
        filter.must = m;
    }
    if let Some(s) = should {
        filter.should = s;
    }

    Some(filter)
}

fn build_qdrant_subfilters(entries: &[serde_json::Value]) -> Vec<Condition> {
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

/// Parse an ISO-8601 / RFC 3339 datetime string into a protobuf Timestamp.
fn parse_rfc3339_timestamp(s: &str) -> Option<Timestamp> {
    let dt = chrono::DateTime::parse_from_rfc3339(s).ok()?;
    Some(Timestamp {
        seconds: dt.timestamp(),
        nanos: dt.timestamp_subsec_nanos() as i32,
    })
}

fn build_qdrant_filter(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
) -> Option<Condition> {
    match condition_type {
        "match" => {
            let criteria_obj = criteria.as_object()?;
            // match_any: value in a list (keywords or integers).
            if let Some(any) = criteria_obj.get("any").and_then(|v| v.as_array()) {
                if !any.is_empty() && any.iter().all(|v| v.is_i64()) {
                    let vals: Vec<i64> = any.iter().filter_map(|v| v.as_i64()).collect();
                    return Some(Condition::matches(field_name.to_string(), vals));
                }
                let vals: Vec<String> = any
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                return Some(Condition::matches(field_name.to_string(), vals));
            }
            // match_text: full-text match.
            if let Some(text) = criteria_obj.get("text").and_then(|v| v.as_str()) {
                return Some(Condition::matches_text(
                    field_name.to_string(),
                    text.to_string(),
                ));
            }
            // exact match on keyword / integer / bool.
            let value = criteria_obj.get("value")?;
            if let Some(s) = value.as_str() {
                Some(Condition::matches(field_name.to_string(), s.to_string()))
            } else if let Some(b) = value.as_bool() {
                Some(Condition::matches(field_name.to_string(), b))
            } else {
                value
                    .as_i64()
                    .map(|n| Condition::matches(field_name.to_string(), n))
            }
        }
        "range" => {
            let criteria_obj = criteria.as_object()?;
            // A string bound means an ISO-8601 datetime range rather than numeric.
            let is_datetime = ["lt", "gt", "lte", "gte"]
                .iter()
                .any(|k| criteria_obj.get(*k).map(|v| v.is_string()).unwrap_or(false));
            if is_datetime {
                let ts = |k: &str| {
                    criteria_obj
                        .get(k)
                        .and_then(|v| v.as_str())
                        .and_then(parse_rfc3339_timestamp)
                };
                return Some(Condition::datetime_range(
                    field_name.to_string(),
                    DatetimeRange {
                        lt: ts("lt"),
                        gt: ts("gt"),
                        gte: ts("gte"),
                        lte: ts("lte"),
                    },
                ));
            }
            let mut range = qdrant_client::qdrant::Range::default();
            if let Some(lt) = criteria_obj.get("lt").and_then(|v| v.as_f64()) {
                range.lt = Some(lt);
            }
            if let Some(gt) = criteria_obj.get("gt").and_then(|v| v.as_f64()) {
                range.gt = Some(gt);
            }
            if let Some(lte) = criteria_obj.get("lte").and_then(|v| v.as_f64()) {
                range.lte = Some(lte);
            }
            if let Some(gte) = criteria_obj.get("gte").and_then(|v| v.as_f64()) {
                range.gte = Some(gte);
            }
            Some(Condition::range(field_name.to_string(), range))
        }
        "geo" => {
            let lat = criteria.get("lat")?.as_f64()?;
            let lon = criteria.get("lon")?.as_f64()?;
            let radius = criteria
                .get("radius")
                .and_then(|r| r.as_f64())
                .unwrap_or(1000.0);
            Some(Condition::geo_radius(
                field_name.to_string(),
                qdrant_client::qdrant::GeoRadius {
                    center: Some(qdrant_client::qdrant::GeoPoint { lon, lat }),
                    radius: radius as f32,
                },
            ))
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
        println!("Deleting existing collection...");
        self.delete_collection()?;

        println!("Creating collection '{}'...", self.collection_name);
        self.create_collection(dataset)?;
        println!("Collection '{}' created.", self.collection_name);

        Ok(())
    }

    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String> {
        if dataset.is_sparse() {
            return self.upload_sparse(dataset);
        }

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
        self.wait_collection_green()?;

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

        // Build Qdrant search params
        let hnsw_ef: Option<u64> = params.search_params.as_ref().and_then(|sp| {
            sp.ef.map(|e| e as u64).or_else(|| {
                sp.extra
                    .as_ref()
                    .and_then(|e| e.get("hnsw_ef"))
                    .and_then(|v| v.as_u64())
            })
        });

        // Search-time quantization params (rescore/oversampling) from the config's
        // search_params.quantization object — mirrors python rest.SearchParams(**params).
        let quantization_params: Option<QuantizationSearchParams> = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.extra.as_ref())
            .and_then(|e| e.get("quantization"))
            .map(|q| QuantizationSearchParams {
                rescore: q.get("rescore").and_then(|v| v.as_bool()),
                oversampling: q.get("oversampling").and_then(|v| v.as_f64()),
                ..Default::default()
            });

        // Prefetch (two-stage retrieval / rescoring): search_params.prefetch =
        // { "limit": N, "params": { "hnsw_ef": .., "quantization": {..} } }.
        // Mirrors python `models.Prefetch(**prefetch, query=query_vector)`.
        let prefetch = params
            .search_params
            .as_ref()
            .and_then(|sp| sp.extra.as_ref())
            .and_then(|e| e.get("prefetch"));
        let prefetch_enabled = prefetch.is_some();
        let prefetch_limit = prefetch
            .and_then(|p| p.get("limit"))
            .and_then(|v| v.as_u64());
        let prefetch_params = prefetch.and_then(|p| p.get("params"));
        let prefetch_hnsw_ef = prefetch_params
            .and_then(|p| p.get("hnsw_ef"))
            .and_then(|v| v.as_u64());
        let prefetch_quant: Option<QuantizationSearchParams> = prefetch_params
            .and_then(|p| p.get("quantization"))
            .map(|q| QuantizationSearchParams {
                rescore: q.get("rescore").and_then(|v| v.as_bool()),
                oversampling: q.get("oversampling").and_then(|v| v.as_f64()),
                ..Default::default()
            });

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());

        // Dense and sparse queries are read into separate vectors; only one is
        // populated. Filters/prefetch/quantization apply to the dense path only.
        let is_sparse = dataset.is_sparse();
        let (queries, sparse_queries, neighbors, parsed_filters) = if is_sparse {
            let (sq, nb) = dataset.read_sparse_queries()?;
            (Vec::<Vec<f32>>::new(), sq, nb, Vec::<Option<Filter>>::new())
        } else {
            let (q, nb, conditions) = dataset.read_queries()?;
            let pf: Vec<Option<Filter>> = conditions
                .iter()
                .map(|c| c.as_ref().and_then(parse_qdrant_conditions))
                .collect();
            (
                q,
                Vec::<vector_db_benchmark::readers::SparseVector>::new(),
                nb,
                pf,
            )
        };

        let query_count = if is_sparse {
            sparse_queries.len()
        } else {
            queries.len()
        };
        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(query_count)
        } else {
            query_count
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
        let client = Arc::clone(&self.client);
        let collection_name = self.collection_name.clone();

        std::thread::scope(|s| {
            for _ in 0..parallel {
                let client = Arc::clone(&client);
                let collection_name = collection_name.clone();
                let queries = &queries;
                let sparse_queries = &sparse_queries;
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
                    let rt = match tokio::runtime::Runtime::new() {
                        Ok(rt) => rt,
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

                        let mut query_builder = if is_sparse {
                            let sv = &sparse_queries[idx];
                            QueryPointsBuilder::new(collection_name.clone())
                                .query(VectorInput::new_sparse(
                                    sv.indices.clone(),
                                    sv.values.clone(),
                                ))
                                .using("sparse")
                                .limit(top as u64)
                                .with_payload(false)
                        } else {
                            let mut qb = QueryPointsBuilder::new(collection_name.clone())
                                .query(queries[idx].clone())
                                .limit(top as u64)
                                .with_payload(false);
                            if hnsw_ef.is_some() || quantization_params.is_some() {
                                qb = qb.params(QdrantSearchParams {
                                    hnsw_ef,
                                    quantization: quantization_params,
                                    ..Default::default()
                                });
                            }
                            if let Some(filter) = &parsed_filters[idx] {
                                qb = qb.filter(filter.clone());
                            }
                            qb
                        };

                        if !is_sparse && prefetch_enabled {
                            let mut pf =
                                PrefetchQueryBuilder::default().query(queries[idx].clone());
                            if let Some(l) = prefetch_limit {
                                pf = pf.limit(l);
                            }
                            if prefetch_hnsw_ef.is_some() || prefetch_quant.is_some() {
                                pf = pf.params(QdrantSearchParams {
                                    hnsw_ef: prefetch_hnsw_ef,
                                    quantization: prefetch_quant,
                                    ..Default::default()
                                });
                            }
                            query_builder = query_builder.prefetch(vec![pf.build()]);
                        }

                        let query_start = Instant::now();
                        let result = rt.block_on(client.query(query_builder));
                        let query_time = query_start.elapsed().as_secs_f64();

                        search_times.lock().unwrap().push(query_time);

                        if let Ok(response) = result {
                            let ordered_ids: Vec<i64> = response
                                .result
                                .iter()
                                .filter_map(|p| {
                                    if let Some(
                                        qdrant_client::qdrant::point_id::PointIdOptions::Num(n),
                                    ) =
                                        &p.id.as_ref().and_then(|id| id.point_id_options.as_ref())
                                    {
                                        Some(*n as i64)
                                    } else {
                                        None
                                    }
                                })
                                .collect();
                            let m =
                                crate::metrics::compute_metrics(&ordered_ids, &neighbors[idx], top);
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
        self.delete_collection()
    }

    /// Collect memory usage from Qdrant's REST observability endpoints, mirroring
    /// the Redis wrapper's `{used_memory, index_info}` shape.
    ///
    /// - `/metrics` (Prometheus): jemalloc `memory_*_bytes` gauges + collection counts.
    ///   `memory_resident_bytes` (RSS) is used as `used_memory`, the analog of
    ///   Redis' `used_memory`.
    /// - `/telemetry` (JSON): collection/cluster/segment state, the analog of FT.INFO.
    ///
    /// See https://qdrant.tech/documentation/cloud/cluster-monitoring/.
    fn get_memory_usage(&mut self) -> Option<serde_json::Value> {
        let http = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .ok()?;

        let get = |path: &str| -> Option<reqwest::blocking::Response> {
            let mut req = http.get(format!("{}{}", self.rest_url, path));
            if let Some(key) = &self.api_key {
                req = req.header("api-key", key);
            }
            req.send().ok().filter(|r| r.status().is_success())
        };

        // Prometheus /metrics → curated gauge map.
        let metrics = get("/metrics")
            .and_then(|r| r.text().ok())
            .map(|t| parse_qdrant_metrics(&t))
            .unwrap_or_default();
        let resident = metrics
            .get("memory_resident_bytes")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0) as i64;

        // Telemetry JSON → collection/cluster state.
        let telemetry: Option<serde_json::Value> = get("/telemetry?anonymize=true")
            .and_then(|r| r.json::<serde_json::Value>().ok())
            .and_then(|mut v| v.get_mut("result").map(|r| r.take()));

        // Per-collection info from the gRPC client (segments, vector counts, status).
        let collection_info = self
            .rt
            .block_on(self.client.collection_info(&self.collection_name))
            .ok()
            .map(|info| format!("{:?}", info.result));

        Some(serde_json::json!({
            "used_memory": [resident],
            "index_info": telemetry,
            "qdrant_metrics": metrics,
            "collection_info": collection_info,
        }))
    }
}

/// Parse the curated set of Qdrant `/metrics` (Prometheus text) gauges into a JSON
/// map. Only memory and collection-count gauges are kept; labeled/histogram lines
/// and comments are ignored.
fn parse_qdrant_metrics(text: &str) -> serde_json::Map<String, serde_json::Value> {
    const WANTED: &[&str] = &[
        "memory_active_bytes",
        "memory_allocated_bytes",
        "memory_metadata_bytes",
        "memory_resident_bytes",
        "memory_retained_bytes",
        "collections_total",
        "collections_vector_total",
    ];
    let mut out = serde_json::Map::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Split off the trailing value: "metric_name{labels} 123" or "metric_name 123".
        let mut it = line.rsplitn(2, char::is_whitespace);
        let value_str = match it.next() {
            Some(v) => v,
            None => continue,
        };
        let name_part = it.next().unwrap_or("").trim();
        let name = name_part.split('{').next().unwrap_or(name_part).trim();
        if WANTED.contains(&name) {
            if let Ok(v) = value_str.parse::<f64>() {
                out.insert(name.to_string(), serde_json::json!(v));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{build_qdrant_filter, parse_qdrant_metrics, parse_rfc3339_timestamp};
    use qdrant_client::qdrant::{condition::ConditionOneOf, FieldCondition};
    use serde_json::json;

    fn field_condition(c: &qdrant_client::qdrant::Condition) -> FieldCondition {
        match c.condition_one_of.clone().unwrap() {
            ConditionOneOf::Field(fc) => fc,
            other => panic!("expected FieldCondition, got {:?}", other),
        }
    }

    #[test]
    fn parses_rfc3339_timestamp() {
        let ts = parse_rfc3339_timestamp("1970-01-01T00:00:01Z").unwrap();
        assert_eq!(ts.seconds, 1);
        assert!(parse_rfc3339_timestamp("not-a-date").is_none());
    }

    #[test]
    fn builds_match_any_integers() {
        let c = build_qdrant_filter("cat", "match", &json!({"any": [1, 2, 3]})).unwrap();
        let fc = field_condition(&c);
        assert_eq!(fc.key, "cat");
        assert!(fc.r#match.is_some(), "match_any should set a Match");
    }

    #[test]
    fn builds_match_text() {
        let c = build_qdrant_filter("body", "match", &json!({"text": "hello"})).unwrap();
        let fc = field_condition(&c);
        assert_eq!(fc.key, "body");
        assert!(fc.r#match.is_some());
    }

    #[test]
    fn builds_bool_exact_match() {
        let c = build_qdrant_filter("flag", "match", &json!({"value": true})).unwrap();
        assert!(field_condition(&c).r#match.is_some());
    }

    #[test]
    fn range_with_string_bound_becomes_datetime_range() {
        let dt =
            build_qdrant_filter("ts", "range", &json!({"gte": "2023-01-01T00:00:00Z"})).unwrap();
        let fc = field_condition(&dt);
        assert!(fc.datetime_range.is_some(), "string bound → datetime_range");
        assert!(fc.range.is_none());

        let num = build_qdrant_filter("n", "range", &json!({"gte": 5, "lt": 10})).unwrap();
        let fc = field_condition(&num);
        assert!(fc.range.is_some(), "numeric bounds → numeric range");
        assert!(fc.datetime_range.is_none());
    }

    #[test]
    fn parses_qdrant_memory_and_collection_gauges() {
        let sample = "\
# HELP memory_resident_bytes Resident memory
# TYPE memory_resident_bytes gauge
app_info{name=\"qdrant\",version=\"1.13.4\"} 1
collections_total 3
collections_vector_total 1500000
cluster_enabled 0
memory_active_bytes 57212928
memory_allocated_bytes 48281048
memory_resident_bytes 74133504
rest_responses_total{method=\"GET\"} 42
";
        let m = parse_qdrant_metrics(sample);
        assert_eq!(
            m.get("memory_resident_bytes").unwrap().as_f64(),
            Some(74133504.0)
        );
        assert_eq!(
            m.get("collections_vector_total").unwrap().as_f64(),
            Some(1500000.0)
        );
        assert_eq!(m.get("collections_total").unwrap().as_f64(), Some(3.0));
        // Non-curated / labeled / comment lines are ignored.
        assert!(m.get("app_info").is_none());
        assert!(m.get("rest_responses_total").is_none());
        assert!(m.get("cluster_enabled").is_none());
        // 5 curated gauges present in the sample.
        assert_eq!(m.len(), 5);
    }
}
