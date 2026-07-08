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
    BinaryQuantization, Condition, CreateCollectionBuilder, DeleteCollectionBuilder, Distance,
    FieldType, Filter, HnswConfigDiff, MaxOptimizationThreads, OptimizersConfigDiff, PointStruct,
    QuantizationSearchParams, QuantizationType, ScalarQuantization, SearchPointsBuilder,
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
    #[allow(dead_code)]
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

        let grpc_url = if let Some(url) = std::env::var("QDRANT_URL").ok() {
            url
        } else {
            format!("http://{}:{}", clean_host, grpc_port)
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

        let vector_params = VectorParamsBuilder::new(vector_size as u64, qdrant_distance);

        let mut create_builder = CreateCollectionBuilder::new(&self.collection_name)
            .vectors_config(VectorsConfig {
                config: Some(Config::Params(vector_params.build())),
            });

        // Apply HNSW config if specified
        if hnsw_m.is_some() || hnsw_ef.is_some() {
            let mut hnsw_config = HnswConfigDiff::default();
            if let Some(m) = hnsw_m {
                hnsw_config.m = Some(m as u64);
            }
            if let Some(ef) = hnsw_ef {
                hnsw_config.ef_construct = Some(ef as u64);
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
            } else if let Some(b) = q.get("binary") {
                Some(Quantization::Binary(BinaryQuantization {
                    always_ram: b.get("always_ram").and_then(|v| v.as_bool()),
                    ..Default::default()
                }))
            } else {
                None
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

        // Create payload indexes for schema fields
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

        Ok(())
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

        for _ in 0..600 {
            std::thread::sleep(std::time::Duration::from_secs(5));

            if let Ok(info) = self
                .rt
                .block_on(self.client.collection_info(&self.collection_name))
            {
                // status: 1 = Green, 2 = Yellow, 3 = Red (from protobuf enum)
                if let Some(result) = info.result {
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
        Err("Timed out waiting for collection to reach GREEN status".to_string())
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

fn build_qdrant_filter(
    field_name: &str,
    condition_type: &str,
    criteria: &serde_json::Value,
) -> Option<Condition> {
    match condition_type {
        "match" => {
            let value = criteria.get("value")?;
            if let Some(s) = value.as_str() {
                Some(Condition::matches(field_name.to_string(), s.to_string()))
            } else if let Some(n) = value.as_i64() {
                Some(Condition::matches(field_name.to_string(), n))
            } else {
                None
            }
        }
        "range" => {
            let criteria_obj = criteria.as_object()?;
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

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<Filter>> = conditions
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

                        let mut search_builder = SearchPointsBuilder::new(
                            &collection_name,
                            queries[idx].clone(),
                            top as u64,
                        )
                        .with_payload(false);

                        if hnsw_ef.is_some() || quantization_params.is_some() {
                            let search_params = qdrant_client::qdrant::SearchParams {
                                hnsw_ef,
                                quantization: quantization_params.clone(),
                                ..Default::default()
                            };
                            search_builder = search_builder.params(search_params);
                        }

                        if let Some(filter) = &parsed_filters[idx] {
                            search_builder = search_builder.filter(filter.clone());
                        }

                        let query_start = Instant::now();
                        let result = rt.block_on(client.search_points(search_builder));
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

    fn get_memory_usage(&mut self) -> Option<serde_json::Value> {
        let info = self
            .rt
            .block_on(self.client.collection_info(&self.collection_name))
            .ok()?;
        Some(serde_json::json!({
            "collection_info": format!("{:?}", info.result),
        }))
    }
}
