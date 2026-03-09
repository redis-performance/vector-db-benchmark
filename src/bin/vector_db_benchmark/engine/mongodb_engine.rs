//! MongoDB Atlas Vector Search engine implementation.
//!
//! Uses the official `mongodb` crate with sync feature.
//! Supports Atlas Vector Search with HNSW index via `$vectorSearch` aggregation.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};
use mongodb::bson::{doc, Document};
use mongodb::sync::Client;

use rand::{seq::SliceRandom, SeedableRng};

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;
use crate::engine::{Engine, SearchResults, UpdateSearchRatio, UploadStats};
use vector_db_benchmark::readers::metadata::MetadataItem;

const DEFAULT_DB: &str = "bench";
const DEFAULT_COLLECTION: &str = "vectors";
const DEFAULT_INDEX_NAME: &str = "vector_index";

#[derive(Clone)]
struct MongoConfig {
    batch_size: usize,
    parallel: usize,
    num_candidates_factor: i64,
    skip_vector_index: bool,
}

pub struct MongoDBEngine {
    name: String,
    db_name: String,
    collection_name: String,
    index_name: String,
    config: MongoConfig,
    search_params: Vec<SearchParams>,
    /// MongoDB connection URI
    uri: String,
    /// Shared MongoDB client (connection pool)
    client: Client,
}

impl MongoDBEngine {
    pub fn new(engine_config: &EngineConfig, host: &str) -> Result<Self, String> {
        let port: u16 = std::env::var("MONGODB_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(27017);

        let db_name = std::env::var("MONGODB_DB").unwrap_or_else(|_| DEFAULT_DB.to_string());
        let collection_name =
            std::env::var("MONGODB_COLLECTION").unwrap_or_else(|_| DEFAULT_COLLECTION.to_string());
        let index_name =
            std::env::var("MONGODB_INDEX_NAME").unwrap_or_else(|_| DEFAULT_INDEX_NAME.to_string());

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
            .unwrap_or(500) as usize;

        let num_candidates_factor = engine_config
            .upload_params
            .as_ref()
            .and_then(|p| p.get("num_candidates_factor"))
            .and_then(|v| v.as_i64())
            .unwrap_or(10);

        let uri = build_uri(host, port);

        let client = Client::with_uri_str(&uri)
            .map_err(|e| format!("Failed to create MongoDB client: {}", e))?;

        Ok(Self {
            name: engine_config.name.clone(),
            db_name,
            collection_name,
            index_name,
            config: MongoConfig {
                batch_size,
                parallel,
                num_candidates_factor,
                skip_vector_index: engine_config.skip_vector_index,
            },
            search_params: engine_config.search_params.clone().unwrap_or_default(),
            uri,
            client,
        })
    }

    /// Filter-only search: run collection.find(filter).limit(top) with no vector search.
    fn search_filter_only(
        &self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String> {
        let parallel = params.parallel.unwrap_or(1) as usize;

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (_queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<Document>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_mongo_conditions))
            .collect();

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(parsed_filters.len())
        } else {
            parsed_filters.len()
        };

        let runnable_indices: Vec<usize> = (0..num_to_run)
            .filter(|&i| parsed_filters[i].is_some())
            .collect();

        if runnable_indices.is_empty() {
            return Err(
                "No queries with filter conditions for filter-only search".to_string(),
            );
        }

        let search_times: Arc<Mutex<Vec<f64>>> =
            Arc::new(Mutex::new(Vec::with_capacity(runnable_indices.len())));
        let query_idx = Arc::new(AtomicUsize::new(0));

        let pb = self.create_progress_bar(runnable_indices.len());
        let start_time = Instant::now();

        let uri = self.uri.clone();
        let db_name = self.db_name.clone();
        let collection_name = self.collection_name.clone();

        std::thread::scope(|s| {
            for _ in 0..parallel {
                let uri = uri.clone();
                let db_name = db_name.clone();
                let collection_name = collection_name.clone();
                let parsed_filters = &parsed_filters;
                let runnable_indices = &runnable_indices;
                let neighbors = &neighbors;
                let search_times = Arc::clone(&search_times);
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                s.spawn(move || {
                    let client = match Client::with_uri_str(&uri) {
                        Ok(c) => c,
                        Err(_) => return,
                    };
                    let coll = client
                        .database(&db_name)
                        .collection::<Document>(&collection_name);

                    loop {
                        let seq = query_idx.fetch_add(1, Ordering::SeqCst);
                        if seq >= runnable_indices.len() {
                            break;
                        }
                        let idx = runnable_indices[seq];

                        let top = explicit_top.unwrap_or_else(|| {
                            let n = neighbors[idx].len();
                            if n > 0 { n } else { 10 }
                        });

                        let filter = parsed_filters[idx].as_ref().unwrap();

                        let query_start = Instant::now();
                        let _ = filter_only_find(&coll, filter, top);
                        let query_time = query_start.elapsed().as_secs_f64();

                        search_times.lock().unwrap().push(query_time);
                        pb.inc(1);
                    }
                });
            }
        });

        pb.finish_and_clear();
        let total_time = start_time.elapsed().as_secs_f64();

        let times = search_times.lock().unwrap();
        if times.is_empty() {
            return Err("No filter-only searches completed".to_string());
        }

        let rps = times.len() as f64 / total_time;
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
            mean_precision: -1.0,
            std_time,
            min_time,
            max_time,
            rps,
            p50_time,
            p95_time,
            p99_time,
            precisions: vec![],
            latencies: times.to_vec(),
            top: 0,
            num_queries: times.len(),
            parallel,
            ..Default::default()
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

    fn drop_collection(&self) -> Result<(), String> {
        let db = self.client.database(&self.db_name);

        // 1. Drop the search index explicitly and wait for it to disappear.
        //    On Atlas, stale indexes can prevent clean recreation.
        println!("Dropping search index '{}'...", self.index_name);
        let drop_cmd = doc! {
            "dropSearchIndex": &self.collection_name,
            "name": &self.index_name,
        };
        // Ignore errors (e.g. IndexNotFound, collection doesn't exist)
        let _ = db.run_command(drop_cmd).run();

        let deadline = Instant::now() + std::time::Duration::from_secs(120);
        loop {
            let cmd = doc! { "listSearchIndexes": &self.collection_name };
            let index_exists = db.run_command(cmd).run().ok().map_or(false, |result| {
                result
                    .get_document("cursor")
                    .ok()
                    .and_then(|c| c.get_array("firstBatch").ok())
                    .map_or(false, |batch| {
                        batch.iter().any(|idx| {
                            idx.as_document()
                                .and_then(|d| d.get_str("name").ok())
                                .map_or(false, |n| n == self.index_name)
                        })
                    })
            });

            if !index_exists {
                break;
            }
            if Instant::now() > deadline {
                eprintln!(
                    "Warning: search index '{}' still exists after 120s, proceeding anyway",
                    self.index_name
                );
                break;
            }
            std::thread::sleep(std::time::Duration::from_secs(5));
        }

        // 2. Drop the collection and verify it's gone.
        let coll = db.collection::<Document>(&self.collection_name);
        coll.drop()
            .run()
            .map_err(|e| format!("Failed to drop collection: {}", e))?;

        let deadline = Instant::now() + std::time::Duration::from_secs(60);
        loop {
            let names = db.list_collection_names().run().unwrap_or_default();
            if !names.contains(&self.collection_name.to_string()) {
                break;
            }
            if Instant::now() > deadline {
                eprintln!(
                    "Warning: collection '{}' still exists after 60s, proceeding anyway",
                    self.collection_name
                );
                break;
            }
            std::thread::sleep(std::time::Duration::from_secs(5));
        }

        Ok(())
    }

    fn create_vector_index(&self, dataset: &Dataset) -> Result<(), String> {
        let vector_size = dataset.vector_size();
        let distance = dataset.distance();

        let similarity = match distance.to_lowercase().as_str() {
            "l2" | "euclidean" => "euclidean",
            "cosine" | "angular" => "cosine",
            "dot" | "ip" => "dotProduct",
            other => {
                return Err(format!(
                    "Unsupported distance metric for MongoDB: {}",
                    other
                ))
            }
        };

        // Build vector search index definition
        let mut fields = vec![doc! {
            "type": "vector",
            "path": "vector",
            "numDimensions": vector_size as i32,
            "similarity": similarity,
        }];

        // Add filter fields from dataset schema
        if let Some(schema) = &dataset.config.schema {
            if let Some(schema_obj) = schema.as_object() {
                for (field_name, _field_type) in schema_obj {
                    fields.push(doc! {
                        "type": "filter",
                        "path": field_name,
                    });
                }
            }
        }

        let index_def = doc! {
            "name": &self.index_name,
            "type": "vectorSearch",
            "definition": {
                "fields": fields,
            }
        };

        let db = self.client.database(&self.db_name);
        let cmd = doc! {
            "createSearchIndexes": &self.collection_name,
            "indexes": [index_def],
        };

        db.run_command(cmd)
            .run()
            .map_err(|e| format!("Failed to create vector search index: {}", e))?;

        // Wait for index to become ready
        self.wait_for_index_ready()?;

        Ok(())
    }

    fn wait_for_index_ready(&self) -> Result<(), String> {
        println!("Waiting for vector search index to become ready...");
        let db = self.client.database(&self.db_name);
        let deadline = Instant::now() + std::time::Duration::from_secs(120);

        loop {
            let cmd = doc! {
                "listSearchIndexes": &self.collection_name,
            };

            if let Ok(result) = db.run_command(cmd).run() {
                if let Ok(cursor) = result.get_document("cursor") {
                    if let Ok(batch) = cursor.get_array("firstBatch") {
                        for index in batch {
                            if let Some(index_doc) = index.as_document() {
                                let name = index_doc.get_str("name").unwrap_or("");
                                let status = index_doc.get_str("status").unwrap_or("");
                                let queryable = index_doc.get_bool("queryable").unwrap_or(false);
                                // Atlas uses READY, local uses ACTIVE
                                if name == self.index_name
                                    && (status == "READY" || status == "ACTIVE")
                                    && queryable
                                {
                                    println!(
                                        "Vector search index is ready (status={}, queryable=true).",
                                        status
                                    );
                                    return Ok(());
                                }
                            }
                        }
                    }
                }
            }

            if Instant::now() > deadline {
                return Err(
                    "Vector search index did not become ready within 120 seconds".to_string(),
                );
            }
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
    }

    /// Wait for the vector search index to finish indexing all uploaded documents.
    ///
    /// Polls `listSearchIndexes` until the index reports `queryable=true` and
    /// `status` is READY or ACTIVE, then verifies with a probe search using
    /// the first uploaded vector to confirm the index has actually ingested docs.
    fn wait_for_index_catchup(
        &self,
        expected_count: usize,
        probe_vector: &[f32],
    ) -> Result<(), String> {
        println!(
            "Waiting for vector search index to index all {} documents...",
            expected_count
        );
        let db = self.client.database(&self.db_name);
        let coll = db.collection::<Document>(&self.collection_name);
        let start = Instant::now();
        let deadline = start + std::time::Duration::from_secs(600);
        let mut last_print = Instant::now();
        let mut index_ready = false;

        loop {
            if !index_ready {
                let cmd = doc! { "listSearchIndexes": &self.collection_name };
                if let Ok(result) = db.run_command(cmd).run() {
                    if let Ok(cursor) = result.get_document("cursor") {
                        if let Ok(batch) = cursor.get_array("firstBatch") {
                            for index in batch {
                                if let Some(index_doc) = index.as_document() {
                                    let name = index_doc.get_str("name").unwrap_or("");
                                    if name != self.index_name {
                                        continue;
                                    }
                                    let status = index_doc.get_str("status").unwrap_or("");
                                    let queryable =
                                        index_doc.get_bool("queryable").unwrap_or(false);

                                    if (status == "READY" || status == "ACTIVE") && queryable {
                                        index_ready = true;
                                    } else if last_print.elapsed().as_secs() >= 10 {
                                        println!(
                                            "  index building... status={}, queryable={}",
                                            status, queryable
                                        );
                                        last_print = Instant::now();
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // Once the index reports ready, do a probe search to verify ingestion
            if index_ready {
                match vector_search(&coll, &self.index_name, probe_vector, 1, 10, None) {
                    Ok(results) if !results.is_empty() => {
                        println!(
                            "Index ready (probe search returned results) after {:.1}s.",
                            start.elapsed().as_secs_f64()
                        );
                        return Ok(());
                    }
                    _ => {
                        if last_print.elapsed().as_secs() >= 10 {
                            println!(
                                "  index ready but probe search returned no results, waiting..."
                            );
                            last_print = Instant::now();
                        }
                    }
                }
            }

            if Instant::now() > deadline {
                return Err(
                    "Vector search index did not finish indexing within 600 seconds".to_string(),
                );
            }
            std::thread::sleep(std::time::Duration::from_secs(2));
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
            .step_by(self.config.batch_size)
            .map(|start| (start, (start + self.config.batch_size).min(ids.len())))
            .collect();

        let total_batches = batches.len();
        let batch_idx = Arc::new(AtomicUsize::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let uri = self.uri.clone();
        let db_name = self.db_name.clone();
        let collection_name = self.collection_name.clone();

        std::thread::scope(|s| {
            for _ in 0..self.config.parallel {
                let uri = uri.clone();
                let db_name = db_name.clone();
                let collection_name = collection_name.clone();
                let batches = &batches;
                let batch_idx = Arc::clone(&batch_idx);
                let error = Arc::clone(&error);
                let pb = &pb;

                s.spawn(move || {
                    let client = match Client::with_uri_str(&uri) {
                        Ok(c) => c,
                        Err(e) => {
                            *error.lock().unwrap() = Some(e.to_string());
                            return;
                        }
                    };

                    let coll = client
                        .database(&db_name)
                        .collection::<Document>(&collection_name);

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
                            &coll,
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

fn build_uri(host: &str, port: u16) -> String {
    let user = std::env::var("MONGODB_USER").ok();
    let password = std::env::var("MONGODB_PASSWORD").ok();

    let host_part = if host.starts_with("mongodb") {
        // Already a full URI
        return host.to_string();
    } else {
        host
    };

    match (user, password) {
        (Some(u), Some(p)) => {
            format!(
                "mongodb://{}:{}@{}:{}/?directConnection=true",
                u, p, host_part, port
            )
        }
        _ => format!("mongodb://{}:{}/?directConnection=true", host_part, port),
    }
}

/// Insert a batch of documents into MongoDB.
fn insert_batch(
    coll: &mongodb::sync::Collection<Document>,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
) -> Result<(), String> {
    use vector_db_benchmark::readers::metadata::MetadataValue;

    let docs: Vec<Document> = ids
        .iter()
        .zip(vectors.iter().zip(metadata.iter()))
        .map(|(&id, (vec, meta))| {
            let bson_vec: Vec<mongodb::bson::Bson> = vec
                .iter()
                .map(|&f| mongodb::bson::Bson::Double(f as f64))
                .collect();

            let mut doc = doc! {
                "_id": id,
                "vector": bson_vec,
            };

            if let Some(meta) = meta {
                for (k, v) in &meta.fields {
                    let bson_val = match v {
                        MetadataValue::String(s) => mongodb::bson::Bson::String(s.clone()),
                        MetadataValue::Labels(labels) => {
                            let arr: Vec<mongodb::bson::Bson> = labels
                                .iter()
                                .map(|l| mongodb::bson::Bson::String(l.clone()))
                                .collect();
                            mongodb::bson::Bson::Array(arr)
                        }
                        MetadataValue::Geo { lon, lat } => mongodb::bson::Bson::Document(doc! {
                            "type": "Point",
                            "coordinates": [*lon, *lat],
                        }),
                    };
                    doc.insert(k.clone(), bson_val);
                }
            }

            doc
        })
        .collect();

    coll.insert_many(docs)
        .run()
        .map_err(|e| format!("Insert batch failed: {}", e))?;

    Ok(())
}

/// Update a single document's vector and metadata.
fn update_one_doc(
    coll: &mongodb::sync::Collection<Document>,
    id: i64,
    vector: &[f32],
    metadata: Option<&MetadataItem>,
) -> Result<(), String> {
    use vector_db_benchmark::readers::metadata::MetadataValue;

    let bson_vec: Vec<mongodb::bson::Bson> = vector
        .iter()
        .map(|&f| mongodb::bson::Bson::Double(f as f64))
        .collect();

    let mut set_doc = doc! { "vector": bson_vec };

    if let Some(meta) = metadata {
        for (k, v) in &meta.fields {
            let bson_val = match v {
                MetadataValue::String(s) => mongodb::bson::Bson::String(s.clone()),
                MetadataValue::Labels(labels) => {
                    let arr: Vec<mongodb::bson::Bson> = labels
                        .iter()
                        .map(|l| mongodb::bson::Bson::String(l.clone()))
                        .collect();
                    mongodb::bson::Bson::Array(arr)
                }
                MetadataValue::Geo { lon, lat } => mongodb::bson::Bson::Document(doc! {
                    "type": "Point",
                    "coordinates": [*lon, *lat],
                }),
            };
            set_doc.insert(k.clone(), bson_val);
        }
    }

    coll.update_one(doc! { "_id": id }, doc! { "$set": set_doc })
        .run()
        .map_err(|e| format!("Update failed for id {}: {}", id, e))?;

    Ok(())
}

/// Execute a filter-only find (no vector search).
fn filter_only_find(
    coll: &mongodb::sync::Collection<Document>,
    filter: &Document,
    top: usize,
) -> Result<usize, String> {
    let cursor = coll
        .find(filter.clone())
        .limit(top as i64)
        .projection(doc! { "_id": 1 })
        .run()
        .map_err(|e| format!("Filter-only find failed: {}", e))?;

    let mut count = 0usize;
    for result in cursor {
        let _ = result.map_err(|e| format!("Failed to read result: {}", e))?;
        count += 1;
    }
    Ok(count)
}

/// Execute a vector search using $vectorSearch aggregation pipeline.
fn vector_search(
    coll: &mongodb::sync::Collection<Document>,
    index_name: &str,
    query_vector: &[f32],
    top: usize,
    num_candidates: i64,
    filter: Option<&Document>,
) -> Result<Vec<(i64, f64)>, String> {
    let bson_vec: Vec<mongodb::bson::Bson> = query_vector
        .iter()
        .map(|&f| mongodb::bson::Bson::Double(f as f64))
        .collect();

    let mut vs_stage = doc! {
        "index": index_name,
        "path": "vector",
        "queryVector": bson_vec,
        "numCandidates": num_candidates,
        "limit": top as i64,
    };

    if let Some(f) = filter {
        vs_stage.insert("filter", f.clone());
    }

    let pipeline = vec![
        doc! { "$vectorSearch": vs_stage },
        doc! {
            "$project": {
                "_id": 1,
                "score": { "$meta": "vectorSearchScore" },
            }
        },
    ];

    let cursor = coll
        .aggregate(pipeline)
        .run()
        .map_err(|e| format!("Vector search failed: {}", e))?;

    let mut results = Vec::with_capacity(top);
    for result in cursor {
        let doc = result.map_err(|e| format!("Failed to read result: {}", e))?;
        let id = doc.get_i64("_id").unwrap_or(0);
        let score = doc.get_f64("score").unwrap_or(0.0);
        results.push((id, score));
    }

    Ok(results)
}

/// Parse filter conditions into MongoDB query document.
fn parse_mongo_conditions(conditions: &serde_json::Value) -> Option<Document> {
    let obj = conditions.as_object()?;
    if obj.is_empty() {
        return None;
    }

    let mut filter_clauses: Vec<mongodb::bson::Bson> = Vec::new();

    if let Some(and_entries) = obj.get("and").and_then(|v| v.as_array()) {
        for entry in and_entries {
            if let Some(clause) = build_mongo_filter_entry(entry) {
                filter_clauses.push(mongodb::bson::Bson::Document(clause));
            }
        }
    }

    if let Some(or_entries) = obj.get("or").and_then(|v| v.as_array()) {
        let or_clauses: Vec<mongodb::bson::Bson> = or_entries
            .iter()
            .filter_map(build_mongo_filter_entry)
            .map(mongodb::bson::Bson::Document)
            .collect();
        if !or_clauses.is_empty() {
            filter_clauses.push(mongodb::bson::Bson::Document(doc! {
                "$or": or_clauses,
            }));
        }
    }

    if filter_clauses.is_empty() {
        return None;
    }

    if filter_clauses.len() == 1 {
        if let Some(mongodb::bson::Bson::Document(d)) = filter_clauses.first().cloned() {
            return Some(d);
        }
    }

    Some(doc! { "$and": filter_clauses })
}

fn build_mongo_filter_entry(entry: &serde_json::Value) -> Option<Document> {
    let entry_obj = entry.as_object()?;
    let mut clauses = Document::new();

    for (field_name, field_filters) in entry_obj {
        let filter_obj = field_filters.as_object()?;
        for (condition_type, criteria) in filter_obj {
            match condition_type.as_str() {
                "match" => {
                    if let Some(value) = criteria.get("value") {
                        clauses.insert(field_name.clone(), json_to_bson(value));
                    }
                }
                "range" => {
                    let mut range_doc = Document::new();
                    if let Some(gt) = criteria.get("gt") {
                        if !gt.is_null() {
                            range_doc.insert("$gt", json_to_bson(gt));
                        }
                    }
                    if let Some(lt) = criteria.get("lt") {
                        if !lt.is_null() {
                            range_doc.insert("$lt", json_to_bson(lt));
                        }
                    }
                    if let Some(gte) = criteria.get("gte") {
                        if !gte.is_null() {
                            range_doc.insert("$gte", json_to_bson(gte));
                        }
                    }
                    if let Some(lte) = criteria.get("lte") {
                        if !lte.is_null() {
                            range_doc.insert("$lte", json_to_bson(lte));
                        }
                    }
                    if !range_doc.is_empty() {
                        clauses.insert(field_name.clone(), range_doc);
                    }
                }
                _ => {}
            }
        }
    }

    if clauses.is_empty() {
        None
    } else {
        Some(clauses)
    }
}

fn json_to_bson(value: &serde_json::Value) -> mongodb::bson::Bson {
    match value {
        serde_json::Value::Null => mongodb::bson::Bson::Null,
        serde_json::Value::Bool(b) => mongodb::bson::Bson::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                mongodb::bson::Bson::Int64(i)
            } else if let Some(f) = n.as_f64() {
                mongodb::bson::Bson::Double(f)
            } else {
                mongodb::bson::Bson::Null
            }
        }
        serde_json::Value::String(s) => mongodb::bson::Bson::String(s.clone()),
        serde_json::Value::Array(arr) => {
            let bson_arr: Vec<mongodb::bson::Bson> = arr.iter().map(json_to_bson).collect();
            mongodb::bson::Bson::Array(bson_arr)
        }
        serde_json::Value::Object(obj) => {
            let mut doc = Document::new();
            for (k, v) in obj {
                doc.insert(k.clone(), json_to_bson(v));
            }
            mongodb::bson::Bson::Document(doc)
        }
    }
}

// ── Engine trait implementation ──────────────────────────────────────────

impl Engine for MongoDBEngine {
    fn name(&self) -> &str {
        &self.name
    }

    fn search_params(&self) -> &[SearchParams] {
        &self.search_params
    }

    fn configure(&mut self, dataset: &Dataset) -> Result<(), String> {
        println!("Dropping existing collection...");
        let _ = self.drop_collection();

        // Create the collection explicitly so we can add the index
        let db = self.client.database(&self.db_name);
        db.create_collection(&self.collection_name)
            .run()
            .map_err(|e| format!("Failed to create collection: {}", e))?;

        println!(
            "Collection '{}.{}' created.",
            self.db_name, self.collection_name
        );

        if self.config.skip_vector_index {
            println!("Skipping vector index (filter-only mode)");
            return Ok(());
        }

        // Insert a dummy document so the index has something to build on
        let coll = db.collection::<Document>(&self.collection_name);
        let dim = dataset.vector_size();
        let dummy_vec: Vec<mongodb::bson::Bson> =
            (0..dim).map(|_| mongodb::bson::Bson::Double(0.0)).collect();
        coll.insert_one(doc! { "_id": -1i64, "vector": dummy_vec })
            .run()
            .map_err(|e| format!("Failed to insert dummy document: {}", e))?;

        println!("Creating vector search index '{}'...", self.index_name);
        self.create_vector_index(dataset)?;

        // Remove dummy document
        coll.delete_one(doc! { "_id": -1i64 })
            .run()
            .map_err(|e| format!("Failed to remove dummy document: {}", e))?;

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

        let total_time;
        if self.config.skip_vector_index {
            total_time = read_time + upload_time;
            println!("Total time (read+upload): {:.3}s (no vector index)", total_time);
        } else {
            // Wait for the search index to finish indexing all uploaded documents
            // Use the first vector as a probe query to verify actual search readiness
            let probe_vector = vectors.first().ok_or("No vectors uploaded")?;
            let index_start = Instant::now();
            self.wait_for_index_catchup(vectors.len(), probe_vector)?;
            let index_time = index_start.elapsed().as_secs_f64();

            total_time = read_time + upload_time + index_time;
            println!(
                "Index time: {:.3}s, Total time (read+upload+index): {:.3}s",
                index_time, total_time
            );
        }

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
        if self.config.skip_vector_index {
            return self.search_filter_only(dataset, params, num_queries);
        }

        let parallel = params.parallel.unwrap_or(1) as usize;
        let num_candidates_factor = params
            .num_candidates
            .unwrap_or(self.config.num_candidates_factor);

        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<Document>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_mongo_conditions))
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

        let pb = self.create_progress_bar(num_to_run);

        let start_time = Instant::now();
        let uri = self.uri.clone();
        let db_name = self.db_name.clone();
        let collection_name = self.collection_name.clone();
        let index_name = self.index_name.clone();

        std::thread::scope(|s| {
            for _ in 0..parallel {
                let uri = uri.clone();
                let db_name = db_name.clone();
                let collection_name = collection_name.clone();
                let index_name = index_name.clone();
                let queries = &queries;
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
                let search_times = Arc::clone(&search_times);
                let precisions = Arc::clone(&precisions);
                let query_idx = Arc::clone(&query_idx);
                let pb = &pb;

                s.spawn(move || {
                    let client = match Client::with_uri_str(&uri) {
                        Ok(c) => c,
                        Err(_) => return,
                    };
                    let coll = client
                        .database(&db_name)
                        .collection::<Document>(&collection_name);

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

                        let num_candidates = (top as i64) * num_candidates_factor;

                        let query_start = Instant::now();
                        let results = vector_search(
                            &coll,
                            &index_name,
                            &queries[idx],
                            top,
                            num_candidates,
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
                        pb.inc(1);
                    }
                });
            }
        });

        pb.finish_and_clear();
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
            ..Default::default()
        })
    }

    fn search_mixed(
        &mut self,
        dataset: &Dataset,
        params: &SearchParams,
        num_queries: i64,
        ratio: &UpdateSearchRatio,
    ) -> Result<SearchResults, String> {
        let parallel = params.parallel.unwrap_or(1) as usize;
        let num_candidates_factor = params
            .num_candidates
            .unwrap_or(self.config.num_candidates_factor);

        // Read queries and ground truth
        let query_path = dataset.get_path()?;
        println!("\tReading queries from {}...", query_path.display());
        let (queries, neighbors, conditions) = dataset.read_queries()?;

        let parsed_filters: Vec<Option<Document>> = conditions
            .iter()
            .map(|c| c.as_ref().and_then(parse_mongo_conditions))
            .collect();

        // Read vectors for updates
        let normalize = dataset.needs_normalization();
        println!("\tReading vectors for updates...");
        let (upd_ids, upd_vectors, upd_metadata) = dataset.read_vectors(normalize)?;

        // Create deterministic shuffled update sequence
        let mut update_seq: Vec<usize> = (0..upd_ids.len()).collect();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        update_seq.shuffle(&mut rng);

        let explicit_top: Option<usize> = params.top.map(|t| t as usize);
        let num_to_run = if num_queries > 0 {
            (num_queries as usize).min(queries.len())
        } else {
            queries.len()
        };

        let search_times: Arc<Mutex<Vec<f64>>> =
            Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let update_times: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
        let precisions: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::with_capacity(num_to_run)));
        let search_idx = Arc::new(AtomicUsize::new(0));
        let update_idx = Arc::new(AtomicUsize::new(0));

        let ratio_searches = ratio.searches as usize;
        let ratio_updates = ratio.updates as usize;
        let update_seq_len = update_seq.len();

        let pb = self.create_progress_bar(num_to_run);
        let start_time = Instant::now();

        let uri = self.uri.clone();
        let db_name = self.db_name.clone();
        let collection_name = self.collection_name.clone();
        let index_name = self.index_name.clone();

        std::thread::scope(|s| {
            for _ in 0..parallel {
                let uri = uri.clone();
                let db_name = db_name.clone();
                let collection_name = collection_name.clone();
                let index_name = index_name.clone();
                let queries = &queries;
                let neighbors = &neighbors;
                let parsed_filters = &parsed_filters;
                let upd_ids = &upd_ids;
                let upd_vectors = &upd_vectors;
                let upd_metadata = &upd_metadata;
                let update_seq = &update_seq;
                let search_times = Arc::clone(&search_times);
                let update_times = Arc::clone(&update_times);
                let precisions = Arc::clone(&precisions);
                let search_idx = Arc::clone(&search_idx);
                let update_idx = Arc::clone(&update_idx);
                let pb = &pb;

                s.spawn(move || {
                    let client = match Client::with_uri_str(&uri) {
                        Ok(c) => c,
                        Err(_) => return,
                    };
                    let coll = client
                        .database(&db_name)
                        .collection::<Document>(&collection_name);

                    'outer: loop {
                        // Search phase: do S searches
                        for _ in 0..ratio_searches {
                            let idx = search_idx.fetch_add(1, Ordering::SeqCst);
                            if idx >= num_to_run {
                                break 'outer;
                            }

                            let top = explicit_top.unwrap_or_else(|| {
                                let n = neighbors[idx].len();
                                if n > 0 {
                                    n
                                } else {
                                    10
                                }
                            });

                            let num_candidates = (top as i64) * num_candidates_factor;

                            let query_start = Instant::now();
                            let results = vector_search(
                                &coll,
                                &index_name,
                                &queries[idx],
                                top,
                                num_candidates,
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
                            pb.inc(1);
                        }

                        // Update phase: do U updates
                        for _ in 0..ratio_updates {
                            let uidx = update_idx.fetch_add(1, Ordering::SeqCst);
                            let data_idx = update_seq[uidx % update_seq_len];

                            let update_start = Instant::now();
                            let _ = update_one_doc(
                                &coll,
                                upd_ids[data_idx],
                                &upd_vectors[data_idx],
                                upd_metadata[data_idx].as_ref(),
                            );
                            let update_time = update_start.elapsed().as_secs_f64();
                            update_times.lock().unwrap().push(update_time);
                        }
                    }
                });
            }
        });

        pb.finish_and_clear();
        let total_time = start_time.elapsed().as_secs_f64();

        let times = search_times.lock().unwrap();
        let precs = precisions.lock().unwrap();
        let u_times = update_times.lock().unwrap();

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

        // Update latency stats
        let (update_count, update_rps, update_mean_time, update_p50, update_p95, update_p99) =
            if !u_times.is_empty() {
                let u_rps = u_times.len() as f64 / total_time;
                let u_mean = u_times.iter().sum::<f64>() / u_times.len() as f64;
                let mut u_sorted: Vec<f64> = u_times.clone();
                u_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let u_p50 = u_sorted
                    .get((u_sorted.len() as f64 * 0.50) as usize)
                    .copied()
                    .unwrap_or(0.0);
                let u_p95 = u_sorted
                    .get(((u_sorted.len() as f64 * 0.95) as usize).min(u_sorted.len() - 1))
                    .copied()
                    .unwrap_or(0.0);
                let u_p99 = u_sorted
                    .get(((u_sorted.len() as f64 * 0.99) as usize).min(u_sorted.len() - 1))
                    .copied()
                    .unwrap_or(0.0);
                (
                    Some(u_times.len()),
                    Some(u_rps),
                    Some(u_mean),
                    Some(u_p50),
                    Some(u_p95),
                    Some(u_p99),
                )
            } else {
                (None, None, None, None, None, None)
            };

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
            update_count,
            update_rps,
            update_mean_time,
            update_p50_time: update_p50,
            update_p95_time: update_p95,
            update_p99_time: update_p99,
            update_latencies: Some(u_times.to_vec()),
            update_search_ratio: Some(format!("{}:{}", ratio.updates, ratio.searches)),
        })
    }

    fn delete(&mut self) -> Result<(), String> {
        self.drop_collection()
    }
}
