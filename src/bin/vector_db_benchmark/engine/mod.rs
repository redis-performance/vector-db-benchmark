//! Engine module - Modular vector database engine implementations.
//!
//! Mirrors Python v0/engine/ structure:
//! - `Engine` trait = BaseClient
//! - `Configurator` trait = BaseConfigurator  
//! - `Uploader` trait = BaseUploader
//! - `Searcher` trait = BaseSearcher

mod elasticsearch;
mod milvus;
mod mongodb_engine;
mod opensearch;
mod pgvector;
mod qdrant;
mod redis;
mod valkey;
mod vectorsets;
mod weaviate;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;

pub use elasticsearch::ElasticsearchEngine;
pub use milvus::MilvusEngine;
pub use mongodb_engine::MongoDBEngine;
pub use opensearch::OpenSearchEngine;
pub use pgvector::PgVectorEngine;
pub use qdrant::QdrantEngine;
pub use redis::RedisEngine;
pub use valkey::ValkeyEngine;
pub use vectorsets::VectorSetsEngine;
pub use weaviate::WeaviateEngine;

/// Upload statistics
#[derive(Debug, Clone, Default)]
pub struct UploadStats {
    pub upload_time: f64,
    pub total_time: f64,
    pub upload_count: usize,
    pub parallel: usize,
    pub batch_size: usize,
    pub memory_usage: Option<serde_json::Value>,
}

/// Search results — matches Python v0 search result JSON fields
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct SearchResults {
    pub total_time: f64,
    pub mean_time: f64,
    pub mean_precision: f64,
    pub std_time: f64,
    pub min_time: f64,
    pub max_time: f64,
    pub rps: f64,
    pub p50_time: f64,
    pub p95_time: f64,
    pub p99_time: f64,
    pub precisions: Vec<f64>,
    pub latencies: Vec<f64>,
    pub top: usize,
    pub num_queries: usize,
    pub parallel: usize,
}

/// Engine trait - equivalent to Python BaseClient
///
/// Each engine implementation provides:
/// - configure: Create/setup the index
/// - upload: Upload vectors to the index
/// - search: Run search queries
/// - delete: Clean up resources
pub trait Engine {
    /// Get engine name
    fn name(&self) -> &str;

    /// Configure the index (create if needed)
    fn configure(&mut self, dataset: &Dataset) -> Result<(), String>;

    /// Upload vectors to the index
    fn upload(&mut self, dataset: &Dataset) -> Result<UploadStats, String>;

    /// Run search benchmark
    fn search(
        &mut self,
        dataset: &Dataset,
        search_params: &SearchParams,
        num_queries: i64,
    ) -> Result<SearchResults, String>;

    /// Delete/cleanup the index
    fn delete(&mut self) -> Result<(), String>;

    /// Get search parameter configurations
    fn search_params(&self) -> &[SearchParams];

    /// Collect memory usage stats after upload (matches Python v0 get_memory_usage)
    fn get_memory_usage(&mut self) -> Option<serde_json::Value> {
        None
    }
}

/// Create an engine based on config
pub fn create_engine(engine_config: &EngineConfig, host: &str) -> Result<Box<dyn Engine>, String> {
    let engine_type = engine_config.engine.as_deref().unwrap_or("unknown");

    match engine_type {
        "redis" => Ok(Box::new(RedisEngine::new(engine_config, host)?)),
        "vectorsets" => Ok(Box::new(VectorSetsEngine::new(engine_config, host)?)),
        "elasticsearch" => Ok(Box::new(ElasticsearchEngine::new(engine_config, host)?)),
        "opensearch" => Ok(Box::new(OpenSearchEngine::new(engine_config, host)?)),
        "qdrant" => Ok(Box::new(QdrantEngine::new(engine_config, host)?)),
        "weaviate" => Ok(Box::new(WeaviateEngine::new(engine_config, host)?)),
        "pgvector" => Ok(Box::new(PgVectorEngine::new(engine_config, host)?)),
        "milvus" => Ok(Box::new(MilvusEngine::new(engine_config, host)?)),
        "mongodb" => Ok(Box::new(MongoDBEngine::new(engine_config, host)?)),
        "valkey" => Ok(Box::new(ValkeyEngine::new(engine_config, host)?)),
        other => Err(format!(
            "Unsupported engine type: '{}'. Supported: 'redis', 'vectorsets', 'elasticsearch', 'opensearch', 'qdrant', 'weaviate', 'pgvector', 'milvus', 'mongodb', 'valkey'.",
            other
        )),
    }
}
