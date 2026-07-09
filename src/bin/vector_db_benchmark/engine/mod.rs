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
mod redis_utils;
mod turbopuffer;
mod valkey;
mod vectorsets;
mod weaviate;
mod weaviate_grpc;

use crate::config::{EngineConfig, SearchParams};
use crate::dataset::Dataset;

pub use elasticsearch::ElasticsearchEngine;
pub use milvus::MilvusEngine;
pub use mongodb_engine::MongoDBEngine;
pub use opensearch::OpenSearchEngine;
pub use pgvector::PgVectorEngine;
pub use qdrant::QdrantEngine;
pub use redis::RedisEngine;
pub use turbopuffer::TurbopufferEngine;
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

/// Update-to-search ratio for mixed workload benchmarks.
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateSearchRatio {
    pub updates: u64,
    pub searches: u64,
}

/// Search results — matches Python v0 search result JSON fields
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct SearchResults {
    pub total_time: f64,
    pub mean_time: f64,
    pub mean_precision: f64,
    pub mean_recall: f64,
    pub mean_mrr: f64,
    pub mean_ndcg: f64,
    pub recalls: Vec<f64>,
    pub mrrs: Vec<f64>,
    pub ndcgs: Vec<f64>,
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
    // Mixed benchmark update metrics (None when search-only)
    pub update_count: Option<usize>,
    pub update_rps: Option<f64>,
    pub update_mean_time: Option<f64>,
    pub update_p50_time: Option<f64>,
    pub update_p95_time: Option<f64>,
    pub update_p99_time: Option<f64>,
    pub update_latencies: Option<Vec<f64>>,
    pub update_search_ratio: Option<String>,
}

/// Build `SearchResults` for a search-only run from the per-query samples
/// collected by an engine's parallel harness. Centralizes rps/means/std/
/// percentile computation so every engine reports metrics identically.
///
/// `times`/`precisions`/`recalls`/`mrrs`/`ndcgs` are the per-successful-query
/// samples (see the engines' search loops), `total_time` the wall clock,
/// `top` the k used, and `parallel` the client concurrency.
#[allow(clippy::too_many_arguments)]
pub fn compute_search_stats(
    times: &[f64],
    precisions: &[f64],
    recalls: &[f64],
    mrrs: &[f64],
    ndcgs: &[f64],
    total_time: f64,
    top: usize,
    parallel: usize,
) -> Result<SearchResults, String> {
    if times.is_empty() {
        return Err("No searches completed".to_string());
    }

    let mean = |v: &[f64]| -> f64 {
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    };

    let mean_time = mean(times);
    let std_time =
        (times.iter().map(|t| (t - mean_time).powi(2)).sum::<f64>() / times.len() as f64).sqrt();
    let min_time = times.iter().copied().fold(f64::INFINITY, f64::min);
    let max_time = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let mut sorted = times.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Nearest-rank percentile, clamped to a valid index.
    let pct = |q: f64| sorted[((sorted.len() as f64 * q) as usize).min(sorted.len() - 1)];

    Ok(SearchResults {
        total_time,
        mean_time,
        mean_precision: mean(precisions),
        mean_recall: mean(recalls),
        mean_mrr: mean(mrrs),
        mean_ndcg: mean(ndcgs),
        recalls: recalls.to_vec(),
        mrrs: mrrs.to_vec(),
        ndcgs: ndcgs.to_vec(),
        std_time,
        min_time,
        max_time,
        rps: times.len() as f64 / total_time,
        p50_time: pct(0.50),
        p95_time: pct(0.95),
        p99_time: pct(0.99),
        precisions: precisions.to_vec(),
        latencies: times.to_vec(),
        top,
        num_queries: times.len(),
        parallel,
        ..Default::default()
    })
}

#[cfg(test)]
mod stats_tests {
    use super::compute_search_stats;

    #[test]
    fn empty_times_errors() {
        assert!(compute_search_stats(&[], &[], &[], &[], &[], 1.0, 10, 1).is_err());
    }

    #[test]
    fn computes_means_rps_and_clamped_percentiles() {
        let times = vec![0.1, 0.2, 0.3, 0.4];
        let ones = vec![1.0, 1.0, 1.0, 1.0];
        let r = compute_search_stats(&times, &ones, &ones, &ones, &ones, 2.0, 10, 4).unwrap();
        assert_eq!(r.num_queries, 4);
        assert!((r.rps - 2.0).abs() < 1e-9); // 4 / 2.0s
        assert!((r.mean_recall - 1.0).abs() < 1e-9);
        assert!((r.mean_time - 0.25).abs() < 1e-9);
        assert!((r.min_time - 0.1).abs() < 1e-9 && (r.max_time - 0.4).abs() < 1e-9);
        // percentile indices stay in-bounds (no panic, no 0.0 fallback)
        assert!(r.p50_time > 0.0 && r.p95_time > 0.0 && r.p99_time > 0.0);
        assert_eq!(r.parallel, 4);
        assert_eq!(r.top, 10);
        assert!(r.update_count.is_none());
    }

    #[test]
    fn single_query_percentiles_dont_panic() {
        let r = compute_search_stats(&[0.5], &[1.0], &[1.0], &[1.0], &[1.0], 1.0, 5, 1).unwrap();
        assert!((r.p99_time - 0.5).abs() < 1e-9);
    }
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

    /// Run mixed benchmark (interleaved search + update).
    /// Default: not supported. Override in engines that support it.
    fn search_mixed(
        &mut self,
        _dataset: &Dataset,
        _search_params: &SearchParams,
        _num_queries: i64,
        _ratio: &UpdateSearchRatio,
    ) -> Result<SearchResults, String> {
        Err(format!(
            "mixed benchmark not supported for engine '{}'",
            self.name()
        ))
    }
}

/// Build a Redis connection URL.
///
/// Priority: `REDIS_URI` env var > `REDIS_USER`/`REDIS_AUTH` env vars + host/port.
pub fn build_redis_url(host: &str) -> String {
    if let Ok(uri) = std::env::var("REDIS_URI") {
        return uri;
    }

    let port: u16 = std::env::var("REDIS_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6379);

    let auth = std::env::var("REDIS_AUTH").ok();
    let user = std::env::var("REDIS_USER").ok();

    let auth_part = match (&user, &auth) {
        (Some(u), Some(p)) => format!("{}:{}@", u, p),
        (None, Some(p)) => format!(":{}@", p),
        _ => String::new(),
    };

    format!("redis://{}{}:{}/", auth_part, host, port)
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
        "turbopuffer" => Ok(Box::new(TurbopufferEngine::new(engine_config, host)?)),
        other => Err(format!(
            "Unsupported engine type: '{}'. Supported: 'redis', 'vectorsets', 'elasticsearch', 'opensearch', 'qdrant', 'weaviate', 'pgvector', 'milvus', 'mongodb', 'valkey', 'turbopuffer'.",
            other
        )),
    }
}
