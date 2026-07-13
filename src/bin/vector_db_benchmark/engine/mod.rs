//! Engine module - Modular vector database engine implementations.
//!
//! Mirrors Python v0/engine/ structure:
//! - `Engine` trait = BaseClient
//! - `Configurator` trait = BaseConfigurator  
//! - `Uploader` trait = BaseUploader
//! - `Searcher` trait = BaseSearcher

mod dragonfly;
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

pub use dragonfly::DragonflyEngine;
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
    /// Number of *successful* queries folded into the latency/quality stats.
    pub num_queries: usize,
    /// Number of queries requested for this run (num_to_run).
    pub requested_queries: usize,
    /// requested_queries - num_queries: queries that errored/timed out and were
    /// excluded from the latency percentiles. Nonzero means the reported numbers
    /// are over a partial set (e.g. a saturated client shedding timeouts).
    pub failed_queries: usize,
    pub parallel: usize,
    // Client CPU / concurrency-saturation coverage (filled by the runner after
    // the timed window; see proc_cpu). When client_saturated is true the latency
    // and QPS above reflect a client-bound run, not clean server-side numbers.
    pub available_cores: usize,
    pub oversubscribed: bool,
    pub client_cpu_cores_used: Option<f64>,
    pub system_cpu_pct: Option<f64>,
    pub client_saturated: bool,
    pub saturation_reason: String,
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

/// `numpy.percentile` with linear interpolation — the method v0 uses
/// (`np.percentile(..., 50/95/99)` defaults to linear). `sorted` must be
/// ascending and `q` a fraction in `[0, 1]`. The percentile position is
/// `q * (N - 1)`, interpolating between the two neighbouring samples.
///
/// This replaces nearest-rank indexing (`sorted[floor(N*q)]`), which biased
/// every percentile upward and made `p99 == max` for any `N <= 100` (e.g. with
/// N=100, `floor(0.99*100)=99` always selected the single largest sample).
pub fn percentile_linear(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = q * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Build `SearchResults` for a search-only run from the per-query samples
/// collected by an engine's parallel harness. Centralizes rps/means/std/
/// percentile computation so every engine reports metrics identically.
///
/// `times`/`precisions`/`recalls`/`mrrs`/`ndcgs` are the per-successful-query
/// samples (see the engines' search loops), `total_time` the wall clock,
/// `top` the k used, `parallel` the client concurrency, and `requested_queries`
/// the number of queries dispatched (num_to_run) so failures can be counted as
/// `requested_queries - times.len()`. RPS stays successes/wall-clock; a nonzero
/// `failed_queries` flags that the stats cover only the successful subset.
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
    requested_queries: usize,
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
    let pct = |q: f64| percentile_linear(&sorted, q);

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
        requested_queries,
        failed_queries: requested_queries.saturating_sub(times.len()),
        parallel,
        ..Default::default()
    })
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
        "dragonfly" => Ok(Box::new(DragonflyEngine::new(engine_config, host)?)),
        other => Err(format!(
            "Unsupported engine type: '{}'. Supported: 'redis', 'vectorsets', 'elasticsearch', 'opensearch', 'qdrant', 'weaviate', 'pgvector', 'milvus', 'mongodb', 'valkey', 'turbopuffer', 'dragonfly'.",
            other
        )),
    }
}

#[cfg(test)]
mod stats_tests {
    use super::compute_search_stats;

    #[test]
    fn empty_times_errors() {
        assert!(compute_search_stats(&[], &[], &[], &[], &[], 1.0, 10, 1, 0).is_err());
    }

    #[test]
    fn computes_means_rps_and_clamped_percentiles() {
        let times = vec![0.1, 0.2, 0.3, 0.4];
        let ones = vec![1.0, 1.0, 1.0, 1.0];
        let r = compute_search_stats(&times, &ones, &ones, &ones, &ones, 2.0, 10, 4, 5).unwrap();
        assert_eq!(r.num_queries, 4);
        assert_eq!(r.requested_queries, 5);
        assert_eq!(r.failed_queries, 1); // 5 requested, 4 succeeded
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
        let r = compute_search_stats(&[0.5], &[1.0], &[1.0], &[1.0], &[1.0], 1.0, 5, 1, 1).unwrap();
        assert!((r.p99_time - 0.5).abs() < 1e-9);
    }

    #[test]
    fn percentile_linear_matches_numpy() {
        use super::percentile_linear;
        // np.percentile([1..=4], [50,95,99]) with linear interpolation:
        // position = q*(N-1) = q*3.
        let v = [1.0, 2.0, 3.0, 4.0];
        assert!((percentile_linear(&v, 0.50) - 2.5).abs() < 1e-9); // 1.5 -> 2.5
        assert!((percentile_linear(&v, 0.95) - 3.85).abs() < 1e-9); // 2.85 -> 3.85
        assert!((percentile_linear(&v, 0.99) - 3.97).abs() < 1e-9); // 2.97 -> 3.97
                                                                    // Degenerate cases.
        assert_eq!(percentile_linear(&[], 0.5), 0.0);
        assert_eq!(percentile_linear(&[7.0], 0.99), 7.0);
    }

    #[test]
    fn p99_is_not_max_for_n100() {
        use super::percentile_linear;
        // The nearest-rank pathology: with N=100, floor(0.99*100)=99 always
        // returned the single max. Linear gives position 0.99*99=98.01, i.e.
        // just above sorted[98], strictly below the max at sorted[99].
        let sorted: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let p99 = percentile_linear(&sorted, 0.99);
        assert!(p99 < 100.0, "p99={} should be below max", p99);
        assert!(p99 > 99.0, "p99={} should be above sorted[98]", p99);
        assert!((p99 - 99.01).abs() < 1e-9, "p99={}", p99);
    }

    #[test]
    fn filter_mixed_stats_use_linear_percentiles() {
        // The filter-only and mixed harnesses now route their latency samples
        // through compute_search_stats (linear interpolation) instead of the
        // old hand-rolled nearest-rank indexing `(len*q) as usize`, which
        // returned the single max as p99 for N<=100. Feeding a known sample set
        // (1..=100) through the shared path must yield the numpy-linear
        // percentiles, proving the biased method is gone for these harnesses.
        let times: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let r = compute_search_stats(&times, &[], &[], &[], &[], 10.0, 0, 4, 100).unwrap();
        // Nearest-rank would have produced p99 == 100 (the max); linear gives 99.01.
        assert!((r.p99_time - 99.01).abs() < 1e-9, "p99={}", r.p99_time);
        assert!((r.p95_time - 95.05).abs() < 1e-9, "p95={}", r.p95_time);
        assert!((r.p50_time - 50.5).abs() < 1e-9, "p50={}", r.p50_time);
        assert!(r.p99_time < 100.0, "p99={} must be below max", r.p99_time);
    }
}
