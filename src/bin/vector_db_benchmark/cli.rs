//! CLI argument parsing for vector-db-benchmark.
//!
//! Mirrors the Python run.py CLI interface.

use clap::Parser;

/// Rust implementation of vector-db-benchmark.
/// Supports redis and vectorsets engines.
#[derive(Parser, Debug, Clone)]
#[command(name = "vector-db-benchmark")]
#[command(version, about = "Run vector database benchmarks", long_about = None)]
pub struct Args {
    /// Engine patterns to run (supports wildcards like "redis*")
    #[arg(long, default_value = "*")]
    pub engines: String,

    /// Path to JSON file containing engine configurations
    #[arg(long)]
    pub engines_file: Option<String>,

    /// Dataset patterns to run (supports wildcards)
    #[arg(long, default_value = "*")]
    pub datasets: String,

    /// Filter by parallel thread counts (comma-separated)
    #[arg(long, value_delimiter = ',')]
    pub parallels: Vec<i32>,

    /// Redis host
    #[arg(long, default_value = "localhost")]
    pub host: String,

    /// Skip upload phase
    #[arg(long, default_value = "false")]
    pub skip_upload: bool,

    /// Skip search phase
    #[arg(long, default_value = "false")]
    pub skip_search: bool,

    /// Skip if results already exist
    #[arg(long, default_value = "true")]
    pub skip_if_exists: bool,

    /// Exit on first error
    #[arg(long, default_value = "true")]
    pub exit_on_error: bool,

    /// Timeout in seconds
    #[arg(long, default_value = "86400.0")]
    pub timeout: f64,

    /// Upload start index
    #[arg(long, default_value = "0")]
    pub upload_start_idx: usize,

    /// Upload end index (-1 means all)
    #[arg(long, default_value = "-1")]
    pub upload_end_idx: i64,

    /// Number of queries to run (-1 means all)
    #[arg(long, default_value = "-1")]
    pub queries: i64,

    /// Filter search experiments by ef runtime values (comma-separated)
    #[arg(long, value_delimiter = ',')]
    pub ef_runtime: Vec<i64>,

    /// Describe available options: 'datasets' or 'engines'
    #[arg(long)]
    pub describe: Option<String>,

    /// Show detailed information when using --describe
    #[arg(long, short)]
    pub verbose: bool,

    /// Mixed benchmark: update-to-search ratio (e.g., "1:10" = 1 update per 10 searches)
    #[arg(long)]
    pub update_search_ratio: Option<String>,
}
