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
    /// Engine patterns to run (supports wildcards like "redis*", repeatable)
    #[arg(long, default_value = "*")]
    pub engines: Vec<String>,

    /// Path to JSON file containing engine configurations
    #[arg(long)]
    pub engines_file: Option<String>,

    /// Dataset patterns to run (supports wildcards, repeatable)
    #[arg(long, default_value = "*")]
    pub datasets: Vec<String>,

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

    /// Keep the configured index and uploaded data after the experiment
    #[arg(long, default_value = "false")]
    pub keep_data: bool,

    /// Skip if results already exist (accepts an optional value, e.g.
    /// `--skip-if-exists false`; bare `--skip-if-exists` means true)
    #[arg(
        long,
        num_args = 0..=1,
        default_value_t = true,
        default_missing_value = "true",
        action = clap::ArgAction::Set
    )]
    pub skip_if_exists: bool,

    /// Exit on first error (accepts an optional value, e.g.
    /// `--exit-on-error false`; bare `--exit-on-error` means true)
    #[arg(
        long,
        num_args = 0..=1,
        default_value_t = true,
        default_missing_value = "true",
        action = clap::ArgAction::Set
    )]
    pub exit_on_error: bool,

    /// Overall wall-clock budget in seconds: stop launching new experiments once
    /// total elapsed exceeds this (any in-flight experiment finishes). 0 disables.
    #[arg(long, default_value = "86400.0")]
    pub timeout: f64,

    /// Per-search-point wall-clock watchdog in seconds. A single search/mixed
    /// call that runs longer than this (e.g. a proxy/connection-pool stall at
    /// high `parallel`) is aborted with a diagnostic instead of hanging the whole
    /// sweep silently. Progress is logged while a point is in flight. 0 disables
    /// (default) — behavior is then unchanged.
    #[arg(long, default_value = "0.0")]
    pub search_timeout: f64,

    /// Upload start index
    #[arg(long, default_value = "0")]
    pub upload_start_idx: usize,

    /// Upload end index (-1 means all)
    #[arg(long, default_value = "-1")]
    pub upload_end_idx: i64,

    /// Number of queries to run (-1 means all)
    #[arg(long, default_value = "-1")]
    pub queries: i64,

    /// Fixed offered query rate. 0 keeps the existing closed-loop behavior.
    /// Open-loop mode is currently supported by Redis and Vertex.
    #[arg(long, default_value = "0.0")]
    pub target_qps: f64,

    /// Measured search duration in seconds. With --target-qps this is open-loop;
    /// without it Redis and Vertex run unrestricted closed-loop for this long.
    #[arg(long, default_value = "0.0")]
    pub search_duration: f64,

    /// Warm-up duration before each measured open-loop search configuration.
    #[arg(long, default_value = "0.0")]
    pub warmup_seconds: f64,

    /// Drop an open-loop request when dispatch is this many milliseconds late.
    #[arg(long, default_value = "1000.0")]
    pub max_lateness_ms: f64,

    /// Repeat each measured search config this many times and report the
    /// best-RPS run (warm best-of, matching v0's REPETITIONS). The first run is
    /// often cold (OS page cache / index warm-up); best-of discards it. Set 1 to
    /// disable. Also honored via the REPETITIONS environment variable.
    #[arg(long, env = "REPETITIONS", default_value = "3")]
    pub repetitions: usize,

    /// Filter search experiments by ef runtime values (comma-separated)
    #[arg(long, value_delimiter = ',')]
    pub ef_runtime: Vec<i64>,

    /// Describe available options: 'datasets' or 'engines'
    #[arg(long)]
    pub describe: Option<String>,

    /// Instead of benchmarking, render a QPS-vs-precision trade-off chart (SVG)
    /// from existing `*-summary.json` files in results/, filtered by --engines
    /// and --datasets. One colored series per engine. Value is the output path.
    #[arg(long, value_name = "OUTPUT.svg")]
    pub plot: Option<String>,

    /// Show detailed information when using --describe
    #[arg(long, short)]
    pub verbose: bool,

    /// Mixed benchmark: update-to-search ratio (e.g., "1:10" = 1 update per 10 searches).
    /// Can be specified multiple times. "0:S" means pure search.
    #[arg(long)]
    pub update_search_ratio: Vec<String>,

    /// Skip vector indexing: upload vectors but don't index them, run filter-only queries.
    /// Collapses all M/EF variants of the same engine into a single "<engine>-no-vector" experiment.
    #[arg(long, default_value = "false")]
    pub skip_vector_index: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(extra: &[&str]) -> Args {
        let mut argv = vec!["vector-db-benchmark", "--engines", "x", "--datasets", "y"];
        argv.extend_from_slice(extra);
        Args::try_parse_from(argv).expect("args should parse")
    }

    // Regression: the default-true bool flags must accept an explicit value
    // (`--skip-if-exists false`, as the mongodb integration test invokes it),
    // an `=` value, a bare flag, and default to true when omitted. Previously
    // they were SetTrue flags that rejected any value and could never be false.
    #[test]
    fn skip_if_exists_accepts_all_forms() {
        assert!(parse(&[]).skip_if_exists, "omitted → true");
        assert!(parse(&["--skip-if-exists"]).skip_if_exists, "bare → true");
        assert!(
            !parse(&["--skip-if-exists", "false"]).skip_if_exists,
            "space value → false"
        );
        assert!(
            !parse(&["--skip-if-exists=false"]).skip_if_exists,
            "= value → false"
        );
        assert!(parse(&["--skip-if-exists", "true"]).skip_if_exists);
    }

    #[test]
    fn exit_on_error_accepts_all_forms() {
        assert!(parse(&[]).exit_on_error, "omitted → true");
        assert!(parse(&["--exit-on-error"]).exit_on_error, "bare → true");
        assert!(!parse(&["--exit-on-error", "false"]).exit_on_error);
        assert!(!parse(&["--exit-on-error=false"]).exit_on_error);
    }

    #[test]
    fn parses_open_loop_options() {
        let args = parse(&[
            "--target-qps",
            "1500",
            "--search-duration",
            "300",
            "--warmup-seconds",
            "10",
            "--max-lateness-ms",
            "250",
        ]);
        assert_eq!(args.target_qps, 1500.0);
        assert_eq!(args.search_duration, 300.0);
        assert_eq!(args.warmup_seconds, 10.0);
        assert_eq!(args.max_lateness_ms, 250.0);
    }

    // Per-search watchdog (#151-5): opt-in, defaults to 0.0 (disabled) so the
    // unchanged behavior is preserved, and parses an explicit value.
    #[test]
    fn search_timeout_parses() {
        assert_eq!(parse(&[]).search_timeout, 0.0, "omitted → disabled");
        assert_eq!(parse(&["--search-timeout", "300"]).search_timeout, 300.0);
    }

    // `--describe datasets|engines` is what the docker-build smoke test exercises;
    // pin that it parses (and is absent by default) so the dispatch in main.rs
    // always receives the expected value.
    #[test]
    fn describe_option_parses() {
        assert_eq!(parse(&[]).describe, None, "omitted → None");
        assert_eq!(
            parse(&["--describe", "datasets"]).describe.as_deref(),
            Some("datasets")
        );
        assert_eq!(
            parse(&["--describe", "engines"]).describe.as_deref(),
            Some("engines")
        );
    }
}
