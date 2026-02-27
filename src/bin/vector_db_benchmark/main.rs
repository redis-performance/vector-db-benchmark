//! Vector DB Benchmark - Pure Rust CLI
//!
//! This is the main entry point for the Rust implementation of vector-db-benchmark.
//! Supports redis and vectorsets engines.

mod cli;
mod config;
mod dataset;
mod download;
mod engine;
mod experiment;

use clap::Parser;
use cli::Args;

fn main() {
    let args = Args::parse();

    // Handle --describe option
    if let Some(describe) = &args.describe {
        match describe.as_str() {
            "datasets" => {
                if let Err(e) = config::describe_datasets(args.verbose) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
                return;
            }
            "engines" => {
                if let Err(e) = config::describe_engines(args.verbose) {
                    eprintln!("Error: {}", e);
                    std::process::exit(1);
                }
                return;
            }
            other => {
                eprintln!(
                    "Unknown describe option: '{}'. Use 'datasets' or 'engines'.",
                    other
                );
                std::process::exit(1);
            }
        }
    }

    // Run experiments
    if let Err(e) = experiment::run(&args) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
