#!/usr/bin/env rust
//! Benchmark JSONL reading performance in Rust using serde_json.
//!
//! Usage:
//!     cargo run --release --bin bench_jsonl -- <path_to_jsonl> [num_iterations]

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

/// Read vectors line by line (like Python's current approach)
fn read_vectors_iterate(path: &str, normalize: bool) -> (usize, usize) {
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);

    let mut count = 0;
    let mut dim = 0;

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let vector: Vec<f32> = serde_json::from_str(&line).expect("Failed to parse JSON");

        let mut vec = vector;
        if normalize {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in vec.iter_mut() {
                    *x /= norm;
                }
            }
        }

        dim = vec.len();
        count += 1;
    }

    (count, dim)
}

/// Read all vectors into memory at once
fn read_vectors_bulk(path: &str, normalize: bool) -> (usize, usize) {
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);

    let mut vectors: Vec<Vec<f32>> = Vec::new();

    for line in reader.lines() {
        let line = line.expect("Failed to read line");
        let mut vector: Vec<f32> = serde_json::from_str(&line).expect("Failed to parse JSON");

        if normalize {
            let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in vector.iter_mut() {
                    *x /= norm;
                }
            }
        }

        vectors.push(vector);
    }

    let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
    (vectors.len(), dim)
}

fn benchmark<F>(func: F, path: &str, name: &str, iterations: usize, normalize: bool)
where
    F: Fn(&str, bool) -> (usize, usize),
{
    let mut times = Vec::new();
    let mut count = 0;
    let mut dim = 0;

    for i in 0..iterations {
        let start = Instant::now();
        let result = func(path, normalize);
        let elapsed = start.elapsed().as_secs_f64();
        count = result.0;
        dim = result.1;
        times.push(elapsed);
        println!(
            "  Run {}: {:.3}s ({:.0} vectors/sec)",
            i + 1,
            elapsed,
            count as f64 / elapsed
        );
    }

    let min_time = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let avg_time: f64 = times.iter().sum::<f64>() / times.len() as f64;
    let throughput = count as f64 / min_time;

    println!("\n{}:", name);
    println!("  Vectors: {} × {}d", count, dim);
    println!("  Best time: {:.3}s", min_time);
    println!("  Avg time: {:.3}s", avg_time);
    println!("  Throughput: {:.0} vectors/sec", throughput);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("datasets/random-100k/vectors.jsonl");
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("======================================================================");
    println!("JSONL Reading Benchmark (Rust serde_json)");
    println!("======================================================================");
    println!("File: {}", path);
    println!("Iterations: {}", iterations);
    println!();

    // Benchmark 1: Line by line (like Python)
    println!("Benchmark 1: Line-by-line iterate");
    println!("--------------------------------------------------");
    benchmark(
        read_vectors_iterate,
        path,
        "Line-by-line",
        iterations,
        false,
    );
    println!();

    // Benchmark 2: Line by line with normalize
    println!("Benchmark 2: Line-by-line + normalize");
    println!("--------------------------------------------------");
    benchmark(
        read_vectors_iterate,
        path,
        "Line-by-line + normalize",
        iterations,
        true,
    );
    println!();

    // Benchmark 3: Bulk read with normalize
    println!("Benchmark 3: Bulk read + normalize");
    println!("--------------------------------------------------");
    benchmark(
        read_vectors_bulk,
        path,
        "Bulk + normalize",
        iterations,
        true,
    );
}
