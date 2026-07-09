#!/usr/bin/env rust
//! Benchmark NPY reading performance in Rust using ndarray-npy.
//!
//! Usage:
//!     cargo run --release --bin bench_npy -- <path_to_npy> [num_iterations]

use std::env;
use std::fs::File;
use std::io::BufReader;
use std::time::Instant;

use ndarray::Array2;
use ndarray_npy::ReadNpyExt;

/// Read vectors from NPY file and convert to Vec<Vec<f32>>
fn read_vectors_bulk(path: &str, normalize: bool) -> (usize, usize) {
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);

    let arr: Array2<f32> = Array2::read_npy(reader).expect("Failed to read NPY file");
    let (rows, cols) = arr.dim();

    // Convert to Vec<Vec<f32>> (like the upload code needs)
    let mut vectors: Vec<Vec<f32>> = arr.rows().into_iter().map(|row| row.to_vec()).collect();

    if normalize {
        for vec in vectors.iter_mut() {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in vec.iter_mut() {
                    *x /= norm;
                }
            }
        }
    }

    (rows, cols)
}

/// Read vectors and iterate row by row (simulating Python's for vector in np.load())
fn read_vectors_iterate(path: &str, normalize: bool) -> (usize, usize) {
    let file = File::open(path).expect("Failed to open file");
    let reader = BufReader::new(file);

    let arr: Array2<f32> = Array2::read_npy(reader).expect("Failed to read NPY file");
    let (_rows, cols) = arr.dim();

    let mut count = 0;
    for row in arr.rows() {
        let mut vec: Vec<f32> = row.to_vec();

        if normalize {
            let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in vec.iter_mut() {
                    *x /= norm;
                }
            }
        }

        count += 1;
        let _ = vec; // Simulate using the vector
    }

    (count, cols)
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
    let default_path = "/home/fco/.local/lib/python3.12/site-packages/datasets/h-and-m-2048-angular/hnm/vectors.npy";
    let path = args.get(1).map(|s| s.as_str()).unwrap_or(default_path);
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("======================================================================");
    println!("NPY Reading Benchmark (Rust ndarray-npy)");
    println!("======================================================================");
    println!("File: {}", path);
    println!("Iterations: {}", iterations);
    println!();

    // Benchmark 1: Row-by-row iterate
    println!("Benchmark 1: Row-by-row iterate");
    println!("--------------------------------------------------");
    benchmark(read_vectors_iterate, path, "Row-by-row", iterations, false);
    println!();

    // Benchmark 2: Row-by-row + normalize
    println!("Benchmark 2: Row-by-row + normalize");
    println!("--------------------------------------------------");
    benchmark(
        read_vectors_iterate,
        path,
        "Row-by-row + normalize",
        iterations,
        true,
    );
    println!();

    // Benchmark 3: Bulk read + normalize
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
