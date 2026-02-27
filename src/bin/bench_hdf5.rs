//! Benchmark HDF5 reading performance in Rust.
//!
//! Usage:
//!     cargo run --release --bin bench_hdf5 [path_to_hdf5] [num_iterations]

use hdf5::File;
use std::env;
use std::time::Instant;

/// Read vectors in bulk as a 2D array, then convert to Vec<Vec<f32>>
fn read_vectors_bulk_to_vec(path: &str) -> (usize, usize) {
    let file = File::open(path).expect("Failed to open HDF5 file");
    let train = file
        .dataset("train")
        .expect("Failed to open 'train' dataset");

    // Get shape first
    let shape = train.shape();
    let count = shape[0];
    let dim = shape[1];

    // Read as flat 1D array
    let flat: Vec<f32> = train.read_raw().expect("Failed to read dataset");

    // Convert to Vec<Vec<f32>> like Python's tolist()
    let _vectors: Vec<Vec<f32>> = flat.chunks(dim).map(|chunk| chunk.to_vec()).collect();

    (count, dim)
}

/// Read vectors in bulk as a flat Vec<f32> (most efficient)
fn read_vectors_bulk_flat(path: &str) -> (usize, usize) {
    let file = File::open(path).expect("Failed to open HDF5 file");
    let train = file
        .dataset("train")
        .expect("Failed to open 'train' dataset");

    // Get shape first
    let shape = train.shape();
    let count = shape[0];
    let dim = shape[1];

    // Read as flat 1D array - most efficient
    let _flat: Vec<f32> = train.read_raw().expect("Failed to read dataset");

    (count, dim)
}

fn benchmark<F>(func: F, path: &str, name: &str, iterations: usize) -> (f64, usize)
where
    F: Fn(&str) -> (usize, usize),
{
    let mut times = Vec::with_capacity(iterations);
    let mut count = 0;
    let mut dim = 0;

    for i in 0..iterations {
        let start = Instant::now();
        let result = func(path);
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

    let avg_time: f64 = times.iter().sum::<f64>() / times.len() as f64;
    let min_time = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let vectors_per_sec = count as f64 / min_time;

    println!("\n{}:", name);
    println!("  Vectors: {:} × {}d", count, dim);
    println!("  Best time: {:.3}s", min_time);
    println!("  Avg time: {:.3}s", avg_time);
    println!("  Throughput: {:.0} vectors/sec", vectors_per_sec);

    (min_time, count)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("datasets/glove-25-angular/glove-25-angular.hdf5");
    let iterations: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(3);

    println!("HDF5 Reading Benchmark (Rust hdf5 crate)");
    println!("{}", "=".repeat(50));
    println!("File: {}", path);
    println!("Iterations: {}", iterations);
    println!();

    // Warmup: just open the file once
    {
        let file = File::open(path).expect("Failed to open HDF5 file");
        let train = file
            .dataset("train")
            .expect("Failed to open 'train' dataset");
        let shape = train.shape();
        println!("Dataset shape: {:?}", shape);
    }
    println!();

    // Benchmark 1: Bulk read (flat Vec<f32>)
    println!("Benchmark 1: Bulk read (flat Vec<f32>)");
    println!("{}", "-".repeat(40));
    let (t1, count) = benchmark(read_vectors_bulk_flat, path, "Bulk flat", iterations);
    println!();

    // Benchmark 2: Bulk read + convert to Vec<Vec<f32>>
    println!("Benchmark 2: Bulk read + convert to Vec<Vec<f32>>");
    println!("{}", "-".repeat(40));
    let (t2, _) = benchmark(
        read_vectors_bulk_to_vec,
        path,
        "Bulk + Vec<Vec>",
        iterations,
    );
    println!();

    // Summary
    println!("{}", "=".repeat(50));
    println!("SUMMARY");
    println!("{}", "=".repeat(50));
    println!(
        "Bulk flat Vec<f32>:    {:.3}s  ({:.0} vec/s)",
        t1,
        count as f64 / t1
    );
    println!(
        "Bulk + Vec<Vec<f32>>:  {:.3}s  ({:.0} vec/s)",
        t2,
        count as f64 / t2
    );
}
