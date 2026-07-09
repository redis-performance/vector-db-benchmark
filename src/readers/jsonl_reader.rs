//! JSONL file reader for vector datasets.
//!
//! Supports:
//! - vectors.jsonl: one JSON array of floats per line
//! - queries.jsonl: one JSON array of floats per line (query vectors)
//! - neighbours.jsonl: one JSON array of ints per line (ground truth neighbor IDs)

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Read vectors from JSONL file, returning (ids, vectors).
/// Each line should be a JSON array of floats.
/// If normalize is true, each vector is divided by its L2 norm.
pub fn read_jsonl_vectors(
    path: &str,
    normalize: bool,
) -> Result<(Vec<i64>, Vec<Vec<f32>>), String> {
    let file = File::open(path).map_err(|e| format!("Failed to open JSONL file: {}", e))?;
    let reader = BufReader::new(file);

    let mut vectors: Vec<Vec<f32>> = Vec::new();

    for line_result in reader.lines() {
        let line = line_result.map_err(|e| format!("Failed to read line: {}", e))?;
        if line.trim().is_empty() {
            continue;
        }

        let vec_f64: Vec<f64> =
            serde_json::from_str(&line).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let mut vector: Vec<f32> = vec_f64.iter().map(|&x| x as f32).collect();

        if normalize {
            normalize_vector(&mut vector);
        }

        vectors.push(vector);
    }

    let count = vectors.len();
    let ids: Vec<i64> = (0..count as i64).collect();

    Ok((ids, vectors))
}

/// Read queries and ground truth neighbors from a JSONL dataset directory.
/// Expects: queries.jsonl (required) and neighbours.jsonl (optional).
/// Returns (query_vectors, neighbors).
#[allow(clippy::type_complexity)]
pub fn read_jsonl_queries(
    dir_path: &str,
    normalize: bool,
) -> Result<(Vec<Vec<f32>>, Vec<Vec<i64>>), String> {
    let dir = Path::new(dir_path);

    // Read query vectors
    let queries_path = dir.join("queries.jsonl");
    let queries_file =
        File::open(&queries_path).map_err(|e| format!("Failed to open queries.jsonl: {}", e))?;
    let reader = BufReader::new(queries_file);

    let mut queries: Vec<Vec<f32>> = Vec::new();
    for line_result in reader.lines() {
        let line = line_result.map_err(|e| format!("Failed to read query line: {}", e))?;
        if line.trim().is_empty() {
            continue;
        }

        let vec_f64: Vec<f64> = serde_json::from_str(&line)
            .map_err(|e| format!("Failed to parse query JSON: {}", e))?;
        let mut vector: Vec<f32> = vec_f64.iter().map(|&x| x as f32).collect();

        if normalize {
            normalize_vector(&mut vector);
        }

        queries.push(vector);
    }

    // Read ground truth neighbors (optional)
    let neighbours_path = dir.join("neighbours.jsonl");
    let neighbors = if neighbours_path.exists() {
        let file = File::open(&neighbours_path)
            .map_err(|e| format!("Failed to open neighbours.jsonl: {}", e))?;
        let reader = BufReader::new(file);
        let mut result: Vec<Vec<i64>> = Vec::new();
        for line_result in reader.lines() {
            let line = line_result.map_err(|e| format!("Failed to read neighbour line: {}", e))?;
            if line.trim().is_empty() {
                continue;
            }
            let ids: Vec<i64> = serde_json::from_str(&line)
                .map_err(|e| format!("Failed to parse neighbours JSON: {}", e))?;
            result.push(ids);
        }
        result
    } else {
        // No ground truth available
        vec![vec![]; queries.len()]
    };

    Ok((queries, neighbors))
}

fn normalize_vector(vector: &mut [f32]) {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vector.iter_mut() {
            *x /= norm;
        }
    }
}
