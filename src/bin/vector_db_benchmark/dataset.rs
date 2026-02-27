//! Dataset handling and loading.
//!
//! Provides Dataset struct that wraps config and provides data access.

use std::path::PathBuf;

use crate::config::{datasets_dir, project_root, DatasetConfig};
use crate::download;
use vector_db_benchmark::readers::metadata::MetadataItem;
use vector_db_benchmark::readers::{
    read_compound_data, read_compound_queries, read_hdf5_vectors, read_jsonl_queries,
    read_jsonl_vectors, read_npy_vectors,
};

/// Dataset wrapper that provides access to vectors and metadata
pub struct Dataset {
    pub config: DatasetConfig,
}

impl Dataset {
    pub fn new(config: DatasetConfig) -> Self {
        Self { config }
    }

    /// Get the file path for this dataset
    pub fn get_path(&self) -> Result<PathBuf, String> {
        if let Some(path_str) = self.config.path.as_str() {
            // Check in datasets/ directory
            let datasets_path = datasets_dir().join(path_str);
            if datasets_path.exists() {
                return Ok(datasets_path);
            }
            // Not found locally — try downloading if link is available
            if let Some(link) = &self.config.link {
                let target_path = datasets_dir().join(path_str);
                download::download_dataset(link, &target_path)?;
                // Re-check after download
                if target_path.exists() {
                    return Ok(target_path);
                }
                Err(format!(
                    "Downloaded from {} but path still not found: {}",
                    link, path_str
                ))
            } else {
                Err(format!(
                    "Dataset path not found and no download link: {} (tried {:?})",
                    path_str, datasets_path
                ))
            }
        } else if let Some(path_obj) = self.config.path.as_object() {
            // For dict-style paths (like h5-multi)
            if let Some(data_files) = path_obj.get("data").and_then(|d| d.as_array()) {
                if let Some(first) = data_files.first() {
                    if let Some(p) = first.get("path").and_then(|p| p.as_str()) {
                        let datasets_path = datasets_dir().join(p);
                        if datasets_path.exists() {
                            return Ok(datasets_path);
                        }
                    }
                }
            }
            Err("Could not resolve dict-style path".to_string())
        } else {
            Err("Dataset path is not a string or object".to_string())
        }
    }

    /// Get the distance metric for this dataset
    pub fn distance(&self) -> &str {
        self.config.distance.as_deref().unwrap_or("cosine")
    }

    /// Get vector dimensions
    pub fn vector_size(&self) -> i64 {
        self.config.vector_size.unwrap_or(128)
    }

    /// Check if normalization is needed (for angular/cosine distance)
    pub fn needs_normalization(&self) -> bool {
        matches!(
            self.distance().to_lowercase().as_str(),
            "angular" | "cosine"
        )
    }

    /// Read all vectors and metadata from the dataset
    pub fn read_vectors(
        &self,
        normalize: bool,
    ) -> Result<(Vec<i64>, Vec<Vec<f32>>, Vec<Option<MetadataItem>>), String> {
        let path = self.get_path()?;
        let path_str = path.to_str().ok_or("Invalid path encoding")?;
        let dataset_type = self.config.dataset_type.as_deref().unwrap_or("");

        match dataset_type {
            "tar" => {
                // Compound format (vectors.npy + payloads.jsonl)
                read_compound_data(path_str, normalize)
            }
            "hdf5" | "h5" | "" => {
                // HDF5 format - check file extension
                let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
                match ext.to_lowercase().as_str() {
                    "hdf5" | "h5" => {
                        let (ids, vectors) = read_hdf5_vectors(path_str, normalize)?;
                        let metadata: Vec<Option<MetadataItem>> = vec![None; vectors.len()];
                        Ok((ids, vectors, metadata))
                    }
                    "jsonl" => {
                        let (ids, vectors) = read_jsonl_vectors(path_str, normalize)?;
                        let metadata: Vec<Option<MetadataItem>> = vec![None; vectors.len()];
                        Ok((ids, vectors, metadata))
                    }
                    "npy" => {
                        let (ids, vectors) = read_npy_vectors(path_str, normalize)?;
                        let metadata: Vec<Option<MetadataItem>> = vec![None; vectors.len()];
                        Ok((ids, vectors, metadata))
                    }
                    _ => Err(format!("Unsupported file extension: {}", ext)),
                }
            }
            "jsonl" => {
                // JSONL path is a directory containing vectors.jsonl
                let vectors_file = if path.is_dir() {
                    path.join("vectors.jsonl")
                } else {
                    path.clone()
                };
                let vectors_str = vectors_file.to_str().ok_or("Invalid vectors.jsonl path")?;
                let (ids, vectors) = read_jsonl_vectors(vectors_str, normalize)?;
                let metadata: Vec<Option<MetadataItem>> = vec![None; vectors.len()];
                Ok((ids, vectors, metadata))
            }
            other => Err(format!("Unsupported dataset type: {}", other)),
        }
    }

    /// Read query vectors, ground truth neighbors, and filter conditions from the dataset.
    /// Returns (queries, neighbors, conditions) where conditions is per-query filter JSON.
    pub fn read_queries(
        &self,
    ) -> Result<(Vec<Vec<f32>>, Vec<Vec<i64>>, Vec<Option<serde_json::Value>>), String> {
        let path = self.get_path()?;
        let path_str = path.to_str().ok_or("Invalid path encoding")?;
        let dataset_type = self.config.dataset_type.as_deref().unwrap_or("");
        let normalize = self.needs_normalization();

        match dataset_type {
            "tar" => {
                // Compound format: tests.jsonl includes conditions
                read_compound_queries(path_str, normalize)
            }
            "jsonl" => {
                let dir = if path.is_dir() {
                    path.clone()
                } else {
                    path.parent()
                        .ok_or_else(|| "Cannot get parent dir of JSONL path".to_string())?
                        .to_path_buf()
                };
                let (queries, neighbors) = read_jsonl_queries(
                    dir.to_str().ok_or("Invalid dir path encoding")?,
                    normalize,
                )?;
                let conditions = vec![None; queries.len()];
                Ok((queries, neighbors, conditions))
            }
            _ => {
                if path_str.ends_with(".hdf5") || path_str.ends_with(".h5") {
                    let (queries, neighbors) = self.read_hdf5_queries(path_str)?;
                    let conditions = vec![None; queries.len()];
                    Ok((queries, neighbors, conditions))
                } else if path.is_dir() {
                    let tests_path = path.join("tests.jsonl");
                    let queries_path = path.join("queries.jsonl");
                    if tests_path.exists() {
                        read_compound_queries(path_str, normalize)
                    } else if queries_path.exists() {
                        let (queries, neighbors) = read_jsonl_queries(path_str, normalize)?;
                        let conditions = vec![None; queries.len()];
                        Ok((queries, neighbors, conditions))
                    } else {
                        Err(format!("No query files found in directory: {}", path_str))
                    }
                } else {
                    Err(format!(
                        "Query reading not supported for dataset type '{}' at path: {}",
                        dataset_type, path_str
                    ))
                }
            }
        }
    }

    /// Read queries from HDF5 file (test + neighbors datasets).
    fn read_hdf5_queries(&self, path_str: &str) -> Result<(Vec<Vec<f32>>, Vec<Vec<i64>>), String> {
        let file =
            hdf5::File::open(path_str).map_err(|e| format!("Failed to open HDF5 file: {}", e))?;

        // Read test vectors
        let test_ds = file
            .dataset("test")
            .map_err(|e| format!("Failed to read 'test' dataset: {}", e))?;
        let shape = test_ds.shape();
        if shape.len() != 2 {
            return Err("Expected 2D test dataset".to_string());
        }
        let num_queries = shape[0];
        let dims = shape[1];
        let flat_data: Vec<f32> = test_ds
            .read_raw()
            .map_err(|e| format!("Failed to read test data: {}", e))?;
        let queries: Vec<Vec<f32>> = flat_data
            .chunks(dims)
            .take(num_queries)
            .map(|chunk| chunk.to_vec())
            .collect();

        // Read neighbors (ground truth)
        let neighbors_ds = file
            .dataset("neighbors")
            .map_err(|e| format!("Failed to read 'neighbors' dataset: {}", e))?;
        let shape = neighbors_ds.shape();
        if shape.len() != 2 {
            return Err("Expected 2D neighbors dataset".to_string());
        }
        let num_neighbors = shape[0];
        let k = shape[1];
        let flat_neighbors: Vec<i64> = neighbors_ds
            .read_raw()
            .map_err(|e| format!("Failed to read neighbors data: {}", e))?;
        let neighbors: Vec<Vec<i64>> = flat_neighbors
            .chunks(k)
            .take(num_neighbors)
            .map(|chunk| chunk.to_vec())
            .collect();

        Ok((queries, neighbors))
    }
}
