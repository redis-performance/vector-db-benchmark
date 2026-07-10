//! Dataset handling and loading.
//!
//! Provides Dataset struct that wraps config and provides data access.

use std::path::PathBuf;

use crate::config::{datasets_dir, DatasetConfig};
use crate::download;
use vector_db_benchmark::readers::metadata::MetadataItem;
use vector_db_benchmark::readers::{
    read_compound_data, read_compound_queries, read_hdf5_vectors, read_jsonl_queries,
    read_jsonl_vectors, read_npy_vectors, read_sparse_matrix, SparseVector,
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
    #[allow(clippy::type_complexity)]
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
            "hdf5" | "h5" => {
                // Explicit HDF5 type — trust it regardless of file extension
                let (ids, vectors) = read_hdf5_vectors(path_str, normalize)?;
                let metadata: Vec<Option<MetadataItem>> = vec![None; vectors.len()];
                Ok((ids, vectors, metadata))
            }
            "" => {
                // No type specified — infer from file extension
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
    #[allow(clippy::type_complexity)]
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
            "hdf5" | "h5" => {
                // Explicit HDF5 type — trust it regardless of file extension
                let (queries, neighbors) = self.read_hdf5_queries(path_str)?;
                let conditions = vec![None; queries.len()];
                Ok((queries, neighbors, conditions))
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

    /// Whether this is a sparse-vector dataset (`dataset_type: "sparse"`).
    pub fn is_sparse(&self) -> bool {
        self.config.dataset_type.as_deref() == Some("sparse")
    }

    /// Read sparse data vectors from `<dir>/data.csr`. Ids are the row indices.
    pub fn read_sparse_data(&self) -> Result<(Vec<i64>, Vec<SparseVector>), String> {
        let dir = self.get_path()?;
        let data = dir.join("data.csr");
        let vectors = read_sparse_matrix(data.to_str().ok_or("Invalid data.csr path")?)?;
        let ids: Vec<i64> = (0..vectors.len() as i64).collect();
        Ok((ids, vectors))
    }

    /// Read sparse queries from `<dir>/queries.csr` and ground-truth neighbours
    /// from `<dir>/neighbours.jsonl` (one JSON array of ids per line).
    pub fn read_sparse_queries(&self) -> Result<(Vec<SparseVector>, Vec<Vec<i64>>), String> {
        let dir = self.get_path()?;
        let queries = read_sparse_matrix(
            dir.join("queries.csr")
                .to_str()
                .ok_or("Invalid queries.csr path")?,
        )?;

        let gt_path = dir.join("neighbours.jsonl");
        let neighbours: Vec<Vec<i64>> = std::fs::read_to_string(&gt_path)
            .map_err(|e| format!("read {}: {}", gt_path.display(), e))?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<Vec<i64>>(l).map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        Ok((queries, neighbours))
    }

    /// Whether this is a hybrid (dense + sparse) dataset (`dataset_type: "hybrid"`).
    ///
    /// A hybrid dataset directory carries BOTH dense npy files (`vectors.npy` /
    /// `queries.npy`) and sparse CSR files (`data.csr` / `queries.csr`), sharing
    /// a single `neighbours.jsonl` ground truth. This lets an engine fuse a dense
    /// prefetch and a sparse prefetch server-side (e.g. Qdrant RRF). It is a
    /// SUPERSET of the sparse layout: same CSR files, plus the dense npy files.
    pub fn is_hybrid(&self) -> bool {
        self.config.dataset_type.as_deref() == Some("hybrid")
    }

    /// Read hybrid upload data: dense vectors from `<dir>/vectors.npy` (reusing
    /// the npy reader) and the row-aligned sparse vectors from `<dir>/data.csr`
    /// (reusing the sparse CSR reader). Ids are the row indices. The dense and
    /// sparse matrices MUST have the same row count — one dense AND one sparse
    /// vector per point.
    #[allow(clippy::type_complexity)]
    pub fn read_hybrid_data(
        &self,
        normalize: bool,
    ) -> Result<(Vec<i64>, Vec<Vec<f32>>, Vec<SparseVector>), String> {
        let dir = self.get_path()?;
        let (_ids, dense_vectors) = read_npy_vectors(
            dir.join("vectors.npy")
                .to_str()
                .ok_or("Invalid vectors.npy path")?,
            normalize,
        )?;
        let sparse = read_sparse_matrix(
            dir.join("data.csr")
                .to_str()
                .ok_or("Invalid data.csr path")?,
        )?;
        if dense_vectors.len() != sparse.len() {
            return Err(format!(
                "hybrid data row mismatch: {} dense rows vs {} sparse rows",
                dense_vectors.len(),
                sparse.len()
            ));
        }
        let ids: Vec<i64> = (0..dense_vectors.len() as i64).collect();
        Ok((ids, dense_vectors, sparse))
    }

    /// Read hybrid queries: dense queries from `<dir>/queries.npy`, the
    /// row-aligned sparse queries from `<dir>/queries.csr`, and shared
    /// ground-truth neighbours from `<dir>/neighbours.jsonl` (one JSON array of
    /// ids per line). The ground truth is shared because it describes the FUSED
    /// result, not either modality alone.
    #[allow(clippy::type_complexity)]
    pub fn read_hybrid_queries(
        &self,
    ) -> Result<(Vec<Vec<f32>>, Vec<SparseVector>, Vec<Vec<i64>>), String> {
        let dir = self.get_path()?;
        let normalize = self.needs_normalization();
        let (_ids, dense_queries) = read_npy_vectors(
            dir.join("queries.npy")
                .to_str()
                .ok_or("Invalid queries.npy path")?,
            normalize,
        )?;
        let sparse_queries = read_sparse_matrix(
            dir.join("queries.csr")
                .to_str()
                .ok_or("Invalid queries.csr path")?,
        )?;
        if dense_queries.len() != sparse_queries.len() {
            return Err(format!(
                "hybrid query row mismatch: {} dense vs {} sparse",
                dense_queries.len(),
                sparse_queries.len()
            ));
        }

        let gt_path = dir.join("neighbours.jsonl");
        let neighbours: Vec<Vec<i64>> = std::fs::read_to_string(&gt_path)
            .map_err(|e| format!("read {}: {}", gt_path.display(), e))?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<Vec<i64>>(l).map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        Ok((dense_queries, sparse_queries, neighbours))
    }

    /// Read queries from HDF5 file (test + neighbors datasets).
    #[allow(clippy::type_complexity)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::DatasetConfig;
    use vector_db_benchmark::readers::{write_npy_vectors, write_sparse_matrix};

    /// Build a `Dataset` whose `path` is an absolute temp dir (so `get_path`
    /// resolves to it directly — `datasets_dir().join(abs)` == `abs`).
    fn hybrid_dataset(dir: &std::path::Path) -> Dataset {
        Dataset::new(DatasetConfig {
            name: "hybrid-unit".to_string(),
            dataset_type: Some("hybrid".to_string()),
            path: serde_json::Value::String(dir.to_str().unwrap().to_string()),
            distance: Some("l2".to_string()),
            vector_size: Some(3),
            vector_count: Some(2),
            link: None,
            schema: None,
            description: None,
        })
    }

    #[test]
    fn is_hybrid_only_for_hybrid_type() {
        let dir = tempfile::tempdir().unwrap();
        let ds = hybrid_dataset(dir.path());
        assert!(ds.is_hybrid());
        assert!(!ds.is_sparse());

        let mut cfg = ds.config.clone();
        cfg.dataset_type = Some("sparse".to_string());
        let sparse = Dataset::new(cfg);
        assert!(!sparse.is_hybrid());
        assert!(sparse.is_sparse());
    }

    #[test]
    fn reads_hybrid_data_and_queries() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();

        let dense = vec![vec![1.0f32, 0.0, 0.0], vec![0.0, 2.0, 0.0]];
        let sparse = vec![
            SparseVector {
                indices: vec![0, 2],
                values: vec![1.0, 3.0],
            },
            SparseVector {
                indices: vec![1],
                values: vec![5.0],
            },
        ];
        write_npy_vectors(p.join("vectors.npy").to_str().unwrap(), &dense).unwrap();
        write_sparse_matrix(p.join("data.csr").to_str().unwrap(), &sparse).unwrap();

        let dense_q = vec![vec![1.0f32, 1.0, 1.0]];
        let sparse_q = vec![SparseVector {
            indices: vec![0],
            values: vec![2.0],
        }];
        write_npy_vectors(p.join("queries.npy").to_str().unwrap(), &dense_q).unwrap();
        write_sparse_matrix(p.join("queries.csr").to_str().unwrap(), &sparse_q).unwrap();
        std::fs::write(p.join("neighbours.jsonl"), "[0, 1]\n").unwrap();

        let ds = hybrid_dataset(p);
        let (ids, d, s) = ds.read_hybrid_data(false).unwrap();
        assert_eq!(ids, vec![0, 1]);
        assert_eq!(d, dense);
        assert_eq!(s, sparse);

        let (dq, sq, nb) = ds.read_hybrid_queries().unwrap();
        assert_eq!(dq, dense_q);
        assert_eq!(sq, sparse_q);
        assert_eq!(nb, vec![vec![0i64, 1]]);
    }

    #[test]
    fn read_hybrid_data_rejects_row_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        // 2 dense rows but only 1 sparse row → must error.
        write_npy_vectors(
            p.join("vectors.npy").to_str().unwrap(),
            &[vec![1.0f32, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
        )
        .unwrap();
        write_sparse_matrix(
            p.join("data.csr").to_str().unwrap(),
            &[SparseVector {
                indices: vec![0],
                values: vec![1.0],
            }],
        )
        .unwrap();
        let ds = hybrid_dataset(p);
        assert!(ds.read_hybrid_data(false).is_err());
    }
}
