use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use std::sync::Mutex;
use std::time::{Duration, Instant};

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};

use crate::config::RedisConfig;
use crate::readers::{
    extract_metadata_item, read_compound_data, read_hdf5_vectors, read_jsonl_vectors,
    read_npy_vectors, MetadataItem, MetadataValue,
};
use crate::redis_client::create_connection;

#[pyclass]
pub struct RustRedisUploader {
    #[allow(dead_code)]
    host: String,
    #[pyo3(get)]
    connection_params: PyObject,
    #[pyo3(get)]
    upload_params: PyObject,
}

// Module-level state for class methods
static CLIENT: Mutex<Option<redis::Connection>> = Mutex::new(None);
static CLIENT_DECODE: Mutex<Option<redis::Connection>> = Mutex::new(None);
static UPLOAD_PARAMS: Mutex<Option<UploadConfig>> = Mutex::new(None);
static UPLOADER_HOST: Mutex<Option<String>> = Mutex::new(None);

#[derive(Clone)]
struct UploadConfig {
    #[allow(dead_code)]
    algorithm: String,
    data_type: String,
}

#[pymethods]
impl RustRedisUploader {
    #[new]
    fn new(host: String, connection_params: PyObject, upload_params: PyObject) -> Self {
        Self {
            host,
            connection_params,
            upload_params,
        }
    }

    #[classmethod]
    fn init_client(
        _cls: &Bound<'_, pyo3::types::PyType>,
        host: String,
        _distance: &Bound<'_, PyAny>,
        _connection_params: &Bound<'_, PyAny>,
        upload_params: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let config = RedisConfig::from_env();

        let conn = create_connection(&host, &config).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Redis error: {}", e))
        })?;
        let conn_decode = create_connection(&host, &config).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Redis error: {}", e))
        })?;

        let algorithm: String = upload_params
            .call_method1("get", ("algorithm", "hnsw"))?
            .extract()?;
        let data_type: String = upload_params
            .call_method1("get", ("data_type", "FLOAT32"))?
            .extract()?;

        *CLIENT.lock().unwrap() = Some(conn);
        *CLIENT_DECODE.lock().unwrap() = Some(conn_decode);
        *UPLOAD_PARAMS.lock().unwrap() = Some(UploadConfig {
            algorithm: algorithm.to_uppercase(),
            data_type: data_type.to_uppercase(),
        });
        *UPLOADER_HOST.lock().unwrap() = Some(host);
        Ok(())
    }

    #[classmethod]
    fn upload_batch(
        _cls: &Bound<'_, pyo3::types::PyType>,
        ids: &Bound<'_, PyList>,
        vectors: &Bound<'_, PyList>,
        metadata: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let upload_config =
            UPLOAD_PARAMS.lock().unwrap().clone().ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("init_client not called")
            })?;

        let mut client_guard = CLIENT.lock().unwrap();
        let conn = client_guard
            .as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("init_client not called"))?;

        let mut pipe = redis::pipe();

        for i in 0..ids.len() {
            let id: i64 = ids.get_item(i)?.extract()?;
            let key = id.to_string();

            // Convert vector to bytes based on data_type
            let vec_list: Vec<f32> = vectors.get_item(i)?.extract()?;
            let vec_bytes: Vec<u8> = match upload_config.data_type.as_str() {
                "FLOAT64" => {
                    let f64_vec: Vec<f64> = vec_list.iter().map(|&f| f as f64).collect();
                    f64_vec.iter().flat_map(|f| f.to_le_bytes()).collect()
                }
                "FLOAT16" => {
                    let f16_vec: Vec<u16> = vec_list
                        .iter()
                        .map(|&f| half::f16::from_f32(f).to_bits())
                        .collect();
                    f16_vec.iter().flat_map(|v| v.to_le_bytes()).collect()
                }
                _ => {
                    // Default FLOAT32
                    vec_list.iter().flat_map(|f| f.to_le_bytes()).collect()
                }
            };

            // Build field-value pairs for HSET
            let mut fields: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
            fields.push(("vector".as_bytes().to_vec(), vec_bytes));

            // Extract metadata
            let meta = if metadata.is_none() {
                None
            } else {
                let meta_list: &Bound<'_, PyAny> = metadata;
                match meta_list.get_item(i) {
                    Ok(m) if !m.is_none() => Some(m),
                    _ => None,
                }
            };

            if let Some(meta_obj) = meta {
                if let Ok(items) = meta_obj.call_method0("items") {
                    let iter = items.call_method0("__iter__")?;
                    loop {
                        match iter.call_method0("__next__") {
                            Ok(kv) => {
                                let k: String = kv.get_item(0)?.extract()?;
                                let v = kv.get_item(1)?;

                                if v.is_none() {
                                    continue;
                                }

                                // Handle "labels" field (list -> semicolon-separated)
                                if k == "labels" {
                                    if let Ok(label_list) = v.extract::<Vec<String>>() {
                                        fields.push((
                                            k.as_bytes().to_vec(),
                                            label_list.join(";").into_bytes(),
                                        ));
                                    }
                                    continue;
                                }

                                // Handle geopoints (dict with "lon" and "lat")
                                if v.is_instance_of::<pyo3::types::PyDict>() {
                                    let lon = v
                                        .get_item("lon")
                                        .and_then(|l| l.extract::<f64>())
                                        .unwrap_or(0.0);
                                    let lat = v
                                        .get_item("lat")
                                        .and_then(|l| l.extract::<f64>())
                                        .unwrap_or(0.0);
                                    // Clamp latitude
                                    let lat = lat.clamp(-85.05112878, 85.05112878);
                                    let geo_str = format!("{},{}", lon, lat);
                                    fields.push((k.as_bytes().to_vec(), geo_str.into_bytes()));
                                    continue;
                                }

                                // Skip lists (except labels handled above)
                                if v.is_instance_of::<pyo3::types::PyList>() {
                                    continue;
                                }

                                // Scalar values (int, float, string)
                                let val_str: String = v.str()?.to_string();
                                fields.push((k.as_bytes().to_vec(), val_str.into_bytes()));
                            }
                            Err(_) => break,
                        }
                    }
                }
            }

            // Build HSET command in pipeline
            let mut hset_cmd = redis::cmd("HSET");
            hset_cmd.arg(key.as_str());
            for (field_key, field_val) in &fields {
                hset_cmd.arg(&field_key[..]).arg(&field_val[..]);
            }
            pipe.add_command(hset_cmd).ignore();
        }

        pipe.query::<()>(conn).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Pipeline error: {}", e))
        })?;

        Ok(())
    }

    #[classmethod]
    fn post_upload(
        _cls: &Bound<'_, pyo3::types::PyType>,
        _distance: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        // Wait for indexing to complete
        let mut client_guard = CLIENT.lock().unwrap();
        if let Some(conn) = client_guard.as_mut() {
            loop {
                let info: Vec<redis::Value> = redis::cmd("FT.INFO")
                    .arg("idx:benchmark")
                    .query(conn)
                    .unwrap_or_default();

                let percent = extract_ft_info_field(&info, "percent_indexed");
                if let Some(pct) = percent {
                    if pct >= 1.0 {
                        break;
                    }
                    println!(
                        "waiting for index to be fully processed. current percent index: {}",
                        pct * 100.0
                    );
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }

                let lag = extract_ft_info_field(&info, "current_lag");
                if let Some(lag_val) = lag {
                    if lag_val <= 0.0 {
                        break;
                    }
                    println!(
                        "waiting for index to be fully processed. current current_lag: {}",
                        lag_val
                    );
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }

                // Neither field found, assume done
                break;
            }
        }
        Python::with_gil(|py| Ok(PyDict::new_bound(py).into()))
    }

    #[classmethod]
    fn get_memory_usage(_cls: &Bound<'_, pyo3::types::PyType>) -> PyResult<PyObject> {
        let mut client_guard = CLIENT_DECODE.lock().unwrap();
        if let Some(conn) = client_guard.as_mut() {
            let info: String = redis::cmd("INFO")
                .arg("memory")
                .query(conn)
                .unwrap_or_default();

            let used_memory = info
                .lines()
                .find(|l| l.starts_with("used_memory:"))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|v| v.trim().parse::<i64>().ok())
                .unwrap_or(0);

            // Get FT.INFO as well
            let ft_info: Result<Vec<redis::Value>, _> =
                redis::cmd("FT.INFO").arg("idx:benchmark").query(conn);

            Python::with_gil(|py| {
                let dict = PyDict::new_bound(py);
                let mem_list = pyo3::types::PyList::new_bound(py, &[used_memory]);
                dict.set_item("used_memory", mem_list)?;

                // Pass ft info as empty dict for now (matching Python structure)
                let index_info = PyDict::new_bound(py);
                if let Ok(info_values) = ft_info {
                    // Convert FT.INFO flat array to dict
                    let mut i = 0;
                    while i + 1 < info_values.len() {
                        if let redis::Value::BulkString(key_bytes) = &info_values[i] {
                            let key = String::from_utf8_lossy(key_bytes);
                            match &info_values[i + 1] {
                                redis::Value::BulkString(val_bytes) => {
                                    let val = String::from_utf8_lossy(val_bytes);
                                    index_info.set_item(key.as_ref(), val.as_ref())?;
                                }
                                redis::Value::Int(n) => {
                                    index_info.set_item(key.as_ref(), *n)?;
                                }
                                _ => {}
                            }
                        }
                        i += 2;
                    }
                }
                dict.set_item("index_info", index_info)?;
                dict.set_item("device_info", PyDict::new_bound(py))?;
                Ok(dict.into())
            })
        } else {
            Python::with_gil(|py| Ok(PyDict::new_bound(py).into()))
        }
    }

    #[classmethod]
    fn delete_client(_cls: &Bound<'_, pyo3::types::PyType>) -> PyResult<()> {
        *CLIENT.lock().unwrap() = None;
        *CLIENT_DECODE.lock().unwrap() = None;
        *UPLOAD_PARAMS.lock().unwrap() = None;
        *UPLOADER_HOST.lock().unwrap() = None;
        Ok(())
    }

    /// Runs the entire upload loop in Rust, including batching and parallelism.
    /// This replaces the Python BaseUploader.upload() method.
    fn upload_all(
        &self,
        _distance: &Bound<'_, PyAny>,
        records: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        let parallel: i64 = self
            .upload_params
            .bind(records.py())
            .call_method1("get", ("parallel", 1i64))?
            .extract()?;
        let batch_size: usize = self
            .upload_params
            .bind(records.py())
            .call_method1("get", ("batch_size", 64i64))?
            .extract()?;

        let host = self.host.clone();
        let config = RedisConfig::from_env();
        let upload_params_bound = self.upload_params.bind(records.py());
        let algorithm: String = upload_params_bound
            .call_method1("get", ("algorithm", "hnsw"))?
            .extract()?;
        let data_type: String = upload_params_bound
            .call_method1("get", ("data_type", "FLOAT32"))?
            .extract()?;
        let upload_config = UploadConfig {
            algorithm: algorithm.to_uppercase(),
            data_type: data_type.to_uppercase(),
        };

        // Extract all records from the Python iterator while holding GIL
        let mut all_ids: Vec<i64> = Vec::new();
        let mut all_vectors: Vec<Vec<f32>> = Vec::new();
        let mut all_metadata: Vec<Option<MetadataItem>> = Vec::new();

        let iter = records.call_method0("__iter__")?;
        loop {
            match iter.call_method0("__next__") {
                Ok(record) => {
                    let id: i64 = record.getattr("id")?.extract()?;
                    let vector: Vec<f32> = record.getattr("vector")?.extract()?;
                    let meta_obj = record.getattr("metadata")?;
                    let metadata = extract_metadata_item(&meta_obj)?;
                    all_ids.push(id);
                    all_vectors.push(vector);
                    all_metadata.push(metadata);
                }
                Err(_) => break, // StopIteration
            }
        }

        let total_records = all_ids.len();
        println!("Extracted {} records, starting upload...", total_records);

        // Measure wall-clock time for upload
        let start = Instant::now();

        // Release GIL and run the upload loop in Rust
        let latencies: Vec<f64> = Python::with_gil(|py| -> PyResult<Vec<f64>> {
            py.allow_threads(|| {
                if parallel <= 1 {
                    run_sequential_upload(
                        &host,
                        &config,
                        &upload_config,
                        &all_ids,
                        &all_vectors,
                        &all_metadata,
                        batch_size,
                    )
                } else {
                    run_parallel_upload(
                        &host,
                        &config,
                        &upload_config,
                        &all_ids,
                        &all_vectors,
                        &all_metadata,
                        batch_size,
                        parallel as usize,
                    )
                }
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        })?;

        let upload_time = start.elapsed().as_secs_f64();
        println!("Upload time: {}", upload_time);

        // Store connection for post_upload and get_memory_usage
        {
            let conn = create_connection(&host, &config).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Redis error: {}", e))
            })?;
            let conn_decode = create_connection(&host, &config).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Redis error: {}", e))
            })?;
            *CLIENT.lock().unwrap() = Some(conn);
            *CLIENT_DECODE.lock().unwrap() = Some(conn_decode);
        }

        // Call post_upload
        let post_upload_stats = Self::post_upload_internal()?;
        let total_time = start.elapsed().as_secs_f64();
        let docs_per_sec = total_records as f64 / total_time;
        println!("Total import time: {:.3}s", total_time);
        println!("Ingested docs/sec: {:.0}", docs_per_sec);

        // Get memory usage
        let memory_usage = Self::get_memory_usage_internal()?;

        // Clean up
        Self::delete_client_internal();

        // Return stats dict
        Python::with_gil(|py| {
            let dict = PyDict::new_bound(py);
            dict.set_item("post_upload", post_upload_stats)?;
            dict.set_item("upload_time", upload_time)?;
            dict.set_item("total_time", total_time)?;
            dict.set_item("latencies", latencies)?;
            dict.set_item("parallel", parallel)?;
            dict.set_item("batch_size", batch_size as i64)?;
            dict.set_item("memory_usage", memory_usage)?;
            Ok(dict.into())
        })
    }

    /// Runs the entire upload loop in Rust, reading vectors directly from HDF5 file.
    /// This eliminates Python from the hot path entirely.
    fn upload_from_hdf5(
        &self,
        py: Python<'_>,
        _distance: &Bound<'_, PyAny>,
        hdf5_path: String,
        normalize: bool,
    ) -> PyResult<PyObject> {
        let parallel: i64 = self
            .upload_params
            .bind(py)
            .call_method1("get", ("parallel", 1i64))?
            .extract()?;
        let batch_size: usize = self
            .upload_params
            .bind(py)
            .call_method1("get", ("batch_size", 64i64))?
            .extract()?;

        let host = self.host.clone();
        let config = RedisConfig::from_env();
        let upload_params_bound = self.upload_params.bind(py);
        let algorithm: String = upload_params_bound
            .call_method1("get", ("algorithm", "hnsw"))?
            .extract()?;
        let data_type: String = upload_params_bound
            .call_method1("get", ("data_type", "FLOAT32"))?
            .extract()?;
        let upload_config = UploadConfig {
            algorithm: algorithm.to_uppercase(),
            data_type: data_type.to_uppercase(),
        };

        // Read HDF5 file in Rust (no GIL needed)
        println!("Reading HDF5 file: {}", hdf5_path);
        let read_start = Instant::now();

        let (all_ids, all_vectors) = py
            .allow_threads(|| read_hdf5_vectors(&hdf5_path, normalize))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("HDF5 error: {}", e)))?;

        let read_time = read_start.elapsed().as_secs_f64();
        let total_records = all_vectors.len();
        println!(
            "Read {} vectors in {:.3}s ({:.0} vectors/sec)",
            total_records,
            read_time,
            total_records as f64 / read_time
        );

        // No metadata for HDF5 files (AnnH5Reader always yields None for metadata)
        let all_metadata: Vec<Option<MetadataItem>> = vec![None; total_records];

        println!("Starting upload...");

        // Measure wall-clock time for upload
        let start = Instant::now();

        // Release GIL and run the upload loop in Rust
        let latencies: Vec<f64> = py
            .allow_threads(|| {
                if parallel <= 1 {
                    run_sequential_upload(
                        &host,
                        &config,
                        &upload_config,
                        &all_ids,
                        &all_vectors,
                        &all_metadata,
                        batch_size,
                    )
                } else {
                    run_parallel_upload(
                        &host,
                        &config,
                        &upload_config,
                        &all_ids,
                        &all_vectors,
                        &all_metadata,
                        batch_size,
                        parallel as usize,
                    )
                }
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

        let upload_time = start.elapsed().as_secs_f64();
        println!("Upload time: {:.3}s", upload_time);

        // Store connection for post_upload and get_memory_usage
        {
            let conn = create_connection(&host, &config).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Redis error: {}", e))
            })?;
            let conn_decode = create_connection(&host, &config).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Redis error: {}", e))
            })?;
            *CLIENT.lock().unwrap() = Some(conn);
            *CLIENT_DECODE.lock().unwrap() = Some(conn_decode);
        }

        // Call post_upload
        let post_upload_stats = Self::post_upload_internal()?;
        let total_time = read_start.elapsed().as_secs_f64(); // Include HDF5 read time
        let docs_per_sec = total_records as f64 / total_time;
        println!("Total import time: {:.3}s", total_time);
        println!("Ingested docs/sec: {:.0}", docs_per_sec);

        // Get memory usage
        let memory_usage = Self::get_memory_usage_internal()?;

        // Clean up
        Self::delete_client_internal();

        // Return stats dict
        let dict = PyDict::new_bound(py);
        dict.set_item("post_upload", post_upload_stats)?;
        dict.set_item("upload_time", upload_time)?;
        dict.set_item("total_time", total_time)?;
        dict.set_item("read_time", read_time)?;
        dict.set_item("latencies", latencies)?;
        dict.set_item("parallel", parallel)?;
        dict.set_item("batch_size", batch_size as i64)?;
        dict.set_item("memory_usage", memory_usage)?;
        Ok(dict.into())
    }

    /// Runs the entire upload loop in Rust, reading vectors directly from JSONL file.
    fn upload_from_jsonl(
        &self,
        py: Python<'_>,
        _distance: &Bound<'_, PyAny>,
        jsonl_path: String,
        normalize: bool,
    ) -> PyResult<PyObject> {
        self.upload_from_file_internal(py, &jsonl_path, normalize, "jsonl")
    }

    /// Runs the entire upload loop in Rust, reading vectors directly from NPY file.
    fn upload_from_npy(
        &self,
        py: Python<'_>,
        _distance: &Bound<'_, PyAny>,
        npy_path: String,
        normalize: bool,
    ) -> PyResult<PyObject> {
        self.upload_from_file_internal(py, &npy_path, normalize, "npy")
    }

    /// Internal method for file-based upload (shared by HDF5, JSONL, NPY)
    fn upload_from_file_internal(
        &self,
        py: Python<'_>,
        file_path: &str,
        normalize: bool,
        file_type: &str,
    ) -> PyResult<PyObject> {
        let parallel: i64 = self
            .upload_params
            .bind(py)
            .call_method1("get", ("parallel", 1i64))?
            .extract()?;
        let batch_size: usize = self
            .upload_params
            .bind(py)
            .call_method1("get", ("batch_size", 64i64))?
            .extract()?;

        let host = self.host.clone();
        let config = RedisConfig::from_env();
        let upload_params_bound = self.upload_params.bind(py);
        let algorithm: String = upload_params_bound
            .call_method1("get", ("algorithm", "hnsw"))?
            .extract()?;
        let data_type: String = upload_params_bound
            .call_method1("get", ("data_type", "FLOAT32"))?
            .extract()?;
        let upload_config = UploadConfig {
            algorithm: algorithm.to_uppercase(),
            data_type: data_type.to_uppercase(),
        };

        // Read file in Rust (no GIL needed)
        let file_type_upper = file_type.to_uppercase();
        println!("Reading {} file: {}", file_type_upper, file_path);
        let read_start = Instant::now();

        let file_path_owned = file_path.to_string();
        let file_type_owned = file_type.to_string();

        let (all_ids, all_vectors) = py
            .allow_threads(|| match file_type_owned.as_str() {
                "jsonl" => read_jsonl_vectors(&file_path_owned, normalize),
                "npy" => read_npy_vectors(&file_path_owned, normalize),
                "hdf5" => read_hdf5_vectors(&file_path_owned, normalize),
                _ => Err(format!("Unknown file type: {}", file_type_owned)),
            })
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "{} error: {}",
                    file_type_upper, e
                ))
            })?;

        let read_time = read_start.elapsed().as_secs_f64();
        let total_records = all_vectors.len();
        let dim = all_vectors.first().map(|v| v.len()).unwrap_or(0);
        println!(
            "Read {} vectors ({}d) in {:.3}s ({:.0} vectors/sec)",
            total_records,
            dim,
            read_time,
            total_records as f64 / read_time
        );

        // No metadata for file-based uploads
        let all_metadata: Vec<Option<MetadataItem>> = vec![None; total_records];

        println!("Starting upload...");

        // Measure wall-clock time for upload
        let start = Instant::now();

        // Release GIL and run the upload loop in Rust
        let parallel_usize = parallel as usize;
        let latencies: Vec<f64> = py
            .allow_threads(|| {
                if parallel <= 1 {
                    run_sequential_upload(
                        &host,
                        &config,
                        &upload_config,
                        &all_ids,
                        &all_vectors,
                        &all_metadata,
                        batch_size,
                    )
                } else {
                    run_parallel_upload(
                        &host,
                        &config,
                        &upload_config,
                        &all_ids,
                        &all_vectors,
                        &all_metadata,
                        batch_size,
                        parallel_usize,
                    )
                }
            })
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Upload error: {}", e))
            })?;

        let upload_time = start.elapsed().as_secs_f64();
        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            total_records as f64 / upload_time
        );

        // Post-upload (back in Python context for Redis info)
        let post_upload_stats = Self::post_upload_internal()?;

        let total_time = start.elapsed().as_secs_f64();
        println!("Total import time: {:.3}s", total_time);

        // Get memory usage
        let memory_usage = Self::get_memory_usage_internal()?;

        // Clean up
        Self::delete_client_internal();

        // Return stats dict
        let dict = PyDict::new_bound(py);
        dict.set_item("post_upload", post_upload_stats)?;
        dict.set_item("upload_time", upload_time)?;
        dict.set_item("total_time", total_time)?;
        dict.set_item("read_time", read_time)?;
        dict.set_item("latencies", latencies)?;
        dict.set_item("parallel", parallel)?;
        dict.set_item("batch_size", batch_size as i64)?;
        dict.set_item("memory_usage", memory_usage)?;
        Ok(dict.into())
    }

    /// Runs the entire upload loop in Rust, reading from compound format directory.
    /// Compound format has vectors.npy for vectors and payloads.jsonl for metadata.
    fn upload_from_compound(
        &self,
        py: Python<'_>,
        _distance: &Bound<'_, PyAny>,
        dir_path: String,
        normalize: bool,
    ) -> PyResult<PyObject> {
        let parallel: i64 = self
            .upload_params
            .bind(py)
            .call_method1("get", ("parallel", 1i64))?
            .extract()?;
        let batch_size: usize = self
            .upload_params
            .bind(py)
            .call_method1("get", ("batch_size", 64i64))?
            .extract()?;

        let host = self.host.clone();
        let config = RedisConfig::from_env();
        let upload_params_bound = self.upload_params.bind(py);
        let algorithm: String = upload_params_bound
            .call_method1("get", ("algorithm", "hnsw"))?
            .extract()?;
        let data_type: String = upload_params_bound
            .call_method1("get", ("data_type", "FLOAT32"))?
            .extract()?;
        let upload_config = UploadConfig {
            algorithm: algorithm.to_uppercase(),
            data_type: data_type.to_uppercase(),
        };

        // Read compound data in Rust (no GIL needed)
        println!("Reading compound data from: {}", dir_path);
        let read_start = Instant::now();

        let dir_path_owned = dir_path.clone();
        let (all_ids, all_vectors, all_metadata) = py
            .allow_threads(|| read_compound_data(&dir_path_owned, normalize))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Compound read error: {}", e))
            })?;

        let read_time = read_start.elapsed().as_secs_f64();
        let total_records = all_vectors.len();
        let dim = all_vectors.first().map(|v| v.len()).unwrap_or(0);
        let metadata_count = all_metadata.iter().filter(|m| m.is_some()).count();
        println!(
            "Read {} vectors ({}d) with {} metadata records in {:.3}s ({:.0} vectors/sec)",
            total_records,
            dim,
            metadata_count,
            read_time,
            total_records as f64 / read_time
        );

        println!("Starting upload...");

        // Measure wall-clock time for upload
        let start = Instant::now();

        // Release GIL and run the upload loop in Rust
        let parallel_usize = parallel as usize;
        let latencies: Vec<f64> = py
            .allow_threads(|| {
                if parallel <= 1 {
                    run_sequential_upload(
                        &host,
                        &config,
                        &upload_config,
                        &all_ids,
                        &all_vectors,
                        &all_metadata,
                        batch_size,
                    )
                } else {
                    run_parallel_upload(
                        &host,
                        &config,
                        &upload_config,
                        &all_ids,
                        &all_vectors,
                        &all_metadata,
                        batch_size,
                        parallel_usize,
                    )
                }
            })
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Upload error: {}", e))
            })?;

        let upload_time = start.elapsed().as_secs_f64();
        println!(
            "Upload time: {:.3}s ({:.0} records/sec)",
            upload_time,
            total_records as f64 / upload_time
        );

        // Post-upload (back in Python context for Redis info)
        let post_upload_stats = Self::post_upload_internal()?;

        let total_time = start.elapsed().as_secs_f64();
        println!("Total import time: {:.3}s", total_time);

        // Get memory usage
        let memory_usage = Self::get_memory_usage_internal()?;

        // Clean up
        Self::delete_client_internal();

        // Return stats dict
        let dict = PyDict::new_bound(py);
        dict.set_item("post_upload", post_upload_stats)?;
        dict.set_item("upload_time", upload_time)?;
        dict.set_item("total_time", total_time)?;
        dict.set_item("read_time", read_time)?;
        dict.set_item("latencies", latencies)?;
        dict.set_item("parallel", parallel)?;
        dict.set_item("batch_size", batch_size as i64)?;
        dict.set_item("memory_usage", memory_usage)?;
        Ok(dict.into())
    }
}

// Reader functions moved to crate::readers module

/// Extract a named field from FT.INFO flat key-value response.
fn extract_ft_info_field(info: &[redis::Value], field_name: &str) -> Option<f64> {
    let mut i = 0;
    while i + 1 < info.len() {
        if let redis::Value::BulkString(key_bytes) = &info[i] {
            let key = String::from_utf8_lossy(key_bytes);
            if key == field_name {
                return match &info[i + 1] {
                    redis::Value::BulkString(val_bytes) => {
                        let val = String::from_utf8_lossy(val_bytes);
                        val.trim().parse::<f64>().ok()
                    }
                    redis::Value::Int(n) => Some(*n as f64),
                    redis::Value::Double(f) => Some(*f),
                    _ => None,
                };
            }
        }
        i += 2;
    }
    None
}

impl RustRedisUploader {
    /// Internal post_upload without classmethod decorator
    fn post_upload_internal() -> PyResult<PyObject> {
        let mut client_guard = CLIENT.lock().unwrap();
        if let Some(conn) = client_guard.as_mut() {
            loop {
                let info: Vec<redis::Value> = redis::cmd("FT.INFO")
                    .arg("idx:benchmark")
                    .query(conn)
                    .unwrap_or_default();

                let percent = extract_ft_info_field(&info, "percent_indexed");
                if let Some(pct) = percent {
                    if pct >= 1.0 {
                        break;
                    }
                    println!(
                        "waiting for index to be fully processed. current percent index: {}",
                        pct * 100.0
                    );
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }

                let lag = extract_ft_info_field(&info, "current_lag");
                if let Some(lag_val) = lag {
                    if lag_val <= 0.0 {
                        break;
                    }
                    println!(
                        "waiting for index to be fully processed. current current_lag: {}",
                        lag_val
                    );
                    std::thread::sleep(Duration::from_secs(1));
                    continue;
                }

                break;
            }
        }
        Python::with_gil(|py| Ok(PyDict::new_bound(py).into()))
    }

    /// Internal get_memory_usage without classmethod decorator
    fn get_memory_usage_internal() -> PyResult<PyObject> {
        let mut client_guard = CLIENT_DECODE.lock().unwrap();
        if let Some(conn) = client_guard.as_mut() {
            let info: String = redis::cmd("INFO")
                .arg("memory")
                .query(conn)
                .unwrap_or_default();

            let used_memory = info
                .lines()
                .find(|l| l.starts_with("used_memory:"))
                .and_then(|l| l.split(':').nth(1))
                .and_then(|v| v.trim().parse::<i64>().ok())
                .unwrap_or(0);

            let ft_info: Result<Vec<redis::Value>, _> =
                redis::cmd("FT.INFO").arg("idx:benchmark").query(conn);

            Python::with_gil(|py| {
                let dict = PyDict::new_bound(py);
                let mem_list = pyo3::types::PyList::new_bound(py, &[used_memory]);
                dict.set_item("used_memory", mem_list)?;

                let index_info = PyDict::new_bound(py);
                if let Ok(info_values) = ft_info {
                    let mut i = 0;
                    while i + 1 < info_values.len() {
                        if let redis::Value::BulkString(key_bytes) = &info_values[i] {
                            let key = String::from_utf8_lossy(key_bytes);
                            match &info_values[i + 1] {
                                redis::Value::BulkString(val_bytes) => {
                                    let val = String::from_utf8_lossy(val_bytes);
                                    index_info.set_item(key.as_ref(), val.as_ref())?;
                                }
                                redis::Value::Int(n) => {
                                    index_info.set_item(key.as_ref(), *n)?;
                                }
                                _ => {}
                            }
                        }
                        i += 2;
                    }
                }
                dict.set_item("index_info", index_info)?;
                dict.set_item("device_info", PyDict::new_bound(py))?;
                Ok(dict.into())
            })
        } else {
            Python::with_gil(|py| Ok(PyDict::new_bound(py).into()))
        }
    }

    /// Internal delete_client without classmethod decorator
    fn delete_client_internal() {
        *CLIENT.lock().unwrap() = None;
        *CLIENT_DECODE.lock().unwrap() = None;
        *UPLOAD_PARAMS.lock().unwrap() = None;
        *UPLOADER_HOST.lock().unwrap() = None;
    }
}

/// Run sequential upload (single connection)
fn run_sequential_upload(
    host: &str,
    config: &RedisConfig,
    upload_config: &UploadConfig,
    all_ids: &[i64],
    all_vectors: &[Vec<f32>],
    all_metadata: &[Option<MetadataItem>],
    batch_size: usize,
) -> Result<Vec<f64>, String> {
    let mut conn = create_connection(host, config).map_err(|e| e.to_string())?;
    let mut latencies = Vec::new();

    // Create progress bar
    let pb = ProgressBar::new(all_ids.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec_int}/s)")
            .unwrap()
            .with_key("per_sec_int", |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                write!(w, "{}", HumanCount(state.per_sec() as u64)).unwrap()
            })
            .progress_chars("#>-"),
    );

    for batch_start in (0..all_ids.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(all_ids.len());
        let start = Instant::now();

        upload_batch_internal(
            &mut conn,
            upload_config,
            &all_ids[batch_start..batch_end],
            &all_vectors[batch_start..batch_end],
            &all_metadata[batch_start..batch_end],
        )?;

        latencies.push(start.elapsed().as_secs_f64());
        pb.inc((batch_end - batch_start) as u64);
    }

    pb.finish_with_message("Upload complete");
    Ok(latencies)
}

/// Run parallel upload using thread::scope
fn run_parallel_upload(
    host: &str,
    config: &RedisConfig,
    upload_config: &UploadConfig,
    all_ids: &[i64],
    all_vectors: &[Vec<f32>],
    all_metadata: &[Option<MetadataItem>],
    batch_size: usize,
    parallel: usize,
) -> Result<Vec<f64>, String> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // Create progress bar
    let pb = ProgressBar::new(all_ids.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec_int}/s)")
            .unwrap()
            .with_key("per_sec_int", |state: &ProgressState, w: &mut dyn std::fmt::Write| {
                write!(w, "{}", HumanCount(state.per_sec() as u64)).unwrap()
            })
            .progress_chars("#>-"),
    );

    // Create batches
    let mut batches: Vec<(usize, usize)> = Vec::new();
    for batch_start in (0..all_ids.len()).step_by(batch_size) {
        let batch_end = (batch_start + batch_size).min(all_ids.len());
        batches.push((batch_start, batch_end));
    }

    let total_batches = batches.len();
    let latencies: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(vec![0.0; total_batches]));
    let batch_idx = Arc::new(AtomicUsize::new(0));
    let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

    std::thread::scope(|s| {
        for _ in 0..parallel {
            let host = host.to_string();
            let config = config.clone();
            let upload_config = upload_config.clone();
            let batches = &batches;
            let batch_idx = Arc::clone(&batch_idx);
            let latencies = Arc::clone(&latencies);
            let error = Arc::clone(&error);
            let pb = &pb;

            s.spawn(move || {
                // Each thread gets its own connection
                let mut conn = match create_connection(&host, &config) {
                    Ok(c) => c,
                    Err(e) => {
                        *error.lock().unwrap() = Some(e.to_string());
                        return;
                    }
                };

                loop {
                    let idx = batch_idx.fetch_add(1, Ordering::SeqCst);
                    if idx >= total_batches {
                        break;
                    }

                    let (batch_start, batch_end) = batches[idx];
                    let start = Instant::now();

                    if let Err(e) = upload_batch_internal(
                        &mut conn,
                        &upload_config,
                        &all_ids[batch_start..batch_end],
                        &all_vectors[batch_start..batch_end],
                        &all_metadata[batch_start..batch_end],
                    ) {
                        *error.lock().unwrap() = Some(e);
                        break;
                    }

                    latencies.lock().unwrap()[idx] = start.elapsed().as_secs_f64();
                    pb.inc((batch_end - batch_start) as u64);
                }
            });
        }
    });

    pb.finish_with_message("Upload complete");

    // Check for errors
    if let Some(err) = error.lock().unwrap().take() {
        return Err(err);
    }

    Ok(Arc::try_unwrap(latencies)
        .map_err(|_| "Failed to unwrap latencies Arc")?
        .into_inner()
        .unwrap())
}

/// Internal batch upload function that works without GIL
fn upload_batch_internal(
    conn: &mut redis::Connection,
    upload_config: &UploadConfig,
    ids: &[i64],
    vectors: &[Vec<f32>],
    metadata: &[Option<MetadataItem>],
) -> Result<(), String> {
    let mut pipe = redis::pipe();

    for i in 0..ids.len() {
        let id = ids[i];
        let key = id.to_string();
        let vec_list = &vectors[i];

        // Convert vector to bytes based on data_type
        let vec_bytes: Vec<u8> = match upload_config.data_type.as_str() {
            "FLOAT64" => {
                let f64_vec: Vec<f64> = vec_list.iter().map(|&f| f as f64).collect();
                f64_vec.iter().flat_map(|f| f.to_le_bytes()).collect()
            }
            "FLOAT16" => {
                let f16_vec: Vec<u16> = vec_list
                    .iter()
                    .map(|&f| half::f16::from_f32(f).to_bits())
                    .collect();
                f16_vec.iter().flat_map(|v| v.to_le_bytes()).collect()
            }
            _ => {
                // Default FLOAT32
                vec_list.iter().flat_map(|f| f.to_le_bytes()).collect()
            }
        };

        // Build field-value pairs for HSET
        let mut fields: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        fields.push(("vector".as_bytes().to_vec(), vec_bytes));

        // Add metadata fields
        if let Some(meta) = &metadata[i] {
            for (k, v) in &meta.fields {
                match v {
                    MetadataValue::String(s) => {
                        fields.push((k.as_bytes().to_vec(), s.as_bytes().to_vec()));
                    }
                    MetadataValue::Int(n) => {
                        fields.push((k.as_bytes().to_vec(), n.to_string().into_bytes()));
                    }
                    MetadataValue::Float(f) => {
                        fields.push((k.as_bytes().to_vec(), f.to_string().into_bytes()));
                    }
                    MetadataValue::Labels(labels) => {
                        fields.push((k.as_bytes().to_vec(), labels.join(";").into_bytes()));
                    }
                    MetadataValue::Geo { lon, lat } => {
                        let lat_clamped = lat.clamp(-85.05112878, 85.05112878);
                        let geo_str = format!("{},{}", lon, lat_clamped);
                        fields.push((k.as_bytes().to_vec(), geo_str.into_bytes()));
                    }
                }
            }
        }

        // Build HSET command in pipeline
        let mut hset_cmd = redis::cmd("HSET");
        hset_cmd.arg(key.as_str());
        for (field_key, field_val) in &fields {
            hset_cmd.arg(&field_key[..]).arg(&field_val[..]);
        }
        pipe.add_command(hset_cmd).ignore();
    }

    pipe.query::<()>(conn)
        .map_err(|e| format!("Pipeline error: {}", e))?;

    Ok(())
}
