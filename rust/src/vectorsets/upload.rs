use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use std::sync::Mutex;
use std::time::Instant;

use indicatif::{HumanCount, ProgressBar, ProgressState, ProgressStyle};

use crate::config::RedisConfig;
use crate::redis_client::create_connection;

#[pyclass]
pub struct RustVsetUploader {
    #[allow(dead_code)]
    host: String,
    #[pyo3(get)]
    connection_params: PyObject,
    #[pyo3(get)]
    upload_params: PyObject,
}

// Module-level state for class methods (mirrors Python's classmethod pattern)
static CLIENT: Mutex<Option<redis::Connection>> = Mutex::new(None);
static CLIENT_DECODE: Mutex<Option<redis::Connection>> = Mutex::new(None);
static UPLOAD_PARAMS: Mutex<Option<UploadConfig>> = Mutex::new(None);
static UPLOADER_HOST: Mutex<Option<String>> = Mutex::new(None);

#[derive(Clone)]
struct UploadConfig {
    m: i64,
    efc: i64,
    quant: String,
}

#[pymethods]
impl RustVsetUploader {
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

        let conn = create_connection(&host, &config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Redis error: {}", e)))?;
        let conn_decode = create_connection(&host, &config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Redis error: {}", e)))?;

        let hnsw_config = upload_params.getattr("__getitem__")?.call1(("hnsw_config",))?;
        let m: i64 = hnsw_config
            .call_method1("get", ("M", 16i64))?
            .extract()?;
        let efc: i64 = hnsw_config
            .call_method1("get", ("EF_CONSTRUCTION", 200i64))?
            .extract()?;
        let quant: String = hnsw_config
            .call_method1("get", ("quant", "NOQUANT"))?
            .extract()?;

        *CLIENT.lock().unwrap() = Some(conn);
        *CLIENT_DECODE.lock().unwrap() = Some(conn_decode);
        *UPLOAD_PARAMS.lock().unwrap() = Some(UploadConfig { m, efc, quant });
        *UPLOADER_HOST.lock().unwrap() = Some(host);
        Ok(())
    }

    #[classmethod]
    fn upload_batch(
        _cls: &Bound<'_, pyo3::types::PyType>,
        ids: &Bound<'_, PyList>,
        vectors: &Bound<'_, PyList>,
        _metadata: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let upload_config = UPLOAD_PARAMS.lock().unwrap().clone().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("init_client not called")
        })?;

        let mut client_guard = CLIENT.lock().unwrap();
        let conn = client_guard.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("init_client not called")
        })?;

        let mut pipe = redis::pipe();

        for i in 0..ids.len() {
            let id: i64 = ids.get_item(i)?.extract()?;
            let vec_list: Vec<f32> = vectors.get_item(i)?.extract()?;
            let vec_bytes: Vec<u8> = vec_list
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();

            pipe.cmd("VADD")
                .arg("idx")
                .arg("FP32")
                .arg(&vec_bytes[..])
                .arg(id)
                .arg(&upload_config.quant)
                .arg("M")
                .arg(upload_config.m)
                .arg("EF")
                .arg(upload_config.efc)
                .arg("CAS")
                .ignore();
        }

        pipe.query::<()>(conn)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Pipeline error: {}", e)))?;

        Ok(())
    }

    #[classmethod]
    fn post_upload(_cls: &Bound<'_, pyo3::types::PyType>, _distance: &Bound<'_, PyAny>) -> PyResult<PyObject> {
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

            Python::with_gil(|py| {
                let dict = PyDict::new_bound(py);
                dict.set_item("used_memory", used_memory)?;
                dict.set_item("shards", 1)?;
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

        // Extract hnsw_config
        let upload_params_bound = self.upload_params.bind(records.py());
        let hnsw_config = upload_params_bound.getattr("__getitem__")?.call1(("hnsw_config",))?;
        let m: i64 = hnsw_config
            .call_method1("get", ("M", 16i64))?
            .extract()?;
        let efc: i64 = hnsw_config
            .call_method1("get", ("EF_CONSTRUCTION", 200i64))?
            .extract()?;
        let quant: String = hnsw_config
            .call_method1("get", ("quant", "NOQUANT"))?
            .extract()?;
        let upload_config = UploadConfig { m, efc, quant };

        // Extract all records from the Python iterator while holding GIL
        let mut all_ids: Vec<i64> = Vec::new();
        let mut all_vectors: Vec<Vec<f32>> = Vec::new();

        let iter = records.call_method0("__iter__")?;
        loop {
            match iter.call_method0("__next__") {
                Ok(record) => {
                    let id: i64 = record.getattr("id")?.extract()?;
                    let vector: Vec<f32> = record.getattr("vector")?.extract()?;
                    all_ids.push(id);
                    all_vectors.push(vector);
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
                        batch_size,
                    )
                } else {
                    run_parallel_upload(
                        &host,
                        &config,
                        &upload_config,
                        &all_ids,
                        &all_vectors,
                        batch_size,
                        parallel as usize,
                    )
                }
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
        })?;

        let upload_time = start.elapsed().as_secs_f64();
        println!("Upload time: {}", upload_time);

        // Store connection for get_memory_usage
        {
            let conn = create_connection(&host, &config)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Redis error: {}", e)))?;
            let conn_decode = create_connection(&host, &config)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Redis error: {}", e)))?;
            *CLIENT.lock().unwrap() = Some(conn);
            *CLIENT_DECODE.lock().unwrap() = Some(conn_decode);
        }

        // post_upload is a no-op for vectorsets
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
            let post_upload = PyDict::new_bound(py);
            dict.set_item("post_upload", post_upload)?;
            dict.set_item("upload_time", upload_time)?;
            dict.set_item("total_time", total_time)?;
            dict.set_item("latencies", latencies)?;
            dict.set_item("parallel", parallel)?;
            dict.set_item("batch_size", batch_size as i64)?;
            dict.set_item("memory_usage", memory_usage)?;
            Ok(dict.into())
        })
    }
}

impl RustVsetUploader {
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

            Python::with_gil(|py| {
                let dict = PyDict::new_bound(py);
                dict.set_item("used_memory", used_memory)?;
                dict.set_item("shards", 1)?;
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
) -> Result<(), String> {
    let mut pipe = redis::pipe();

    for i in 0..ids.len() {
        let id = ids[i];
        let vec_list = &vectors[i];
        let vec_bytes: Vec<u8> = vec_list
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        pipe.cmd("VADD")
            .arg("idx")
            .arg("FP32")
            .arg(&vec_bytes[..])
            .arg(id)
            .arg(&upload_config.quant)
            .arg("M")
            .arg(upload_config.m)
            .arg("EF")
            .arg(upload_config.efc)
            .arg("CAS")
            .ignore();
    }

    pipe.query::<()>(conn)
        .map_err(|e| format!("Pipeline error: {}", e))?;

    Ok(())
}
