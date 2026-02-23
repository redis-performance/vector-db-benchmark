use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use std::sync::Mutex;
use std::time::Duration;

use crate::config::RedisConfig;
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

        let conn = create_connection(&host, &config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Redis error: {}", e)))?;
        let conn_decode = create_connection(&host, &config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Redis error: {}", e)))?;

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

        pipe.query::<()>(conn)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Pipeline error: {}", e)))?;

        Ok(())
    }

    #[classmethod]
    fn post_upload(_cls: &Bound<'_, pyo3::types::PyType>, _distance: &Bound<'_, PyAny>) -> PyResult<PyObject> {
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
            let ft_info: Result<Vec<redis::Value>, _> = redis::cmd("FT.INFO")
                .arg("idx:benchmark")
                .query(conn);

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
}

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
