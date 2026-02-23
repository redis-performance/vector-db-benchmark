use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use std::sync::Mutex;

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
}
