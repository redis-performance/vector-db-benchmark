use pyo3::prelude::*;

use crate::config::RedisConfig;
use crate::redis_client::create_connection;

#[pyclass]
pub struct RustVsetConfigurator {
    host: String,
    #[pyo3(get)]
    collection_params: PyObject,
    #[pyo3(get)]
    connection_params: PyObject,
}

#[pymethods]
impl RustVsetConfigurator {
    #[new]
    fn new(
        host: String,
        collection_params: PyObject,
        connection_params: PyObject,
    ) -> Self {
        Self {
            host,
            collection_params,
            connection_params,
        }
    }

    fn clean(&self) -> PyResult<()> {
        let config = RedisConfig::from_env();
        let mut conn = create_connection(&self.host, &config)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Redis error: {}", e)))?;
        redis::cmd("FLUSHALL")
            .query::<()>(&mut conn)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("FLUSHALL error: {}", e)))?;
        Ok(())
    }

    fn recreate(&self, _dataset: &Bound<'_, PyAny>, _collection_params: &Bound<'_, PyAny>) -> PyResult<()> {
        Ok(())
    }

    fn configure(&self, dataset: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        self.clean()?;
        self.recreate(dataset, dataset)?;
        Python::with_gil(|py| Ok(pyo3::types::PyDict::new_bound(py).into()))
    }

    #[pyo3(signature = (**_kwargs))]
    fn execution_params(&self, _kwargs: Option<&Bound<'_, PyAny>>) -> PyResult<PyObject> {
        Python::with_gil(|py| Ok(pyo3::types::PyDict::new_bound(py).into()))
    }

    fn delete_client(&self) -> PyResult<()> {
        Ok(())
    }
}
