use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::config::RedisConfig;
use crate::redis_client::create_connection;

#[pyclass]
pub struct RustRedisConfigurator {
    host: String,
    #[pyo3(get)]
    collection_params: PyObject,
    #[pyo3(get)]
    connection_params: PyObject,
}

#[pymethods]
impl RustRedisConfigurator {
    #[new]
    fn new(host: String, collection_params: PyObject, connection_params: PyObject) -> Self {
        Self {
            host,
            collection_params,
            connection_params,
        }
    }

    fn clean(&self) -> PyResult<()> {
        let config = RedisConfig::from_env();
        let mut conn = create_connection(&self.host, &config).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Redis error: {}", e))
        })?;

        // Try FT.DROPINDEX idx:benchmark DD
        let drop_result: Result<(), redis::RedisError> = redis::cmd("FT.DROPINDEX")
            .arg("idx:benchmark")
            .arg("DD")
            .query(&mut conn);

        match drop_result {
            Ok(()) => {}
            Err(e) => {
                let err_str = e.to_string();
                if err_str.contains("Unknown Index name")
                    || err_str.contains("Index does not exist")
                    || err_str.contains("no such index")
                {
                    // Index doesn't exist, that's fine
                } else if err_str.contains("wrong number of arguments for FT.DROPINDEX") {
                    // Memorystore compatibility: fallback to FLUSHALL
                    println!(
                        "Given the FT.DROPINDEX command failed, we're flushing the entire DB..."
                    );
                    redis::cmd("FLUSHALL").query::<()>(&mut conn).map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!("FLUSHALL error: {}", e))
                    })?;
                } else {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "FT.DROPINDEX error: {}",
                        e
                    )));
                }
            }
        }
        Ok(())
    }

    /// Recreate index. `dataset` is a Python Dataset object, `collection_params` is a dict.
    fn recreate(
        &self,
        dataset: &Bound<'_, PyAny>,
        collection_params: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.clean()?;

        let config = RedisConfig::from_env();
        let mut conn = create_connection(&self.host, &config).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Redis error: {}", e))
        })?;

        // Extract dataset info
        let dataset_config = dataset.getattr("config")?;
        let vector_size: i64 = dataset_config.getattr("vector_size")?.extract()?;
        let distance_str: String = dataset_config.getattr("distance")?.extract()?;

        let distance_metric = match distance_str.as_str() {
            "l2" | "L2" => "L2",
            "cosine" | "COSINE" => "COSINE",
            "dot" | "DOT" | "ip" | "IP" => "IP",
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown distance: {}",
                    other
                )))
            }
        };

        // Extract algorithm and config from collection_params
        let algo: String = collection_params
            .call_method1("get", ("algorithm", "hnsw"))?
            .extract()?;
        let data_type: String = collection_params
            .call_method1("get", ("data_type", "float32"))?
            .extract()?;

        let config_key = format!("{}_config", algo);
        let py = collection_params.py();
        let algo_config =
            collection_params.call_method1("get", (config_key.as_str(), PyDict::new_bound(py)))?;

        println!("Using algorithm {} with config {}", algo, algo_config);

        // Build FT.CREATE command
        // FT.CREATE idx:benchmark SCHEMA vector VECTOR <algo> <num_attrs> <attrs...> [payload_fields...]
        let mut cmd = redis::cmd("FT.CREATE");
        cmd.arg("idx:benchmark").arg("SCHEMA");

        // Vector field
        cmd.arg("vector").arg("VECTOR").arg(&algo);

        // Collect vector field attributes
        let mut attrs: Vec<String> = Vec::new();
        attrs.push("TYPE".to_string());
        attrs.push(data_type);
        attrs.push("DIM".to_string());
        attrs.push(vector_size.to_string());
        attrs.push("DISTANCE_METRIC".to_string());
        attrs.push(distance_metric.to_string());

        // Add algorithm-specific config
        let algo_config_dict: &Bound<'_, PyAny> = &algo_config;
        if let Ok(items) = algo_config_dict.call_method0("items") {
            let items_iter = items.call_method0("__iter__")?;
            loop {
                match items_iter.call_method0("__next__") {
                    Ok(item) => {
                        let key: String = item.get_item(0)?.extract()?;
                        let val: String = item.get_item(1)?.str()?.to_string();
                        attrs.push(key);
                        attrs.push(val);
                    }
                    Err(_) => break,
                }
            }
        }

        // Number of attributes (key-value pairs, so total count of items)
        cmd.arg(attrs.len());
        for attr in &attrs {
            cmd.arg(attr.as_str());
        }

        // Add payload/schema fields from dataset
        let schema = dataset_config.getattr("schema")?;
        if !schema.is_none() {
            if let Ok(items) = schema.call_method0("items") {
                let items_iter = items.call_method0("__iter__")?;
                loop {
                    match items_iter.call_method0("__next__") {
                        Ok(item) => {
                            let field_name: String = item.get_item(0)?.extract()?;
                            let field_type: String = item.get_item(1)?.extract()?;

                            cmd.arg(&field_name);
                            match field_type.as_str() {
                                "int" | "float" => {
                                    cmd.arg("NUMERIC").arg("SORTABLE");
                                }
                                "keyword" => {
                                    cmd.arg("TAG").arg("SEPARATOR").arg(";").arg("SORTABLE");
                                }
                                "text" => {
                                    cmd.arg("TEXT").arg("SORTABLE");
                                }
                                "geo" => {
                                    cmd.arg("GEO").arg("SORTABLE");
                                }
                                other => {
                                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                                        "Unknown field type: {}",
                                        other
                                    )));
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
            }
        }

        let create_result: Result<(), redis::RedisError> = cmd.query(&mut conn);
        match create_result {
            Ok(()) => {}
            Err(e) => {
                let err_str = e.to_string();
                if !err_str.contains("Index already exists") {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "FT.CREATE error: {}",
                        e
                    )));
                }
            }
        }

        Ok(())
    }

    fn configure(&self, dataset: &Bound<'_, PyAny>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let collection_params = self.collection_params.bind(py);
            self.recreate(dataset, collection_params)?;
            Ok(PyDict::new_bound(py).into())
        })
    }

    #[pyo3(signature = (**_kwargs))]
    fn execution_params(&self, _kwargs: Option<&Bound<'_, PyAny>>) -> PyResult<PyObject> {
        Python::with_gil(|py| Ok(PyDict::new_bound(py).into()))
    }

    fn delete_client(&self) -> PyResult<()> {
        Ok(())
    }
}
