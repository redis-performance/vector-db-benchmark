mod config;
mod redis_client;
mod redisearch;
mod vectorsets;

use pyo3::prelude::*;

use redisearch::configure::RustRedisConfigurator;
use redisearch::search::RustRedisSearcher;
use redisearch::upload::RustRedisUploader;
use vectorsets::configure::RustVsetConfigurator;
use vectorsets::search::RustVsetSearcher;
use vectorsets::upload::RustVsetUploader;

#[pymodule]
fn vector_db_benchmark_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustVsetConfigurator>()?;
    m.add_class::<RustVsetUploader>()?;
    m.add_class::<RustVsetSearcher>()?;
    m.add_class::<RustRedisConfigurator>()?;
    m.add_class::<RustRedisUploader>()?;
    m.add_class::<RustRedisSearcher>()?;
    Ok(())
}
