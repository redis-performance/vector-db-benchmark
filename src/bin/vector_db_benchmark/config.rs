//! Configuration loading for datasets and engines.
//!
//! Reads datasets.json and experiments/configurations/*.json

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// Dataset configuration from datasets.json
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DatasetConfig {
    pub name: String,
    #[serde(rename = "type")]
    pub dataset_type: Option<String>,
    pub path: serde_json::Value,
    pub distance: Option<String>,
    pub vector_size: Option<i64>,
    pub vector_count: Option<i64>,
    pub link: Option<String>,
    pub schema: Option<serde_json::Value>,
    pub description: Option<String>,
}

/// HNSW configuration
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct HnswConfig {
    #[serde(rename = "M", alias = "m")]
    pub m: Option<i64>,
    #[serde(rename = "EF_CONSTRUCTION", alias = "ef_construct", alias = "ef_construction")]
    pub ef_construction: Option<i64>,
}

/// Elasticsearch index_options (lowercase keys)
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct IndexOptions {
    pub m: Option<i64>,
    pub ef_construction: Option<i64>,
}

/// Collection parameters
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct CollectionParams {
    pub hnsw_config: Option<HnswConfig>,
    pub index_options: Option<IndexOptions>,
    /// Catch-all for engine-specific collection params (e.g., OpenSearch "method")
    #[serde(flatten)]
    pub extra: Option<HashMap<String, serde_json::Value>>,
}

/// Search parameters for a single search configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SearchParams {
    pub parallel: Option<i64>,
    pub search_params: Option<InnerSearchParams>,
    pub top: Option<i64>,
    pub num_candidates: Option<i64>,
    /// Calibration: name of the search param to tune (e.g., "ef")
    pub calibration_param: Option<String>,
    /// Calibration: target precision to achieve
    pub calibration_precision: Option<f64>,
    /// Catch-all for engine-specific search params (e.g., OpenSearch "knn.algo_param.ef_search")
    #[serde(flatten)]
    pub extra: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InnerSearchParams {
    pub ef: Option<i64>,
    /// Catch-all for additional search params (e.g., SEARCH_WINDOW_SIZE, data_type)
    #[serde(flatten)]
    pub extra: Option<std::collections::HashMap<String, serde_json::Value>>,
}

/// Engine configuration from experiments/configurations/*.json
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EngineConfig {
    pub name: String,
    pub engine: Option<String>,
    pub algorithm: Option<String>,
    pub connection_params: Option<serde_json::Value>,
    pub collection_params: Option<CollectionParams>,
    pub search_params: Option<Vec<SearchParams>>,
    pub upload_params: Option<serde_json::Value>,
    /// When true, vectors are uploaded but not indexed; search is filter-only.
    #[serde(default)]
    pub skip_vector_index: bool,
}

/// Get the project root directory
pub fn project_root() -> PathBuf {
    let current = std::env::current_dir().unwrap_or_default();

    // Look for datasets/datasets.json to verify we're in the right place
    if current.join("datasets/datasets.json").exists() {
        return current;
    }

    // Try parent directories
    let mut search = current.clone();
    for _ in 0..5 {
        if let Some(parent) = search.parent() {
            search = parent.to_path_buf();
            if search.join("datasets/datasets.json").exists() {
                return search;
            }
        }
    }

    current
}

/// Get datasets directory path
pub fn datasets_dir() -> PathBuf {
    project_root().join("datasets")
}

/// Read all dataset configurations
pub fn read_dataset_configs() -> Result<HashMap<String, DatasetConfig>, String> {
    let datasets_json = project_root().join("datasets/datasets.json");
    let content = fs::read_to_string(&datasets_json)
        .map_err(|e| format!("Failed to read datasets.json at {:?}: {}", datasets_json, e))?;

    let configs: Vec<DatasetConfig> = serde_json::from_str(&content)
        .map_err(|e| format!("Failed to parse datasets.json: {}", e))?;

    let mut map = HashMap::new();
    for config in configs {
        map.insert(config.name.clone(), config);
    }
    Ok(map)
}

/// Read all engine configurations from experiments/configurations/*.json
pub fn read_engine_configs() -> Result<HashMap<String, EngineConfig>, String> {
    let configs_dir = project_root().join("experiments/configurations");
    let pattern = configs_dir.join("*.json");

    let mut all_configs = HashMap::new();

    for path in glob::glob(pattern.to_str().unwrap())
        .map_err(|e| e.to_string())?
        .flatten()
    {
        if let Ok(content) = fs::read_to_string(&path) {
            if let Ok(configs) = serde_json::from_str::<Vec<EngineConfig>>(&content) {
                for config in configs {
                    all_configs.insert(config.name.clone(), config);
                }
            }
        }
    }
    Ok(all_configs)
}

/// Match a name against a pattern (supports * wildcard)
pub fn matches_pattern(name: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    glob::Pattern::new(pattern)
        .map(|p| p.matches(name))
        .unwrap_or(false)
}

/// Format a vector count with K/M/B suffixes
fn format_count(count: i64) -> String {
    if count >= 1_000_000_000 {
        format!("{:.1}B", count as f64 / 1_000_000_000.0)
    } else if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

/// Summarize a schema value into a compact string
fn format_schema(schema: &serde_json::Value, max_len: usize) -> String {
    let obj = match schema.as_object() {
        Some(o) => o,
        None => return schema.to_string(),
    };
    let field_count = obj.len();
    let base = if field_count == 1 {
        "1 field".to_string()
    } else {
        format!("{} fields", field_count)
    };
    if field_count == 0 {
        return base;
    }
    // Try to add detail
    let detail = if field_count <= 2 {
        let names: Vec<&String> = obj.keys().collect();
        format!(
            "{}: {}",
            base,
            names
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    } else {
        let mut types: Vec<String> = obj
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        types.sort();
        types.dedup();
        format!("{} ({})", base, types.join(", "))
    };
    if detail.len() <= max_len {
        detail
    } else {
        base
    }
}

/// Describe available datasets
pub fn describe_datasets(verbose: bool) -> Result<(), String> {
    let configs = read_dataset_configs()?;

    // Sort by dimension, then vector count, then name
    let mut sorted: Vec<(&String, &DatasetConfig)> = configs.iter().collect();
    sorted.sort_by(|(_, a), (_, b)| {
        let dim_a = a.vector_size.unwrap_or(0);
        let dim_b = b.vector_size.unwrap_or(0);
        dim_a
            .cmp(&dim_b)
            .then_with(|| {
                a.vector_count
                    .unwrap_or(0)
                    .cmp(&b.vector_count.unwrap_or(0))
            })
            .then_with(|| a.name.to_lowercase().cmp(&b.name.to_lowercase()))
    });

    println!("\nAvailable Datasets ({} found)", configs.len());
    println!("{}", "=".repeat(131));

    if verbose {
        for (name, config) in &sorted {
            println!("\n  {}", name);
            println!(
                "   Vector Size: {}",
                config
                    .vector_size
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "N/A".into())
            );
            println!(
                "   Distance:    {}",
                config.distance.as_deref().unwrap_or("N/A")
            );
            println!(
                "   Type:        {}",
                config.dataset_type.as_deref().unwrap_or("N/A")
            );
            if let serde_json::Value::String(p) = &config.path {
                println!("   Path:        {}", p);
            }
            if let Some(link) = &config.link {
                println!("   Download:    {}", link);
            }
            if let Some(desc) = &config.description {
                println!("   Description: {}", desc);
            }
            if let Some(schema) = &config.schema {
                println!("   Schema:      {}", schema);
            }
        }
    } else {
        // Column widths: Name(35) Dims(6) Distance(10) Count(14) Description(30) Schema(20)
        println!(
            "{:<35}{:<6}{:<10}{:<14}{:<30}{:<20}",
            "Dataset Name", "Dims", "Distance", "Vector Count", "Description", "Schema"
        );
        println!("{}", "-".repeat(115));

        for (name, config) in &sorted {
            let dims = config
                .vector_size
                .map(|v| v.to_string())
                .unwrap_or_else(|| "N/A".into());
            let distance = config.distance.as_deref().unwrap_or("N/A");
            let count_str = config
                .vector_count
                .map(format_count)
                .unwrap_or_else(|| "N/A".into());
            let desc = config.description.as_deref().unwrap_or("");
            let desc_display = if desc.len() > 29 {
                format!("{}...", &desc[..26])
            } else {
                desc.to_string()
            };
            let schema_str = config
                .schema
                .as_ref()
                .map(|s| format_schema(s, 19))
                .unwrap_or_default();

            let display_name = if name.len() > 34 {
                format!("{}...", &name[..31])
            } else {
                name.to_string()
            };

            println!(
                "{:<35}{:<6}{:<10}{:<14}{:<30}{:<20}",
                display_name, dims, distance, count_str, desc_display, schema_str
            );
        }
    }

    println!("\nTotal: {} datasets", configs.len());
    if verbose {
        println!();
    } else {
        println!("Use --verbose for detailed information");
    }
    Ok(())
}

/// Describe available engines
pub fn describe_engines(verbose: bool) -> Result<(), String> {
    let configs = read_engine_configs()?;
    println!("Available engines ({}):", configs.len());
    for (name, config) in configs.iter() {
        if verbose {
            println!(
                "  {} - engine: {:?}, algorithm: {:?}",
                name, config.engine, config.algorithm
            );
        } else {
            println!("  {}", name);
        }
    }
    Ok(())
}
