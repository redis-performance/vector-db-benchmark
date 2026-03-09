//! Experiment runner - orchestrates benchmark runs.
//!
//! Mirrors Python v0/engine/base_client/client.py run_experiment()

use std::fs;
use std::path::PathBuf;

use chrono::Local;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::json;

use crate::cli::Args;
use crate::config::{
    matches_pattern, project_root, read_dataset_configs, read_engine_configs, InnerSearchParams,
    SearchParams,
};
use crate::dataset::Dataset;
use crate::engine::{create_engine, Engine, UpdateSearchRatio};
use crate::summary::{self, SearchEntry};

/// Results directory
fn results_dir() -> PathBuf {
    let dir = project_root().join("results");
    fs::create_dir_all(&dir).ok();
    dir
}

/// Run all matching experiments
pub fn run(args: &Args) -> Result<(), String> {
    println!("vector-db-benchmark v{}", env!("CARGO_PKG_VERSION"));

    let dataset_configs = read_dataset_configs()?;
    let engine_configs = read_engine_configs()?;

    // Filter datasets by pattern
    let datasets: Vec<_> = dataset_configs
        .iter()
        .filter(|(name, _)| args.datasets.iter().any(|p| matches_pattern(name, p)))
        .collect();

    if datasets.is_empty() {
        return Err(format!(
            "No datasets match pattern: '{}'",
            args.datasets.join(", ")
        ));
    }

    // Filter engines by pattern
    let supported_engines = [
        "redis",
        "vectorsets",
        "elasticsearch",
        "opensearch",
        "qdrant",
        "weaviate",
        "pgvector",
        "milvus",
        "mongodb",
        "valkey",
        "turbopuffer",
    ];
    let mut engines: Vec<_> = engine_configs
        .iter()
        .filter(|(name, config)| {
            let engine_type = config.engine.as_deref().unwrap_or("");
            supported_engines.contains(&engine_type)
                && args.engines.iter().any(|p| matches_pattern(name, p))
        })
        .collect();

    if engines.is_empty() {
        return Err(format!(
            "No engines match pattern: '{}'. Supported: {:?}.",
            args.engines.join(", "),
            supported_engines
        ));
    }

    // --skip-vector-index: deduplicate engine configs by engine type.
    // Multiple M/EF variants (e.g. redis-m-16-ef-64, redis-m-32-ef-128) collapse
    // into a single "<engine_type>-no-vector" experiment.
    if args.skip_vector_index {
        let mut seen_engine_types = std::collections::HashSet::new();
        engines.retain(|(_, config)| {
            let engine_type = config.engine.as_deref().unwrap_or("unknown");
            seen_engine_types.insert(engine_type.to_string())
        });
        println!(
            "--skip-vector-index: deduplicated to {} engine(s)",
            engines.len()
        );
    }

    println!(
        "Found {} datasets, {} engines",
        datasets.len(),
        engines.len()
    );

    // Run experiments
    let total_experiments = engines.len() * datasets.len();
    let pb = ProgressBar::new(total_experiments as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "{spinner:.green} Overall: [{elapsed_precise}] [{bar:30.cyan/blue}] {pos}/{len} experiments (ETA: {eta})",
            )
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.enable_steady_tick(std::time::Duration::from_secs(1));

    let skip_vector_engines = ["redis", "valkey", "mongodb"];

    for (_engine_name, engine_config) in &engines {
        // Apply --skip-vector-index: override name and set flag on config
        let mut engine_config = (*engine_config).clone();
        if args.skip_vector_index {
            let engine_type = engine_config.engine.as_deref().unwrap_or("unknown");
            if !skip_vector_engines.contains(&engine_type) {
                eprintln!(
                    "WARNING: --skip-vector-index not implemented for engine '{}', skipping",
                    engine_type
                );
                continue;
            }
            engine_config.name = format!("{}-no-vector", engine_type);
            engine_config.skip_vector_index = true;
        }

        for (dataset_name, dataset_config) in &datasets {
            let experiment_num = pb.position() + 1;
            pb.suspend(|| {
                println!("\n{}", "=".repeat(60));
                println!(
                    "Running experiment ({}/{}): {} - {}",
                    experiment_num, total_experiments, engine_config.name, dataset_name
                );
                println!("{}", "=".repeat(60));
            });

            let dataset = Dataset::new((*dataset_config).clone());

            // Create engine
            let mut engine = create_engine(&engine_config, &args.host)?;

            // Run experiment phases
            if let Err(e) = run_single_experiment(&mut *engine, &dataset, args) {
                eprintln!("Experiment failed: {}", e);
                if args.exit_on_error {
                    pb.finish_and_clear();
                    return Err(e);
                }
            }
            pb.inc(1);
        }
    }

    pb.finish_and_clear();
    Ok(())
}

/// Parse "U:S" ratio string into UpdateSearchRatio.
fn parse_update_search_ratio(s: &str) -> Result<UpdateSearchRatio, String> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 2 {
        return Err(format!(
            "Invalid update-search-ratio format: '{}'. Expected 'U:S' (e.g., '1:10')",
            s
        ));
    }
    let updates: u64 = parts[0]
        .parse()
        .map_err(|_| format!("Invalid update count: '{}'", parts[0]))?;
    let searches: u64 = parts[1]
        .parse()
        .map_err(|_| format!("Invalid search count: '{}'", parts[1]))?;
    if searches == 0 {
        return Err("Search count must be > 0".to_string());
    }
    Ok(UpdateSearchRatio { updates, searches })
}

/// Run a single experiment (configure, upload, search)
fn run_single_experiment(
    engine: &mut dyn Engine,
    dataset: &Dataset,
    args: &Args,
) -> Result<(), String> {
    // Check if we should skip
    if args.skip_if_exists {
        let glob_pattern = format!("{}-{}-upload-*.json", engine.name(), dataset.config.name);
        let existing: Vec<_> = glob::glob(results_dir().join(&glob_pattern).to_str().unwrap())
            .map(|paths| paths.filter_map(|p| p.ok()).collect())
            .unwrap_or_default();

        if !existing.is_empty() && args.skip_upload {
            println!("Skipping (results exist): {}", glob_pattern);
            return Ok(());
        }
    }

    // Configure phase
    if !args.skip_upload {
        println!("Experiment stage: Configure");
        engine.configure(dataset)?;

        // Upload phase
        println!("Experiment stage: Upload");
        let mut upload_stats = engine.upload(dataset)?;

        // Collect memory usage after upload
        upload_stats.memory_usage = engine.get_memory_usage();

        // Save upload results
        save_upload_results(engine.name(), &dataset.config.name, &upload_stats)?;
    }

    // Build ordered search phases: pure search first, then mixed ratios ascending
    let search_phases: Vec<Option<UpdateSearchRatio>> = if args.update_search_ratio.is_empty() {
        vec![None]
    } else {
        let mut phases = Vec::new();
        let mut ratios: Vec<UpdateSearchRatio> = Vec::new();

        for s in &args.update_search_ratio {
            let ratio = parse_update_search_ratio(s)?;
            if ratio.updates == 0 {
                // 0:S means pure search
                if !phases.contains(&None) {
                    phases.push(None);
                }
            } else {
                ratios.push(ratio);
            }
        }

        // Sort mixed ratios ascending by updates/searches
        ratios.sort_by(|a, b| {
            let ra = a.updates as f64 / a.searches as f64;
            let rb = b.updates as f64 / b.searches as f64;
            ra.partial_cmp(&rb).unwrap()
        });

        for r in ratios {
            phases.push(Some(r));
        }
        phases
    };

    // Search phase
    let mut search_entries: Vec<SearchEntry> = Vec::new();
    let skip_vector_index = args.skip_vector_index;

    if !args.skip_search {
        // --skip-vector-index + no query conditions = nothing to search for
        if skip_vector_index {
            let has_schema = dataset
                .config
                .schema
                .as_ref()
                .and_then(|s| s.as_object())
                .map(|o| !o.is_empty())
                .unwrap_or(false);
            if !has_schema {
                println!(
                    "WARNING: --skip-vector-index with no schema fields on dataset '{}' — \
                     skipping search (no filter conditions possible)",
                    dataset.config.name
                );
                engine.delete()?;
                println!("Experiment stage: Done");
                return Ok(());
            }
        }

        // Clone search params to avoid borrow conflict
        let all_search_params: Vec<_> = engine.search_params().to_vec();

        // --skip-vector-index: dedup search params by parallel level only
        // (ef values are irrelevant for filter-only queries)
        let effective_search_params: Vec<(usize, SearchParams)> = if skip_vector_index {
            let mut seen_parallels = std::collections::HashSet::new();
            all_search_params
                .into_iter()
                .enumerate()
                .filter(|(_, sp)| {
                    let p = sp.parallel.unwrap_or(1);
                    seen_parallels.insert(p)
                })
                .collect()
        } else {
            all_search_params.into_iter().enumerate().collect()
        };

        for phase in &search_phases {
            // --skip-vector-index: skip mixed phases (no vector updates to benchmark)
            if skip_vector_index && phase.is_some() {
                continue;
            }

            match phase {
                Some(ratio) => println!(
                    "Experiment stage: Mixed Search+Update (ratio {}:{})",
                    ratio.updates, ratio.searches
                ),
                None => {
                    if skip_vector_index {
                        println!("Experiment stage: Filter-only Search (no vector index)");
                    } else {
                        println!("Experiment stage: Search");
                    }
                }
            }

            for (search_id, search_params) in &effective_search_params {
                // Filter by parallel if specified
                if !args.parallels.is_empty() {
                    let parallel = search_params.parallel.unwrap_or(1) as i32;
                    if !args.parallels.contains(&parallel) {
                        continue;
                    }
                }

                // Filter by ef_runtime if specified (irrelevant for skip_vector_index)
                if !skip_vector_index && !args.ef_runtime.is_empty() {
                    if let Some(ref inner) = search_params.search_params {
                        if let Some(ef) = inner.ef {
                            if !args.ef_runtime.contains(&ef) {
                                continue;
                            }
                        }
                    }
                }

                let parallel = search_params.parallel.unwrap_or(1);

                // Calibration is skipped for filter-only mode (no vector search to tune)
                let calibrated_params = if skip_vector_index {
                    None
                } else if let (Some(cal_param), Some(cal_precision)) = (
                    &search_params.calibration_param,
                    search_params.calibration_precision,
                ) {
                    println!(
                        "\tCalibrating {}: target precision={:.4}, parallel={}",
                        cal_param, cal_precision, parallel
                    );
                    match calibrate(
                        engine,
                        dataset,
                        search_params,
                        cal_param,
                        cal_precision,
                        args.queries,
                    ) {
                        Ok((value, precision)) => {
                            println!(
                                "\tCalibrated {}={} → precision={:.4}",
                                cal_param, value, precision
                            );
                            // Create a new SearchParams with calibrated value
                            let mut calibrated = search_params.clone();
                            let inner =
                                calibrated
                                    .search_params
                                    .get_or_insert_with(|| InnerSearchParams {
                                        ef: None,
                                        extra: None,
                                    });
                            if cal_param == "ef" {
                                inner.ef = Some(value);
                            } else {
                                let extras = inner.extra.get_or_insert_with(Default::default);
                                extras.insert(cal_param.clone(), serde_json::json!(value));
                            }
                            Some(calibrated)
                        }
                        Err(e) => {
                            eprintln!("\tCalibration failed: {}", e);
                            None
                        }
                    }
                } else {
                    None
                };

                let effective_params = calibrated_params.as_ref().unwrap_or(search_params);
                let effective_ef = if skip_vector_index {
                    "n/a".to_string()
                } else {
                    effective_params
                        .search_params
                        .as_ref()
                        .and_then(|p| p.ef)
                        .map(|e| e.to_string())
                        .unwrap_or_else(|| "default".to_string())
                };

                if skip_vector_index {
                    println!(
                        "\tRunning filter-only search {}: parallel={}",
                        search_id, parallel
                    );
                } else {
                    println!(
                        "\tRunning search {}: ef={}, parallel={}",
                        search_id, effective_ef, parallel
                    );
                }

                let search_result = match phase {
                    Some(ratio) => {
                        engine.search_mixed(dataset, effective_params, args.queries, ratio)
                    }
                    None => engine.search(dataset, effective_params, args.queries),
                };

                match search_result {
                    Ok(results) => {
                        if skip_vector_index {
                            println!("\t→ QPS: {:.1} (filter-only, no precision)", results.rps);
                        } else {
                            println!(
                                "\t→ QPS: {:.1}, Precision: {:.4}",
                                results.rps, results.mean_precision
                            );
                        }
                        save_search_results(
                            engine.name(),
                            &dataset.config.name,
                            *search_id,
                            effective_params,
                            &results,
                        )?;

                        search_entries.push(SearchEntry {
                            search_id: *search_id,
                            ef: effective_ef.clone(),
                            parallel,
                            results,
                        });
                    }
                    Err(e) => {
                        eprintln!("\tSearch failed: {}", e);
                    }
                }
            }
        }
    }

    // Display precision summary and save summary JSON
    if !search_entries.is_empty() {
        summary::display_results_summary(engine.name(), &dataset.config.name, &search_entries);
        if search_phases.len() > 1 {
            summary::display_mixed_summary(&search_entries);
        }
        summary::save_summary(
            engine.name(),
            &dataset.config.name,
            &search_entries,
            None,
            &results_dir(),
        )?;
    }

    // Cleanup
    engine.delete()?;

    println!("Experiment stage: Done");
    Ok(())
}

/// Binary search calibration matching Python v0.
///
/// Searches for the value of `calibration_param` (e.g., "ef") that achieves
/// the target precision. Uses binary search between `min_value` (from `top` in
/// search params, default 10) and `max_value` (1000).
fn calibrate(
    engine: &mut dyn Engine,
    dataset: &Dataset,
    search_params: &SearchParams,
    calibration_param: &str,
    target_precision: f64,
    num_queries: i64,
) -> Result<(i64, f64), String> {
    let min_value = search_params.top.unwrap_or(10);
    let max_value: i64 = 1000;

    let mut lower_bound = min_value;
    let mut upper_bound = max_value;
    let mut lower_visited = false;
    let mut upper_visited = false;
    let mut current = (lower_bound + upper_bound) / 2;
    let mut previous = current;
    let mut previous_precision = 0.0_f64;

    loop {
        // Create search params with current calibration value
        let mut test_params = search_params.clone();
        let inner = test_params
            .search_params
            .get_or_insert_with(|| InnerSearchParams {
                ef: None,
                extra: None,
            });
        if calibration_param == "ef" {
            inner.ef = Some(current);
        } else {
            let extras = inner.extra.get_or_insert_with(Default::default);
            extras.insert(calibration_param.to_string(), serde_json::json!(current));
        }

        let results = engine.search(dataset, &test_params, num_queries)?;
        let current_precision = results.mean_precision;

        println!(
            "\t  calibration: {}={} → precision={:.4}",
            calibration_param, current, current_precision
        );

        if (current_precision - target_precision).abs() < 1e-9 {
            return Ok((current, current_precision));
        } else if current_precision > target_precision {
            upper_bound = current;
            upper_visited = true;
        } else {
            lower_bound = current;
            lower_visited = true;
        }

        let next_value = (lower_bound + upper_bound) / 2;

        // Check convergence: if next step would revisit a bound, pick the closer result
        if (lower_visited && next_value == lower_bound)
            || (upper_visited && next_value == upper_bound)
        {
            if (previous_precision - target_precision).abs()
                < (current_precision - target_precision).abs()
            {
                return Ok((previous, previous_precision));
            } else {
                return Ok((current, current_precision));
            }
        }

        previous = current;
        previous_precision = current_precision;
        current = next_value;
    }
}

/// Save search results to JSON file (matches Python v0 format)
fn save_search_results(
    engine_name: &str,
    dataset_name: &str,
    search_id: usize,
    search_params: &crate::config::SearchParams,
    results: &crate::engine::SearchResults,
) -> Result<(), String> {
    let timestamp = Local::now().format("%Y-%m-%d-%H-%M-%S");
    let pid = std::process::id();
    let mixed_tag = results
        .update_search_ratio
        .as_ref()
        .map(|r| format!("-mixed-{}", r.replace(':', "x")))
        .unwrap_or_default();
    let filename = format!(
        "{}-{}-search-{}{}-{}-{}.json",
        engine_name, dataset_name, search_id, mixed_tag, pid, timestamp
    );

    let mut result = json!({
        "params": {
            "dataset": dataset_name,
            "experiment": engine_name,
            "parallel": search_params.parallel.unwrap_or(1),
            "top": results.top,
            "search_params": search_params.search_params,
        },
        "results": {
            "total_time": results.total_time,
            "mean_time": results.mean_time,
            "mean_precisions": results.mean_precision,
            "std_time": results.std_time,
            "min_time": results.min_time,
            "max_time": results.max_time,
            "rps": results.rps,
            "p50_time": results.p50_time,
            "p95_time": results.p95_time,
            "p99_time": results.p99_time,
            "precisions": results.precisions,
            "latencies": results.latencies,
        }
    });

    // Add update metrics when present (mixed benchmark mode)
    if let Some(ref ratio) = results.update_search_ratio {
        let results_obj = result["results"].as_object_mut().unwrap();
        results_obj.insert("update_search_ratio".to_string(), json!(ratio));
        if let Some(count) = results.update_count {
            results_obj.insert("update_count".to_string(), json!(count));
        }
        if let Some(rps) = results.update_rps {
            results_obj.insert("update_rps".to_string(), json!(rps));
        }
        if let Some(t) = results.update_mean_time {
            results_obj.insert("update_mean_time".to_string(), json!(t));
        }
        if let Some(t) = results.update_p50_time {
            results_obj.insert("update_p50_time".to_string(), json!(t));
        }
        if let Some(t) = results.update_p95_time {
            results_obj.insert("update_p95_time".to_string(), json!(t));
        }
        if let Some(t) = results.update_p99_time {
            results_obj.insert("update_p99_time".to_string(), json!(t));
        }
        if let Some(ref lats) = results.update_latencies {
            results_obj.insert("update_latencies".to_string(), json!(lats));
        }
    }

    let path = results_dir().join(&filename);
    fs::write(&path, serde_json::to_string_pretty(&result).unwrap())
        .map_err(|e| format!("Failed to save results: {}", e))?;

    println!("\tResults saved to: {:?}", path);
    Ok(())
}

/// Save upload results to JSON file
fn save_upload_results(
    engine_name: &str,
    dataset_name: &str,
    stats: &crate::engine::UploadStats,
) -> Result<(), String> {
    let timestamp = Local::now().format("%Y-%m-%d-%H-%M-%S");
    let filename = format!(
        "{}-{}-upload-0-{}-{}.json",
        engine_name, dataset_name, stats.upload_count, timestamp
    );

    let result = json!({
        "params": {
            "experiment": engine_name,
            "dataset": dataset_name,
            "parallel": stats.parallel,
            "batch_size": stats.batch_size,
        },
        "results": {
            "upload_time": stats.upload_time,
            "total_time": stats.total_time,
            "upload_count": stats.upload_count,
            "memory_usage": stats.memory_usage,
        }
    });

    let path = results_dir().join(&filename);
    fs::write(&path, serde_json::to_string_pretty(&result).unwrap())
        .map_err(|e| format!("Failed to save results: {}", e))?;

    println!("Results saved to: {:?}", path);
    Ok(())
}
