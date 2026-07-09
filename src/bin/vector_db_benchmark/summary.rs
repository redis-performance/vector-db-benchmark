//! Precision analysis, summary generation, and ASCII scatter plot.
//!
//! Mirrors Python v0/engine/base_client/client.py:
//! - analyze_precision_performance()
//! - _display_results_summary()
//! - _create_ascii_scatter_plot()

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use serde_json::json;

use crate::engine::SearchResults;

/// A single search result entry for precision analysis.
#[derive(Debug, Clone)]
pub struct SearchEntry {
    pub search_id: usize,
    pub ef: String,
    pub parallel: i64,
    pub results: SearchResults,
}

/// Precision analysis result for a single precision bucket.
#[derive(Debug, Clone)]
struct PrecisionBucket {
    qps: f64,
    p50_ms: f64,
    p95_ms: f64,
    precision: f64,
    recall: f64,
    mrr: f64,
    ndcg: f64,
    /// The run behind this bucket was flagged client-saturated.
    saturated: bool,
}

/// Format a precision value into a bucket key (matches Python v0 format_precision_key).
///
/// - For precision < 0.97: rounds to nearest 0.01
/// - For precision >= 0.97: rounds to nearest 0.0025
fn format_precision_key(precision: f64) -> String {
    if precision < 0.97 {
        format!("{:.2}", (precision * 100.0).round() / 100.0)
    } else {
        // Finer granularity near 1.0: 0.0025 steps
        let step = 0.0025;
        let rounded = (precision / step).round() * step;
        format!("{:.4}", rounded)
    }
}

/// Analyze search results and find best QPS at each precision level.
///
/// Groups results by formatted precision bucket and picks the entry
/// with the highest QPS for each bucket.
fn analyze_precision_performance(entries: &[SearchEntry]) -> Vec<PrecisionBucket> {
    let mut buckets: BTreeMap<String, PrecisionBucket> = BTreeMap::new();

    for entry in entries {
        let key = format_precision_key(entry.results.mean_precision);

        let candidate = PrecisionBucket {
            qps: entry.results.rps,
            p50_ms: entry.results.p50_time * 1000.0,
            p95_ms: entry.results.p95_time * 1000.0,
            precision: entry.results.mean_precision,
            recall: entry.results.mean_recall,
            mrr: entry.results.mean_mrr,
            ndcg: entry.results.mean_ndcg,
            saturated: entry.results.client_saturated,
        };

        // Keep the best entry for this precision bucket. Prefer a NON-saturated
        // point over a saturated one so the headline "best QPS" is never a
        // client-bound data point; only fall back to a saturated point when it is
        // the sole candidate. Within the same saturation class, keep higher QPS.
        let replace = match buckets.get(&key) {
            Some(existing) => match (candidate.saturated, existing.saturated) {
                (false, true) => true,
                (true, false) => false,
                _ => candidate.qps > existing.qps,
            },
            None => true,
        };
        if replace {
            buckets.insert(key, candidate);
        }
    }

    // Return sorted by precision descending (for display)
    let mut result: Vec<PrecisionBucket> = buckets.into_values().collect();
    result.sort_by(|a, b| b.precision.partial_cmp(&a.precision).unwrap());
    result
}

/// Detect and describe client/throughput-saturation problems across a set of
/// runs. Two independent signals:
///  1. any single run the client-CPU coverage flagged (`client_saturated`); and
///  2. throughput that stops scaling — grouped by `ef`, when raising `parallel`
///     gains <10% QPS while p95 rises, the higher-parallel point is not a clean
///     scaling data point (client or server saturated).
fn saturation_warnings(entries: &[SearchEntry]) -> Vec<String> {
    let mut warnings = Vec::new();

    let mut flagged: Vec<(&str, i64, &str)> = entries
        .iter()
        .filter(|e| e.results.client_saturated)
        .map(|e| {
            (
                e.ef.as_str(),
                e.parallel,
                e.results.saturation_reason.as_str(),
            )
        })
        .collect();
    flagged.sort_by_key(|f| f.1);
    for (ef, parallel, reason) in flagged {
        warnings.push(format!(
            "client-saturated: ef={} parallel={} ({}) — QPS/latency reflect the client, not the DB",
            ef, parallel, reason
        ));
    }

    // Throughput scaling collapse, per ef.
    let mut by_ef: BTreeMap<&str, Vec<(i64, f64, f64)>> = BTreeMap::new();
    for e in entries {
        by_ef.entry(e.ef.as_str()).or_default().push((
            e.parallel,
            e.results.rps,
            e.results.p95_time,
        ));
    }
    for (ef, mut pts) in by_ef {
        if pts.len() < 2 {
            continue;
        }
        pts.sort_by_key(|p| p.0);
        for w in pts.windows(2) {
            let (p_prev, q_prev, p95_prev) = w[0];
            let (p_cur, q_cur, p95_cur) = w[1];
            if p_cur > p_prev && q_prev > 0.0 && p95_prev > 0.0 {
                let gain = (q_cur - q_prev) / q_prev;
                if gain < 0.10 && p95_cur > p95_prev {
                    warnings.push(format!(
                        "throughput saturated: ef={} parallel {}→{} gained only {:.0}% QPS \
                         while p95 rose {:.0}% — higher concurrency is not paying off",
                        ef,
                        p_prev,
                        p_cur,
                        gain * 100.0,
                        (p95_cur / p95_prev - 1.0) * 100.0
                    ));
                }
            }
        }
    }

    warnings
}

/// Print saturation warnings, if any, under a clear header.
fn print_saturation_warnings(entries: &[SearchEntry]) {
    let warnings = saturation_warnings(entries);
    if warnings.is_empty() {
        return;
    }
    eprintln!("\n⚠ CONCURRENCY / CPU SATURATION WARNINGS (results below may be untrustworthy):");
    for w in &warnings {
        eprintln!("  - {}", w);
    }
    eprintln!();
}

/// Display a results summary table and ASCII scatter plot.
pub fn display_results_summary(engine_name: &str, dataset_name: &str, entries: &[SearchEntry]) {
    if entries.is_empty() {
        return;
    }

    print_saturation_warnings(entries);

    // Filter-only mode: precision is not applicable (mean_precision == -1.0)
    let filter_only = entries.iter().all(|e| e.results.mean_precision < 0.0);

    if filter_only {
        println!("{}", "-".repeat(40));
        println!("{:<10} {:<12} {:<12}", "QPS", "P50 (ms)", "P95 (ms)");
        println!("{}", "-".repeat(40));

        for e in entries {
            println!(
                "{:<10.1} {:<12.3} {:<12.3}",
                e.results.rps,
                e.results.p50_time * 1000.0,
                e.results.p95_time * 1000.0,
            );
        }
        println!();
        return;
    }

    let buckets = analyze_precision_performance(entries);
    if buckets.is_empty() {
        return;
    }

    // Print table
    println!("{}", "-".repeat(82));
    println!(
        "{:<10} {:<10} {:<8} {:<8} {:<10} {:<12} {:<12}",
        "Recall", "Precision", "MRR", "NDCG", "QPS", "P50 (ms)", "P95 (ms)"
    );
    println!("{}", "-".repeat(82));

    for b in &buckets {
        println!(
            "{:<10.4} {:<10.4} {:<8.4} {:<8.4} {:<10.1} {:<12.3} {:<12.3}",
            b.recall, b.precision, b.mrr, b.ndcg, b.qps, b.p50_ms, b.p95_ms
        );
    }
    println!();

    // ASCII scatter plot
    create_ascii_scatter_plot(engine_name, dataset_name, &buckets);
}

/// Display a mixed benchmark summary table with one row per update-search ratio.
/// Only called when multiple phases (ratios) were used.
pub fn display_mixed_summary(entries: &[SearchEntry]) {
    // Group entries by ratio (None = pure search, Some("U:S") = mixed)
    let mut by_ratio: BTreeMap<String, Vec<&SearchEntry>> = BTreeMap::new();
    for entry in entries {
        let key = entry
            .results
            .update_search_ratio
            .clone()
            .unwrap_or_else(|| "search".to_string());
        by_ratio.entry(key).or_default().push(entry);
    }

    if by_ratio.len() < 2 {
        return;
    }

    println!("\n{}", "-".repeat(116));
    println!(
        "{:<14} {:<8} {:<8} {:<8} {:<8} {:<10} {:<12} {:<12} {:<10} {:<12}",
        "Ratio",
        "Recall",
        "Prec",
        "MRR",
        "NDCG",
        "QPS",
        "P50 (ms)",
        "P95 (ms)",
        "Upd QPS",
        "Upd P50 (ms)"
    );
    println!("{}", "-".repeat(116));

    // Sort: "search" first, then ratios by ascending numeric value
    let mut keys: Vec<String> = by_ratio.keys().cloned().collect();
    keys.sort_by(|a, b| {
        let ra = ratio_sort_key(a);
        let rb = ratio_sort_key(b);
        ra.partial_cmp(&rb).unwrap()
    });

    for key in &keys {
        let group = &by_ratio[key];
        // Pick the entry with the highest QPS in this group
        let best = group
            .iter()
            .max_by(|a, b| a.results.rps.partial_cmp(&b.results.rps).unwrap())
            .unwrap();

        let upd_qps = best
            .results
            .update_rps
            .map(|v| format!("{:.1}", v))
            .unwrap_or_else(|| "-".to_string());
        let upd_p50 = best
            .results
            .update_p50_time
            .map(|v| format!("{:.3}", v * 1000.0))
            .unwrap_or_else(|| "-".to_string());

        println!(
            "{:<14} {:<8.4} {:<8.4} {:<8.4} {:<8.4} {:<10.1} {:<12.3} {:<12.3} {:<10} {:<12}",
            key,
            best.results.mean_recall,
            best.results.mean_precision,
            best.results.mean_mrr,
            best.results.mean_ndcg,
            best.results.rps,
            best.results.p50_time * 1000.0,
            best.results.p95_time * 1000.0,
            upd_qps,
            upd_p50,
        );
    }
    println!();
}

/// Return a sort key for ratio strings: "search" → -1, "U:S" → U/S.
fn ratio_sort_key(key: &str) -> f64 {
    if key == "search" {
        return -1.0;
    }
    let parts: Vec<&str> = key.split(':').collect();
    if parts.len() == 2 {
        if let (Ok(u), Ok(s)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
            if s > 0.0 {
                return u / s;
            }
        }
    }
    0.0
}

/// Save summary JSON matching Python v0 format.
pub fn save_summary(
    engine_name: &str,
    dataset_name: &str,
    entries: &[SearchEntry],
    upload_json: Option<&serde_json::Value>,
    results_dir: &Path,
) -> Result<(), String> {
    let buckets = analyze_precision_performance(entries);

    // Build precision_summary matching Python v0 format
    let mut precision_summary = serde_json::Map::new();
    for b in &buckets {
        let key = format_precision_key(b.precision);
        precision_summary.insert(
            key,
            json!({
                "QPS": (b.qps * 10.0).round() / 10.0,
                "P50 (ms)": (b.p50_ms * 1000.0).round() / 1000.0,
                "P95 (ms)": (b.p95_ms * 1000.0).round() / 1000.0,
                "recall": (b.recall * 10000.0).round() / 10000.0,
                "mrr": (b.mrr * 10000.0).round() / 10000.0,
                "ndcg": (b.ndcg * 10000.0).round() / 10000.0,
            }),
        );
    }

    // Build search results array
    let search_results: Vec<serde_json::Value> = entries
        .iter()
        .map(|e| {
            json!({
                "search_id": e.search_id,
                "ef": e.ef,
                "parallel": e.parallel,
                "mean_precisions": e.results.mean_precision,
                "mean_recall": e.results.mean_recall,
                "mean_mrr": e.results.mean_mrr,
                "mean_ndcg": e.results.mean_ndcg,
                "rps": e.results.rps,
                "mean_time": e.results.mean_time,
                "p50_time": e.results.p50_time,
                "p95_time": e.results.p95_time,
                "p99_time": e.results.p99_time,
                "client_saturated": e.results.client_saturated,
                "saturation_reason": e.results.saturation_reason,
                "oversubscribed": e.results.oversubscribed,
                "available_cores": e.results.available_cores,
                "client_cpu_cores_used": e.results.client_cpu_cores_used,
            })
        })
        .collect();

    let mut summary = json!({
        "engine": engine_name,
        "dataset": dataset_name,
        "search_results": search_results,
        "precision_summary": precision_summary,
    });

    if let Some(upload) = upload_json {
        summary
            .as_object_mut()
            .unwrap()
            .insert("upload".to_string(), upload.clone());
    }

    let filename = format!("{}-{}-summary.json", engine_name, dataset_name);
    let path = results_dir.join(&filename);
    fs::write(&path, serde_json::to_string_pretty(&summary).unwrap())
        .map_err(|e| format!("Failed to save summary: {}", e))?;

    println!(
        "\n{}\nResults saved to:  {}",
        "=".repeat(80),
        results_dir.display()
    );
    println!("Summary saved to:  {}", path.display());

    Ok(())
}

/// Create a 60x12 ASCII scatter plot of QPS vs Precision.
fn create_ascii_scatter_plot(engine_name: &str, dataset_name: &str, buckets: &[PrecisionBucket]) {
    if buckets.is_empty() {
        return;
    }

    let width: usize = 60;
    let height: usize = 12;

    // Get data ranges
    let precisions: Vec<f64> = buckets.iter().map(|b| b.precision).collect();
    let qps_values: Vec<f64> = buckets.iter().map(|b| b.qps).collect();

    let min_prec = precisions.iter().copied().fold(f64::INFINITY, f64::min);
    let max_prec = precisions.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_qps = 0.0_f64;
    let max_qps = qps_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    // Add padding to ranges
    let prec_range = if (max_prec - min_prec).abs() < 1e-9 {
        0.01
    } else {
        max_prec - min_prec
    };
    let qps_range = if max_qps.abs() < 1e-9 {
        1.0
    } else {
        max_qps - min_qps
    };

    // Initialize grid
    let mut grid = vec![vec![' '; width]; height];

    // Plot points
    for b in buckets {
        let x = ((b.precision - min_prec) / prec_range * (width - 1) as f64).round() as usize;
        let y = ((b.qps - min_qps) / qps_range * (height - 1) as f64).round() as usize;
        let x = x.min(width - 1);
        let y = y.min(height - 1);
        // Invert y (top = high QPS)
        grid[height - 1 - y][x] = '\u{25cf}'; // ●
    }

    println!(
        "\nQPS vs Precision Trade-off - {} - {} (up and to the right is better):\n",
        engine_name, dataset_name
    );

    // Print grid with Y-axis labels
    for (row_idx, row) in grid.iter().enumerate() {
        let qps_val = max_qps - (row_idx as f64 / (height - 1) as f64) * qps_range;
        let line: String = row.iter().collect();
        print!("{:>8.0} \u{2502}", qps_val);
        println!("{}", line);
    }

    // X-axis
    print!("         \u{2514}");
    println!("{}", "\u{2500}".repeat(width));

    // X-axis labels
    let label_count = 4;
    let label_spacing = width / label_count;
    print!("          ");
    for i in 0..label_count {
        let val = min_prec + (i as f64 / (label_count - 1) as f64) * prec_range;
        print!("{:<width$.3}", val, width = label_spacing);
    }
    println!();
    println!("{}Precision (0.0 = 0%, 1.0 = 100%)", " ".repeat(8));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_precision_key_low() {
        assert_eq!(format_precision_key(0.50), "0.50");
        assert_eq!(format_precision_key(0.9066), "0.91");
        assert_eq!(format_precision_key(0.96), "0.96");
    }

    #[test]
    fn test_format_precision_key_high() {
        // Near 1.0: finer 0.0025 granularity
        // 0.9986 / 0.0025 = 399.44 → rounds to 399 → 399 * 0.0025 = 0.9975
        assert_eq!(format_precision_key(0.9986), "0.9975");
        assert_eq!(format_precision_key(1.0), "1.0000");
        // 0.9988 / 0.0025 = 399.52 → rounds to 400 → 400 * 0.0025 = 1.0000
        assert_eq!(format_precision_key(0.9988), "1.0000");
        // 0.98 / 0.0025 = 392 → 392 * 0.0025 = 0.9800
        assert_eq!(format_precision_key(0.98), "0.9800");
    }

    #[test]
    fn test_analyze_precision_performance_picks_best_qps() {
        let entries = vec![
            SearchEntry {
                search_id: 0,
                ef: "64".to_string(),
                parallel: 100,
                results: SearchResults {
                    mean_precision: 0.90,
                    rps: 5000.0,
                    p50_time: 0.01,
                    p95_time: 0.02,
                    ..Default::default()
                },
            },
            SearchEntry {
                search_id: 1,
                ef: "128".to_string(),
                parallel: 100,
                results: SearchResults {
                    mean_precision: 0.91,
                    rps: 4000.0,
                    p50_time: 0.012,
                    p95_time: 0.025,
                    ..Default::default()
                },
            },
        ];
        let buckets = analyze_precision_performance(&entries);
        // 0.90 rounds to "0.90", 0.91 rounds to "0.91" — separate buckets
        assert_eq!(buckets.len(), 2);
        assert!(buckets[0].precision > buckets[1].precision);
    }
}
