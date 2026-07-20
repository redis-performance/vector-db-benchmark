//! Compact digests for the per-query sample arrays saved in each result file.
//!
//! For large runs (millions of queries) the runner used to serialize the FULL
//! per-query `latencies`/`precisions`/`recalls`/`mrrs`/`ndcgs` arrays into every
//! result JSON — five f64-per-query arrays that reach ~80 MB each on a 10M-query
//! run, pushing files past 250 MB. Nothing in this repo reads those arrays back
//! (plot.rs and summary.rs consume only means + p50/p95/p99 + rps), so by default
//! we replace them with compact, offline-re-derivable digests:
//!
//! * Latencies (unbounded, heavy-tailed): an HDR histogram, serialized as a
//!   base64 string (`hdr-v2-deflate` encoding). Any percentile stays recoverable
//!   offline by decoding with hdrhistogram's V2 deserializer.
//! * Quality metrics (bounded to `[0, 1]`, where HDR is a poor fit): a small
//!   fixed-bin linear histogram plus min/p50/p95/p99/max, so percentiles are
//!   re-derivable from the bin counts.
//!
//! `--dump-raw-latencies` restores the full raw arrays for archival fidelity.

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use hdrhistogram::serialization::{Serializer, V2DeflateSerializer};
use hdrhistogram::Histogram;
use serde_json::{json, Value};

/// HDR histogram bounds: 1 microsecond .. 1 hour, 3 significant figures.
/// 3 sig figs keeps every recoverable percentile within 0.1% of the true value
/// while bounding the histogram to a few KB regardless of sample count.
const HDR_LOWEST: u64 = 1;
const HDR_HIGHEST: u64 = 3_600_000_000; // 1 hour in microseconds
const HDR_SIGFIG: u8 = 3;

/// Number of linear bins used by [`quality_dist`] over `[0.0, 1.0]`.
const QUALITY_BINS: usize = 100;

/// Convert a latency in **seconds** to the microsecond integer the HDR
/// histogram records. Clamped to `>= 1` (the histogram's lowest trackable
/// value) so a sub-microsecond sample can't underflow to 0, and rounded so the
/// digest matches the top-level second-valued percentiles.
fn secs_to_micros(secs: f64) -> u64 {
    if !secs.is_finite() || secs <= 0.0 {
        return 1;
    }
    let micros = (secs * 1_000_000.0).round();
    if micros < 1.0 {
        1
    } else if micros > HDR_HIGHEST as f64 {
        HDR_HIGHEST
    } else {
        micros as u64
    }
}

/// Build an HDR histogram from per-query latencies (seconds). Out-of-range
/// values use `saturating_record` so no input can panic the run.
fn build_histogram(latencies_secs: &[f64]) -> Histogram<u64> {
    let mut hist: Histogram<u64> =
        Histogram::new_with_bounds(HDR_LOWEST, HDR_HIGHEST, HDR_SIGFIG).expect("valid HDR bounds");
    for &secs in latencies_secs {
        let micros = secs_to_micros(secs);
        if hist.record(micros).is_err() {
            hist.saturating_record(micros);
        }
    }
    hist
}

/// Serialize an HDR histogram to a base64 `hdr-v2-deflate` string. The result
/// round-trips: [`decode_histogram`] reproduces the same percentiles.
fn encode_histogram(hist: &Histogram<u64>) -> String {
    let mut serializer = V2DeflateSerializer::new();
    let mut buf: Vec<u8> = Vec::new();
    serializer
        .serialize(hist, &mut buf)
        .expect("HDR V2 deflate serialization never fails on an in-memory buffer");
    BASE64.encode(&buf)
}

/// Decode a base64 `hdr-v2-deflate` string produced by [`encode_histogram`]
/// back into a histogram. Used by the round-trip tests (and by any offline
/// consumer re-deriving percentiles).
#[cfg(test)]
fn decode_histogram(data: &str) -> Histogram<u64> {
    let bytes = BASE64.decode(data).expect("valid base64");
    let mut deserializer = hdrhistogram::serialization::Deserializer::new();
    deserializer
        .deserialize(&mut std::io::Cursor::new(bytes))
        .expect("valid HDR v2 payload")
}

/// Build the `latency_hdr` digest object for a set of per-query latencies
/// (seconds). Millisecond convenience fields are derived from the histogram's
/// microsecond values (`micros / 1000.0`). An empty input yields a zero-count
/// digest with `null` statistics and an empty (but valid) encoded histogram.
pub fn latency_hdr(latencies_secs: &[f64]) -> Value {
    let hist = build_histogram(latencies_secs);
    let data = encode_histogram(&hist);
    let count = hist.len();

    let ms = |micros: u64| micros as f64 / 1000.0;
    let pct_ms = |q: f64| ms(hist.value_at_quantile(q));

    if count == 0 {
        return json!({
            "encoding": "hdr-v2-deflate-base64",
            "unit": "microseconds",
            "count": 0,
            "min_ms": Value::Null,
            "max_ms": Value::Null,
            "mean_ms": Value::Null,
            "p50_ms": Value::Null,
            "p90_ms": Value::Null,
            "p95_ms": Value::Null,
            "p99_ms": Value::Null,
            "p999_ms": Value::Null,
            "data": data,
        });
    }

    json!({
        "encoding": "hdr-v2-deflate-base64",
        "unit": "microseconds",
        "count": count,
        "min_ms": ms(hist.min()),
        "max_ms": ms(hist.max()),
        "mean_ms": hist.mean() / 1000.0,
        "p50_ms": pct_ms(0.50),
        "p90_ms": pct_ms(0.90),
        "p95_ms": pct_ms(0.95),
        "p99_ms": pct_ms(0.99),
        "p999_ms": pct_ms(0.999),
        "data": data,
    })
}

/// Build a compact distribution digest for a bounded quality metric
/// (precision/recall/mrr/ndcg, each in `[0, 1]`). Stores min/p50/p95/p99/max
/// (linear-interpolated, matching the runner's percentile convention) plus a
/// fixed 100-bin linear histogram over `[0.0, 1.0]` so any percentile is
/// re-derivable offline from the bin counts. An empty input yields a zero-count
/// digest with `null` percentiles and all-zero counts.
pub fn quality_dist(values: &[f64]) -> Value {
    let lo = 0.0_f64;
    let hi = 1.0_f64;
    let mut counts = vec![0u32; QUALITY_BINS];
    let mut finite: Vec<f64> = Vec::with_capacity(values.len());

    for &v in values {
        if !v.is_finite() {
            continue;
        }
        finite.push(v);
        // Map v in [lo, hi] to a bin index in [0, QUALITY_BINS-1]. Values on or
        // above `hi` land in the last bin; values on or below `lo` in the first.
        let frac = (v - lo) / (hi - lo);
        let idx = (frac * QUALITY_BINS as f64).floor();
        let idx = if idx < 0.0 {
            0
        } else if idx >= QUALITY_BINS as f64 {
            QUALITY_BINS - 1
        } else {
            idx as usize
        };
        counts[idx] += 1;
    }

    if finite.is_empty() {
        return json!({
            "count": 0,
            "min": Value::Null,
            "p50": Value::Null,
            "p95": Value::Null,
            "p99": Value::Null,
            "max": Value::Null,
            "bins": QUALITY_BINS,
            "lo": lo,
            "hi": hi,
            "counts": counts,
        });
    }

    finite.sort_by(|a, b| a.total_cmp(b));
    let pct = |q: f64| crate::engine::percentile_linear(&finite, q);

    json!({
        "count": finite.len(),
        "min": finite[0],
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": finite[finite.len() - 1],
        "bins": QUALITY_BINS,
        "lo": lo,
        "hi": hi,
        "counts": counts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Round-trip: encoding a histogram to base64 and decoding it back must
    // reproduce identical percentiles (HDR serialization is lossless).
    #[test]
    fn hdr_round_trip_preserves_percentiles() {
        let latencies: Vec<f64> = (1..=1000).map(|ms| ms as f64 / 1000.0).collect();
        let hist = build_histogram(&latencies);
        let encoded = encode_histogram(&hist);
        let decoded = decode_histogram(&encoded);
        for q in [0.0, 0.25, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0] {
            assert_eq!(
                hist.value_at_quantile(q),
                decoded.value_at_quantile(q),
                "quantile {} differs after round-trip",
                q
            );
        }
        assert_eq!(hist.len(), decoded.len());
        assert_eq!(hist.max(), decoded.max());
    }

    // Percentile correctness within HDR's 3-sig-fig precision: record 1..=1000
    // ms and assert p50 ≈ 500 ms and p99 ≈ 990 ms within 1%.
    #[test]
    fn hdr_percentiles_are_accurate() {
        let latencies: Vec<f64> = (1..=1000).map(|ms| ms as f64 / 1000.0).collect();
        let hist = build_histogram(&latencies);
        let p50_ms = hist.value_at_quantile(0.50) as f64 / 1000.0;
        let p99_ms = hist.value_at_quantile(0.99) as f64 / 1000.0;
        assert!((p50_ms - 500.0).abs() < 5.0, "p50={p50_ms}ms");
        assert!((p99_ms - 990.0).abs() < 10.0, "p99={p99_ms}ms");
    }

    // The digest JSON must expose ms fields consistent with the histogram.
    #[test]
    fn latency_hdr_digest_shape() {
        let latencies: Vec<f64> = (1..=1000).map(|ms| ms as f64 / 1000.0).collect();
        let d = latency_hdr(&latencies);
        assert_eq!(d["encoding"], "hdr-v2-deflate-base64");
        assert_eq!(d["unit"], "microseconds");
        assert_eq!(d["count"].as_u64().unwrap(), 1000);
        let p50 = d["p50_ms"].as_f64().unwrap();
        assert!((p50 - 500.0).abs() < 5.0, "p50_ms={p50}");
        // Re-decode the embedded data and confirm it matches the reported p99.
        let decoded = decode_histogram(d["data"].as_str().unwrap());
        let p99_ms = decoded.value_at_quantile(0.99) as f64 / 1000.0;
        assert!((p99_ms - d["p99_ms"].as_f64().unwrap()).abs() < 1e-9);
    }

    // Out-of-range / degenerate latencies must not panic and must clamp.
    #[test]
    fn latency_hdr_handles_out_of_range() {
        // Negative, zero, sub-microsecond, and > 1h all recorded without panic.
        let latencies = vec![-1.0, 0.0, 1e-9, 5_000.0];
        let d = latency_hdr(&latencies);
        assert_eq!(d["count"].as_u64().unwrap(), 4);
    }

    // Empty input: no panic, count 0, null-ish stats, empty-but-valid data.
    #[test]
    fn latency_hdr_empty() {
        let d = latency_hdr(&[]);
        assert_eq!(d["count"].as_u64().unwrap(), 0);
        assert!(d["p50_ms"].is_null());
        assert!(d["min_ms"].is_null());
        // Data still decodes to an empty histogram.
        let decoded = decode_histogram(d["data"].as_str().unwrap());
        assert_eq!(decoded.len(), 0);
    }

    // quality_dist on a known bounded set: correct min/max, sane p50, counts
    // sum to N, correct bin placement.
    #[test]
    fn quality_dist_known_values() {
        let d = quality_dist(&[0.0, 0.5, 0.5, 1.0]);
        assert_eq!(d["count"].as_u64().unwrap(), 4);
        assert_eq!(d["min"].as_f64().unwrap(), 0.0);
        assert_eq!(d["max"].as_f64().unwrap(), 1.0);
        assert_eq!(d["bins"].as_u64().unwrap(), 100);
        let p50 = d["p50"].as_f64().unwrap();
        assert!((0.0..=1.0).contains(&p50), "p50={p50}");
        let counts: Vec<u64> = d["counts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_u64().unwrap())
            .collect();
        assert_eq!(counts.iter().sum::<u64>(), 4, "counts must sum to N");
        // 0.0 -> bin 0; 0.5 -> bin 50 (two of them); 1.0 clamps to bin 99.
        assert_eq!(counts[0], 1);
        assert_eq!(counts[50], 2);
        assert_eq!(counts[99], 1);
    }

    // Empty input: no panic, count 0, null percentiles, all-zero counts.
    #[test]
    fn quality_dist_empty() {
        let d = quality_dist(&[]);
        assert_eq!(d["count"].as_u64().unwrap(), 0);
        assert!(d["p50"].is_null());
        assert!(d["min"].is_null());
        let counts: Vec<u64> = d["counts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| c.as_u64().unwrap())
            .collect();
        assert_eq!(counts.len(), 100);
        assert_eq!(counts.iter().sum::<u64>(), 0);
    }
}
