//! `generate-dataset` — emit small, deterministic synthetic datasets on disk in
//! the exact layouts the benchmark's sparse / hybrid / compound(tar) readers
//! expect, so the sparse, hybrid and filter-datatype code paths can be run
//! end-to-end from a registered dataset name (not just from the integration
//! tests).
//!
//! The datasets are registered in `datasets/datasets.json` with LOCAL paths and
//! NO download link — they do not exist until this binary writes them. Run:
//!
//! ```text
//! cargo run --release --bin generate-dataset          # writes into ./datasets
//! cargo run --release --bin generate-dataset -- --out-dir /tmp/ds --only sparse
//! ```
//!
//! What it writes (all sizes are tiny so generation + a smoke run are fast):
//!   * `synthetic-sparse-300/`  — CSR `data.csr` + `queries.csr` +
//!     `neighbours.jsonl` (sparse, dot/MIPS).
//!   * `synthetic-hybrid-16/`   — dense `vectors.npy`/`queries.npy` + sparse
//!     `data.csr`/`queries.csr` + shared `neighbours.jsonl` (hybrid, RRF fusion).
//!   * `synthetic-filter-32/`   — compound `vectors.npy` + `payloads.jsonl`
//!     (keyword/int/bool/datetime fields) + `tests.jsonl` (per-query `conditions`
//!     filter + filtered ground truth). `type: "tar"`.

use std::path::{Path, PathBuf};

use chrono::{Duration, TimeZone, Utc};
use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use vector_db_benchmark::readers::{write_npy_vectors, write_sparse_matrix};
use vector_db_benchmark::synthetic::{
    generate_hybrid, generate_sparse, write_neighbours_jsonl, SparseData,
};

/// Dataset names as registered in `datasets/datasets.json`.
const SPARSE_NAME: &str = "synthetic-sparse-300";
const HYBRID_NAME: &str = "synthetic-hybrid-16";
const FILTER_NAME: &str = "synthetic-filter-32";

#[derive(Parser, Debug)]
#[command(
    name = "generate-dataset",
    about = "Generate small deterministic synthetic datasets (sparse / hybrid / filter) on disk."
)]
struct Args {
    /// Base directory to write datasets into (each dataset gets its own subdir).
    #[arg(long, default_value = "datasets")]
    out_dir: PathBuf,

    /// Generate only one dataset kind. Omit to generate all three.
    #[arg(long, value_parser = ["sparse", "hybrid", "filter"])]
    only: Option<String>,
}

fn main() {
    let args = Args::parse();
    if let Err(e) = run(&args) {
        eprintln!("generate-dataset: error: {e}");
        std::process::exit(1);
    }
}

fn run(args: &Args) -> Result<(), String> {
    std::fs::create_dir_all(&args.out_dir)
        .map_err(|e| format!("create {}: {}", args.out_dir.display(), e))?;

    let want = |kind: &str| args.only.as_deref().map(|o| o == kind).unwrap_or(true);

    if want("sparse") {
        gen_sparse(&args.out_dir)?;
    }
    if want("hybrid") {
        gen_hybrid(&args.out_dir)?;
    }
    if want("filter") {
        gen_filter(&args.out_dir)?;
    }
    println!("\nDone. Register/select these datasets by the names printed above.");
    Ok(())
}

/// Report a written file with its on-disk size.
fn note(path: &Path) -> Result<(), String> {
    let len = std::fs::metadata(path)
        .map_err(|e| format!("stat {}: {}", path.display(), e))?
        .len();
    println!("    {:<20} {:>8} bytes", file_name(path), len);
    Ok(())
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default()
}

// ── sparse ──────────────────────────────────────────────────────────────────

fn gen_sparse(base: &Path) -> Result<(), String> {
    // dim=300, nnz=10, 150 docs, 10 queries, top-10 GT. Fixed seed → reproducible.
    let SparseData {
        data,
        queries,
        neighbours,
    } = generate_sparse(0x5A5A_5EED, 300, 10, 150, 10, 10);

    let dir = base.join(SPARSE_NAME);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    println!(
        "[sparse]  {SPARSE_NAME}/  ({} docs, {} queries)",
        data.len(),
        queries.len()
    );

    let data_csr = dir.join("data.csr");
    let queries_csr = dir.join("queries.csr");
    let nb = dir.join("neighbours.jsonl");
    write_sparse_matrix(data_csr.to_str().ok_or("bad path")?, &data)?;
    write_sparse_matrix(queries_csr.to_str().ok_or("bad path")?, &queries)?;
    write_neighbours_jsonl(&nb, &neighbours)?;
    note(&data_csr)?;
    note(&queries_csr)?;
    note(&nb)?;
    Ok(())
}

// ── hybrid ──────────────────────────────────────────────────────────────────

fn gen_hybrid(base: &Path) -> Result<(), String> {
    let h = generate_hybrid(0xB19_1DEA);

    let dir = base.join(HYBRID_NAME);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    println!(
        "[hybrid]  {HYBRID_NAME}/  ({} docs, {} queries, dim {}, top {})",
        h.dense.len(),
        h.dense_queries.len(),
        h.dim,
        h.top
    );

    let vectors_npy = dir.join("vectors.npy");
    let queries_npy = dir.join("queries.npy");
    let data_csr = dir.join("data.csr");
    let queries_csr = dir.join("queries.csr");
    let nb = dir.join("neighbours.jsonl");
    write_npy_vectors(vectors_npy.to_str().ok_or("bad path")?, &h.dense)?;
    write_npy_vectors(queries_npy.to_str().ok_or("bad path")?, &h.dense_queries)?;
    write_sparse_matrix(data_csr.to_str().ok_or("bad path")?, &h.sparse)?;
    write_sparse_matrix(queries_csr.to_str().ok_or("bad path")?, &h.sparse_queries)?;
    write_neighbours_jsonl(&nb, &h.neighbours)?;
    for p in [&vectors_npy, &queries_npy, &data_csr, &queries_csr, &nb] {
        note(p)?;
    }
    Ok(())
}

// ── filter (compound / tar) ─────────────────────────────────────────────────

/// Keyword values assigned round-robin by `id % 4`.
const COLORS: [&str; 4] = ["red", "green", "blue", "dark blue"];

fn color_for(id: usize) -> &'static str {
    COLORS[id % COLORS.len()]
}
fn size_for(id: usize) -> i64 {
    (id % 5) as i64 + 1
}
fn flag_for(id: usize) -> bool {
    id.is_multiple_of(2)
}

/// A per-query filter: the `conditions` JSON attached to the query and the
/// predicate deciding which document ids satisfy it (used to brute-force the
/// filtered ground truth). Rotating these across queries exercises the keyword,
/// int, bool and datetime filter datatypes in a single dataset.
struct QueryFilter {
    conditions: serde_json::Value,
    matches: Box<dyn Fn(usize) -> bool>,
}

fn gen_filter(base: &Path) -> Result<(), String> {
    const N: usize = 400;
    const N_QUERIES: usize = 12;
    const DIM: usize = 32;
    const TOP: usize = 10;

    // Fixed seed → reproducible vectors, queries and ground truth.
    let mut rng = StdRng::seed_from_u64(0xF117E_u64);
    let gen_vec =
        |rng: &mut StdRng| -> Vec<f32> { (0..DIM).map(|_| rng.gen_range(-1.0f32..1.0)).collect() };
    let vectors: Vec<Vec<f32>> = (0..N).map(|_| gen_vec(&mut rng)).collect();
    let queries: Vec<Vec<f32>> = (0..N_QUERIES).map(|_| gen_vec(&mut rng)).collect();

    // datetime field: one ISO-8601 timestamp per doc, one day apart from base.
    let base_ts = Utc.timestamp_opt(1_609_459_200, 0).unwrap(); // 2021-01-01T00:00:00Z
    let iso_for = move |day: i64| (base_ts + Duration::days(day)).to_rfc3339();
    let ts_gte = iso_for(100);
    let ts_lt = iso_for(300);

    // Per-query filter, rotating through the four datatypes.
    let filter_for = |q: usize| -> QueryFilter {
        match q % 4 {
            0 => QueryFilter {
                // keyword match_any: color IN {red, blue}
                conditions: serde_json::json!({
                    "and": [ { "color": { "match": { "any": ["red", "blue"] } } } ]
                }),
                matches: Box::new(|id| ["red", "blue"].contains(&color_for(id))),
            },
            1 => QueryFilter {
                // int match_any: size IN {1, 2, 3}
                conditions: serde_json::json!({
                    "and": [ { "size": { "match": { "any": [1, 2, 3] } } } ]
                }),
                matches: Box::new(|id| [1, 2, 3].contains(&size_for(id))),
            },
            2 => QueryFilter {
                // bool: flag == true
                conditions: serde_json::json!({
                    "and": [ { "flag": { "match": { "value": true } } } ]
                }),
                matches: Box::new(flag_for),
            },
            _ => {
                // datetime range: ts IN [day 100, day 300)
                let gte = ts_gte.clone();
                let lt = ts_lt.clone();
                QueryFilter {
                    conditions: serde_json::json!({
                        "and": [ { "ts": { "range": { "gte": gte, "lt": lt } } } ]
                    }),
                    matches: Box::new(|id| (100..300).contains(&id)),
                }
            }
        }
    };

    // Ground truth: top-TOP nearest by L2 over ONLY the docs matching the filter.
    let filtered_gt = |q_vec: &[f32], matches: &dyn Fn(usize) -> bool| -> Vec<i64> {
        let l2 = |a: &[f32], b: &[f32]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
                .sum()
        };
        let mut scored: Vec<(i64, f64)> = (0..N)
            .filter(|id| matches(*id))
            .map(|id| (id as i64, l2(q_vec, &vectors[id])))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        scored.iter().take(TOP).map(|(id, _)| *id).collect()
    };

    let dir = base.join(FILTER_NAME);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    println!(
        "[filter]  {FILTER_NAME}/  ({N} docs, {N_QUERIES} queries, dim {DIM}, \
         fields color/size/flag/ts)"
    );

    // vectors.npy (implicit ids 0..N).
    let vectors_npy = dir.join("vectors.npy");
    write_npy_vectors(vectors_npy.to_str().ok_or("bad path")?, &vectors)?;

    // payloads.jsonl: keyword/int/bool/datetime per doc.
    let payloads: String = (0..N)
        .map(|id| {
            serde_json::json!({
                "color": color_for(id),
                "size": size_for(id),
                "flag": flag_for(id),
                "ts": iso_for(id as i64),
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    let payloads_path = dir.join("payloads.jsonl");
    std::fs::write(&payloads_path, payloads).map_err(|e| e.to_string())?;

    // tests.jsonl: query + conditions + filtered ground truth.
    let tests: String = queries
        .iter()
        .enumerate()
        .map(|(q, q_vec)| {
            let f = filter_for(q);
            serde_json::json!({
                "query": q_vec.iter().map(|x| *x as f64).collect::<Vec<_>>(),
                "conditions": f.conditions,
                "closest_ids": filtered_gt(q_vec, &f.matches),
            })
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n");
    let tests_path = dir.join("tests.jsonl");
    std::fs::write(&tests_path, tests).map_err(|e| e.to_string())?;

    note(&vectors_npy)?;
    note(&payloads_path)?;
    note(&tests_path)?;
    Ok(())
}
