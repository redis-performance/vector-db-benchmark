//! Vertex AI Vector Search integration test — GATED, cloud-only.
//!
//! Vertex AI has no local server: a real run needs a GCP project, credentials,
//! and (for a fresh index) a tens-of-minutes deploy. So this test SELF-SKIPS
//! unless `VERTEX_PROJECT` is set — CI (which has no GCP creds) stays green,
//! while anyone with a project can exercise the engine end-to-end.
//!
//! Run it:
//!
//! ```bash
//! # Auth (either export a token, or rely on gcloud; the engine falls back to
//! # `gcloud auth print-access-token`).
//! gcloud config set project <your-project>
//! export VERTEX_PROJECT=<your-project>
//! export VERTEX_REGION=us-central1
//! # VERTEX_MACHINE_TYPE defaults to e2-standard-16 (e2-standard-2 is rejected
//! # at deploy — too small for the default shard size).
//!
//! # Full lifecycle (creates + DEPLOYS a fresh index — slow, ~30-40 min):
//! cargo test --test integration_vertex --release -- --nocapture
//!
//! # Fast repeat: reuse an already-deployed index and skip the deploy.
//! export VERTEX_INDEX=projects/P/locations/us-central1/indexes/ID
//! export VERTEX_INDEX_ENDPOINT=projects/P/locations/us-central1/indexEndpoints/EID
//! export VERTEX_DEPLOYED_INDEX_ID=vdb_benchmark_deployed
//! cargo test --test integration_vertex --release -- --nocapture
//! ```
//!
//! The test drives the real benchmark binary against the repo's `random-100`
//! dataset (100 × 100-d, cosine, with real brute-force ground truth) and asserts
//! the reported recall clears a floor — proving upload + `findNeighbors` returned
//! the actual nearest neighbours, not garbage. The `VERTEX_*` env vars set for
//! the test are inherited by the spawned binary.

mod common;

use std::fs;

fn gated_out() -> bool {
    if std::env::var("VERTEX_PROJECT").is_err() {
        eprintln!(
            "SKIP integration_vertex: set VERTEX_PROJECT (and auth via VERTEX_ACCESS_TOKEN or \
             `gcloud auth print-access-token`) to run this cloud-only test."
        );
        return true;
    }
    false
}

#[test]
fn test_binary_vertex_knn_recall() {
    if gated_out() {
        return;
    }

    // Build a temp project that reuses the repo's real `random-100` dataset, so
    // the binary's project_root resolves to the temp dir (matching the other
    // integration tests' temp-project pattern).
    let repo = env!("CARGO_MANIFEST_DIR");
    let tmp = tempfile::tempdir().expect("temp dir");
    let root = tmp.path();
    fs::create_dir_all(root.join("datasets/random-100")).unwrap();
    fs::create_dir_all(root.join("experiments/configurations")).unwrap();
    fs::create_dir_all(root.join("results")).unwrap();
    for f in ["vectors.jsonl", "queries.jsonl", "neighbours.jsonl"] {
        fs::copy(
            format!("{}/datasets/random-100/{}", repo, f),
            root.join("datasets/random-100").join(f),
        )
        .unwrap_or_else(|e| panic!("copy random-100/{}: {}", f, e));
    }
    let datasets = serde_json::json!([{
        "name": "random-100",
        "type": "jsonl",
        "path": "random-100/",
        "distance": "cosine",
        "vector_size": 100,
        "vector_count": 100,
    }]);
    fs::write(
        root.join("datasets/datasets.json"),
        serde_json::to_string_pretty(&datasets).unwrap(),
    )
    .unwrap();
    let cfg = serde_json::json!([{
        "name": "vertex-it",
        "engine": "vertex",
        "connection_params": {},
        "collection_params": {},
        "search_params": [{ "parallel": 1 }, { "parallel": 8 }],
        "upload_params": { "parallel": 2, "batch_size": 100 },
    }]);
    fs::write(
        root.join("experiments/configurations/vertex-it.json"),
        serde_json::to_string(&cfg).unwrap(),
    )
    .unwrap();

    // The VERTEX_* vars in the test's environment (project, region, machine type,
    // and any reuse-index vars) are inherited by the spawned binary.
    let ok = common::run_binary(root, "vertex-it", "random-100", "localhost", &[]);
    assert!(ok, "vertex-it run failed");

    let recall = common::read_recall(root, "vertex-it");
    println!("vertex random-100 KNN recall = {:.3}", recall);
    assert!(
        recall >= 0.8,
        "vertex KNN recall {:.3} < 0.8 — findNeighbors did not return the true neighbours",
        recall
    );
}
