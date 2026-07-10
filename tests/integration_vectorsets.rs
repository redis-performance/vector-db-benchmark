//! Integration tests for the VectorSets engine's FILTER path (VADD/VSIM).
//!
//! VectorSets is Redis Vector Sets — same `redis:8.8.0` image as the other Redis
//! tests, but VSIM's FILTER expression grammar (bare-bool syntax error, numeric
//! coercion, value-`in`-field membership) is only observable against a LIVE
//! server, which is exactly why the earlier bool bug slipped past the
//! string-equality unit tests. These tests drive the real benchmark binary
//! end-to-end with filtered ground truth, so a high recall proves the FILTER was
//! actually applied (and, for bool, that it did not error the whole query).
//!
//! VectorSets ranks by COSINE similarity intrinsically (VADD/VSIM take no metric
//! arg), so the fixtures use cosine ground truth (`*_cosine_project`).
//!
//! Requires redis:8.8.0 (Vector Sets) reachable on `VECTORSETS_PORT` (default
//! 6398 — kept distinct from the redis:8.6.0 tests on 6399 because bool FILTER
//! grammar needs 8.8+). Start with:
//!   docker run -d --rm -p 6398:6379 redis:8.8.0
//! Run with:
//!   VECTORSETS_PORT=6398 cargo test --test integration_vectorsets -- --test-threads=1

use std::time::{Duration, Instant};

mod common;

const TEST_HOST: &str = "127.0.0.1";

/// Port of the live Redis 8.8 (Vector Sets) server. The engine reads `REDIS_PORT`
/// (see `engine::build_redis_url`), so we forward this value under that name.
fn test_port() -> u16 {
    std::env::var("VECTORSETS_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6398)
}

/// Block until the server answers PING (or panic after 10s), and verify it
/// actually supports Vector Sets (VADD) so a misconfigured image fails loudly.
fn wait_for_vectorsets() {
    let port = test_port();
    let url = format!("redis://{}:{}/", TEST_HOST, port);
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        if let Ok(client) = redis::Client::open(url.as_str()) {
            if let Ok(mut conn) = client.get_connection() {
                if redis::cmd("PING").query::<String>(&mut conn).is_ok() {
                    // VADD arity error ("wrong number of arguments") proves the
                    // command EXISTS; "unknown command" means no Vector Sets.
                    let probe: Result<redis::Value, redis::RedisError> =
                        redis::cmd("VADD").query(&mut conn);
                    if let Err(e) = probe {
                        let msg = e.to_string().to_lowercase();
                        assert!(
                            !msg.contains("unknown command"),
                            "server on port {port} lacks Vector Sets (VADD). Use redis:8.8.0."
                        );
                    }
                    return;
                }
            }
        }
        if Instant::now() > deadline {
            panic!("Redis (Vector Sets) not available on port {port} after 10s");
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

fn vectorsets_config(name: &str) -> String {
    let configs = serde_json::json!([{
        "name": name,
        "engine": "vectorsets",
        "search_params": [{"parallel": 1, "search_params": {"ef": 400}}],
        "upload_params": {
            "hnsw_config": {"quant": "NOQUANT", "M": 16, "EF_CONSTRUCTION": 200},
            "CAS": true,
            "parallel": 1,
            "batch_size": 100
        }
    }]);
    serde_json::to_string(&configs).unwrap()
}

/// End-to-end keyword `match_any`: filter `color IN {red, blue}` and assert the
/// engine returns the filtered nearest neighbours. Pins the `"<v>" in .field`
/// value-in-field emission (equality on a scalar keyword attribute).
#[test]
fn test_binary_vectorsets_match_any() {
    wait_for_vectorsets();

    let name = "vectorsets-ma";
    let proj = common::write_match_any_cosine_project("vs-match-any", &vectorsets_config(name), 8);
    assert!(
        proj.matching_docs >= proj.top,
        "fixture must have >= top matching docs (got {})",
        proj.matching_docs
    );

    let port = test_port().to_string();
    assert!(
        common::run_binary(
            &proj.root,
            name,
            "vs-match-any",
            TEST_HOST,
            &[("REDIS_PORT", port.as_str())],
        ),
        "vectorsets match_any run failed"
    );

    let recall = common::read_recall(&proj.root, name);
    println!("vectorsets match_any (keyword) recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "vectorsets keyword match_any recall {:.3} < 0.9",
        recall
    );
}

/// End-to-end integer `match_any`: filter `size IN {1, 2}`. Pins the numeric
/// `.field == N` arm (sizes are stored as JSON strings; VSIM coerces "1" == 1).
#[test]
fn test_binary_vectorsets_match_any_int() {
    wait_for_vectorsets();

    let name = "vectorsets-ma-int";
    let proj =
        common::write_match_any_int_cosine_project("vs-match-any-int", &vectorsets_config(name), 8);
    assert!(proj.matching_docs >= proj.top);

    let port = test_port().to_string();
    assert!(
        common::run_binary(
            &proj.root,
            name,
            "vs-match-any-int",
            TEST_HOST,
            &[("REDIS_PORT", port.as_str())],
        ),
        "vectorsets int match_any run failed"
    );

    let recall = common::read_recall(&proj.root, name);
    println!("vectorsets match_any (int) recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "vectorsets int match_any recall {:.3} < 0.9",
        recall
    );
}

/// End-to-end BOOL filter: `flag == true`. This pins FIX 1 end-to-end — a bare
/// `.flag == true` FILTER expression is a SYNTAX ERROR on a live VSIM server and
/// fails the WHOLE query; the fix quotes it (`.flag == "true"`, matching the
/// JSON-string storage), so the query succeeds and returns the filtered NNs.
/// Reverting the quoting turns this test RED (query error → run fails / no
/// result), which is the intended regression teeth.
#[test]
fn test_binary_vectorsets_bool_filter() {
    wait_for_vectorsets();

    let name = "vectorsets-bool";
    let proj = common::write_bool_cosine_project("vs-bool", &vectorsets_config(name), 8);
    assert!(
        proj.matching_docs >= proj.top,
        "fixture must have >= top matching docs (got {})",
        proj.matching_docs
    );

    let port = test_port().to_string();
    assert!(
        common::run_binary(
            &proj.root,
            name,
            "vs-bool",
            TEST_HOST,
            &[("REDIS_PORT", port.as_str())],
        ),
        "vectorsets bool filter run failed (bare `.flag == true` is a VSIM syntax error?)"
    );

    let recall = common::read_recall(&proj.root, name);
    println!("vectorsets bool filter recall={:.3}", recall);
    assert!(
        recall >= 0.9,
        "vectorsets bool filter recall {:.3} < 0.9",
        recall
    );
}
