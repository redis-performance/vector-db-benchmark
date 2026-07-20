//! Parsers for **untrusted** inputs shared by the binary engines.
//!
//! These functions ingest attacker-, corruption-, or server-controlled data:
//! - [`parse_info`] parses raw `INFO` text returned by any Redis-wire server
//!   (Redis / Valkey / Dragonfly / ElastiCache / MemoryStore).
//! - [`datetime_to_epoch_secs`] parses dataset datetime filter values.
//! - [`parse_ft_search_response`] parses an `FT.SEARCH` reply (`redis::Value`).
//!
//! They live in the library crate (not the bin) so they can be reached by
//! coverage-guided fuzzing. The engine binaries re-use these exact
//! implementations, so behavior is identical to when they lived in the bin.

use serde_json::{Map, Value as Json};

/// Parse a Redis `INFO` reply into `{ "<section>": { "<key>": "<value>" } }`.
///
/// A `# Section` line opens a new section; `key:value` lines populate the
/// current section; blank and `#`-only lines are skipped. Values are kept as
/// strings (the first `:` splits key from value, so values may contain colons).
pub fn parse_info(info: &str) -> Json {
    let mut root = Map::new();
    let mut current = String::from("default");
    let mut section = Map::new();
    for raw in info.lines() {
        let line = raw.trim_end_matches('\r');
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Some(name) = trimmed.strip_prefix("# ") {
            if !section.is_empty() {
                root.insert(current.clone(), Json::Object(std::mem::take(&mut section)));
            }
            current = name.trim().to_string();
            continue;
        }
        if trimmed.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once(':') {
            section.insert(k.to_string(), Json::String(v.to_string()));
        }
    }
    if !section.is_empty() {
        root.insert(current, Json::Object(section));
    }
    Json::Object(root)
}

/// Parse an ISO-8601 / RFC 3339 timestamp to epoch **seconds**. Returns `None`
/// for non-datetime strings (e.g. a plain numeric-epoch string), letting callers
/// fall back to numeric handling.
pub fn datetime_to_epoch_secs(s: &str) -> Option<f64> {
    // RFC-3339 (with offset / `Z`) first.
    if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(s) {
        return Some(dt.timestamp() as f64);
    }
    // Naive datetime (no offset) → interpret as UTC. Accepts both the `T` and
    // space separators (upstream tolerates these; RFC-3339 does not).
    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"] {
        if let Ok(ndt) = chrono::NaiveDateTime::parse_from_str(s, fmt) {
            return Some(ndt.and_utc().timestamp() as f64);
        }
    }
    // Date only → midnight UTC.
    if let Ok(nd) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Some(nd.and_hms_opt(0, 0, 0)?.and_utc().timestamp() as f64);
    }
    None
}

/// Recover the numeric doc id from an FT.SEARCH key under the RESP2 arm, where
/// the id is positionally guaranteed present. Since issue #151-4 the doc key is
/// `"<config>:<id>"` (per-config keyspace prefix), so the id is the tail after
/// the last `:`. A bare `"42"` (no prefix, as older uploads and the Class-A
/// integration tests write) still yields `42`, keeping this backward-compatible.
/// An unparseable tail falls back to `0` (mirrors the old `value_as_i64`).
pub fn doc_key_to_id(key: &str) -> i64 {
    key.rsplit(':')
        .next()
        .unwrap_or(key)
        .parse::<i64>()
        .unwrap_or(0)
}

/// Recover the numeric doc id from an FT.SEARCH key under the RESP3 arm, where a
/// missing/unparseable id must be SKIPPED (returns `None`) rather than emitted as
/// a phantom `id=0` hit that would pollute recall. Do NOT collapse this into
/// `doc_key_to_id`: the `Option` semantics are load-bearing here.
pub fn doc_key_to_id_opt(key: &str) -> Option<i64> {
    key.rsplit(':').next().unwrap_or(key).parse::<i64>().ok()
}

/// Parse an FT.SEARCH reply under EITHER protocol:
/// - RESP2: a flat array `[count, id, fields, id, fields, ...]`
/// - RESP3: a map `{ results: [ { id, extra_attributes: { vector_score, .. }, .. } ], .. }`
///
/// The engine connects with RESP2 by default, but a caller can negotiate RESP3
/// (e.g. `REDIS_URI=redis://host/?protocol=resp3`), which returns a completely
/// different shape. Handling both keeps recall correct regardless of protocol.
///
/// Returns `(id, score)` hits. The `Result` never actually errors today (any
/// unexpected shape yields no hits); it is kept so callers stay uniform.
pub fn parse_ft_search_response(response: &redis::Value) -> Result<Vec<(i64, f64)>, String> {
    match response {
        redis::Value::Array(items) => Ok(parse_ft_search_resp2(items)),
        redis::Value::Map(pairs) => Ok(parse_ft_search_resp3(pairs)),
        // Nil (no index/empty) or any unexpected shape → no hits.
        _ => Ok(Vec::new()),
    }
}

/// RESP2 flat array: `[count, id, fields, id, fields, ...]`.
fn parse_ft_search_resp2(response: &[redis::Value]) -> Vec<(i64, f64)> {
    let mut results = Vec::new();
    // First element is total count.
    let mut i = 1;
    while i < response.len() {
        // The reply carries the doc KEY (e.g. "cfg:42"), not a bare id; recover
        // the trailing numeric id. Missing string → 0 (positionally present).
        let id = value_as_string(&response[i])
            .map(|s| doc_key_to_id(&s))
            .unwrap_or(0);
        i += 1;

        if i < response.len() {
            let score = match &response[i] {
                redis::Value::Array(fields) => extract_vector_score(fields),
                _ => 0.0,
            };
            results.push((id, score));
            i += 1;
        }
    }
    results
}

/// RESP3 map: top-level map with a `results` array; each result is a map with an
/// `id` and an `extra_attributes` map carrying `vector_score`.
fn parse_ft_search_resp3(pairs: &[(redis::Value, redis::Value)]) -> Vec<(i64, f64)> {
    let docs = match pairs
        .iter()
        .find(|(k, _)| value_as_string(k).as_deref() == Some("results"))
        .map(|(_, v)| v)
    {
        Some(redis::Value::Array(docs)) => docs.as_slice(),
        _ => return Vec::new(),
    };

    let mut out = Vec::with_capacity(docs.len());
    for doc in docs {
        let redis::Value::Map(fields) = doc else {
            continue;
        };
        // A doc whose `id` is missing or cannot be parsed to an integer is
        // skipped (mirrors the RESP2 trailing-id drop) rather than emitted as a
        // phantom id=0 hit.
        let mut id: Option<i64> = None;
        let mut score = 0.0f64;
        for (k, v) in fields {
            match value_as_string(k).as_deref() {
                Some("id") => id = value_as_string(v).and_then(|s| doc_key_to_id_opt(&s)),
                Some("extra_attributes") => {
                    if let redis::Value::Map(attrs) = v {
                        for (ak, av) in attrs {
                            if value_as_string(ak).as_deref() == Some("vector_score") {
                                score = value_as_string(av)
                                    .and_then(|s| s.parse().ok())
                                    .unwrap_or(0.0);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        if let Some(id) = id {
            out.push((id, score));
        }
    }
    out
}

/// Best-effort string view of a RESP value (BulkString/SimpleString).
fn value_as_string(v: &redis::Value) -> Option<String> {
    match v {
        redis::Value::BulkString(b) => Some(String::from_utf8_lossy(b).into_owned()),
        redis::Value::SimpleString(s) => Some(s.clone()),
        _ => None,
    }
}

/// Parse a RESP value as an i64 doc id (bulk/simple string or integer).
pub fn value_as_i64(v: &redis::Value) -> i64 {
    match v {
        redis::Value::BulkString(data) => String::from_utf8_lossy(data).parse::<i64>().unwrap_or(0),
        redis::Value::Int(n) => *n,
        redis::Value::SimpleString(s) => s.parse().unwrap_or(0),
        _ => 0,
    }
}

/// Extract `vector_score` from an FT.SEARCH field-values array.
pub fn extract_vector_score(fields: &[redis::Value]) -> f64 {
    // Fields are in format: [field_name, field_value, ...]
    let mut i = 0;
    while i + 1 < fields.len() {
        if let redis::Value::BulkString(name) = &fields[i] {
            if name == b"vector_score" {
                if let redis::Value::BulkString(val) = &fields[i + 1] {
                    return String::from_utf8_lossy(val).parse::<f64>().unwrap_or(0.0);
                }
            }
        }
        i += 2;
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use redis::Value;

    fn bulk(s: &str) -> Value {
        Value::BulkString(s.as_bytes().to_vec())
    }

    #[test]
    fn parse_info_builds_nested_sections() {
        let info = "# Server\r\nredis_version:8.8.0\r\nos:Linux\r\n\r\n# Memory\r\nused_memory:1048576\r\n#\r\n# Keyspace\r\ndb0:keys=3,expires=0,avg_ttl=0\r\n";
        let parsed = parse_info(info);
        assert_eq!(parsed["Server"]["redis_version"], "8.8.0");
        assert_eq!(parsed["Memory"]["used_memory"], "1048576");
        assert_eq!(parsed["Keyspace"]["db0"], "keys=3,expires=0,avg_ttl=0");
        assert_eq!(parsed.as_object().unwrap().len(), 3);
    }

    #[test]
    fn parse_info_empty_is_empty_object() {
        assert_eq!(parse_info("").as_object().unwrap().len(), 0);
    }

    #[test]
    fn datetime_parses_rfc3339_and_rejects_plain() {
        assert_eq!(
            datetime_to_epoch_secs("2021-01-01T00:00:00Z").map(|f| f as i64),
            Some(1609459200)
        );
        assert!(datetime_to_epoch_secs("not-a-date").is_none());
        assert!(datetime_to_epoch_secs("1609459200").is_none());
    }

    #[test]
    fn ft_search_resp2_reads_id_score_pairs() {
        let resp = Value::Array(vec![
            Value::Int(1),
            bulk("42"),
            Value::Array(vec![bulk("vector_score"), bulk("0.75")]),
        ]);
        assert_eq!(parse_ft_search_response(&resp).unwrap(), vec![(42, 0.75)]);
    }

    #[test]
    fn ft_search_empty_and_nil() {
        assert_eq!(
            parse_ft_search_response(&Value::Array(vec![])).unwrap(),
            vec![]
        );
        assert_eq!(parse_ft_search_response(&Value::Nil).unwrap(), vec![]);
    }

    // ---- Regression tests for fuzzer-found crashes (see notes in report). ----

    #[test]
    fn doc_key_to_id_recovers_prefixed_and_bare() {
        // Bare (pre-#151-4 uploads + Class-A tests) stays identity.
        assert_eq!(doc_key_to_id("42"), 42);
        // Per-config prefixed key: tail after the last ':'.
        assert_eq!(doc_key_to_id("redis-m-16-ef-64:42"), 42);
        // Pathological colon shapes never panic; unparseable tail → 0.
        assert_eq!(doc_key_to_id("a:b:c"), 0);
        assert_eq!(doc_key_to_id("::"), 0);
        assert_eq!(doc_key_to_id(""), 0);
        assert_eq!(doc_key_to_id("a:b:7"), 7);
    }

    #[test]
    fn doc_key_to_id_opt_skips_unparseable() {
        // resp3 arm PRESERVES Option so a bad id is skipped, not emitted as 0.
        assert_eq!(doc_key_to_id_opt("42"), Some(42));
        assert_eq!(doc_key_to_id_opt("cfg:42"), Some(42));
        assert_eq!(doc_key_to_id_opt("cfg:x"), None);
        assert_eq!(doc_key_to_id_opt(""), None);
        assert_eq!(doc_key_to_id_opt("a:b:c"), None);
    }

    #[test]
    fn ft_search_resp2_recovers_id_from_prefixed_key() {
        // A prefixed key ("cfg:42") must resolve to id 42, not 0.
        let resp = Value::Array(vec![
            Value::Int(1),
            bulk("cfg:42"),
            Value::Array(vec![bulk("vector_score"), bulk("0.5")]),
        ]);
        assert_eq!(parse_ft_search_response(&resp).unwrap(), vec![(42, 0.5)]);
    }

    #[test]
    fn parse_info_no_panic_on_arbitrary_bytes() {
        // A blob of lone `:` and `#` fragments must never panic.
        for s in ["::::", "#", "# ", ":", "\r\r\r", "#\r:x", "a:b:c:d"] {
            let _ = parse_info(s);
        }
    }
}
