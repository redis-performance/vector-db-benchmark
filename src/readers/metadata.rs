//! Metadata types for dataset records.
//!
//! Provides types for representing metadata from dataset records (JSONL payloads).

/// Metadata extracted from JSON, ready for use in engine uploads.
#[derive(Clone, Debug)]
pub struct MetadataItem {
    pub fields: Vec<(String, MetadataValue)>,
}

#[derive(Clone, Debug)]
pub enum MetadataValue {
    String(String),
    /// A JSON integer that fits in `i64`. Kept typed (rather than stringified)
    /// so engines emit a real numeric value — otherwise numeric exact-match,
    /// range, and `match_any` filters silently match nothing on engines that
    /// store the payload verbatim (e.g. Qdrant, Milvus). See issue #87.
    Int(i64),
    /// A JSON non-integer number, kept typed for the same reason as [`Int`].
    Float(f64),
    Labels(Vec<String>),
    Geo {
        lon: f64,
        lat: f64,
    },
}

/// Parse metadata from a serde_json Value.
pub fn parse_metadata_from_json(json_value: serde_json::Value) -> Option<MetadataItem> {
    if let serde_json::Value::Object(map) = json_value {
        let mut fields: Vec<(String, MetadataValue)> = Vec::new();

        for (key, value) in map {
            match value {
                serde_json::Value::String(s) => {
                    fields.push((key, MetadataValue::String(s)));
                }
                serde_json::Value::Number(n) => {
                    if let Some(i) = n.as_i64() {
                        fields.push((key, MetadataValue::Int(i)));
                    } else if let Some(f) = n.as_f64() {
                        fields.push((key, MetadataValue::Float(f)));
                    } else {
                        // u64 above i64::MAX with no exact f64 form: keep the raw
                        // digits rather than lose precision.
                        fields.push((key, MetadataValue::String(n.to_string())));
                    }
                }
                serde_json::Value::Array(arr) => {
                    if key == "labels" {
                        let labels: Vec<String> = arr
                            .iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect();
                        fields.push((key, MetadataValue::Labels(labels)));
                    }
                }
                serde_json::Value::Object(obj) => {
                    if let (Some(lon), Some(lat)) = (
                        obj.get("lon").and_then(|v| v.as_f64()),
                        obj.get("lat").and_then(|v| v.as_f64()),
                    ) {
                        fields.push((key, MetadataValue::Geo { lon, lat }));
                    }
                }
                serde_json::Value::Bool(b) => {
                    fields.push((key, MetadataValue::String(b.to_string())));
                }
                serde_json::Value::Null => {}
            }
        }

        Some(MetadataItem { fields })
    } else {
        None
    }
}

#[cfg(test)]
mod parse_tests {
    use super::*;

    #[test]
    fn integers_and_floats_are_typed() {
        let m = parse_metadata_from_json(serde_json::json!({"i": 7, "f": 1.5})).unwrap();
        let get = |k: &str| {
            m.fields
                .iter()
                .find(|(n, _)| n == k)
                .map(|(_, v)| v)
                .unwrap()
        };
        assert!(matches!(get("i"), MetadataValue::Int(7)));
        assert!(matches!(get("f"), MetadataValue::Float(x) if *x == 1.5));
    }

    #[test]
    fn negative_integer_is_int() {
        let m = parse_metadata_from_json(serde_json::json!({"i": -3})).unwrap();
        assert!(matches!(m.fields[0].1, MetadataValue::Int(-3)));
    }

    #[test]
    fn u64_above_i64_max_falls_back() {
        // u64::MAX does not fit in i64; keep it representable (lossy f64) or as
        // raw digits rather than silently truncating.
        let big = u64::MAX;
        let m = parse_metadata_from_json(serde_json::json!({ "n": big })).unwrap();
        match &m.fields[0].1 {
            MetadataValue::Float(_) | MetadataValue::String(_) => {}
            other => panic!(
                "expected Float/String fallback for huge u64, got {:?}",
                other
            ),
        }
    }
}
