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

impl MetadataValue {
    /// Coerce a stray numeric value to a string when the dataset schema declares
    /// the field as a non-numeric scalar (`keyword`/`text`/`uuid`/`bool`).
    ///
    /// The reader types every JSON number as [`MetadataValue::Int`]/[`Float`],
    /// but engines declare each column/property/index type from the dataset
    /// schema. If a field is declared `keyword` yet a record carries a numeric
    /// value (e.g. an all-digit SKU or zip code), emitting a native number would
    /// make strict engines reject the insert (Milvus VarChar, Weaviate `text`)
    /// or silently fail to match (Qdrant keyword index). Coercing to a string
    /// keeps storage aligned with the declared type. Numeric-declared fields
    /// (`int`/`float`) and non-numeric variants pass through unchanged.
    pub fn coerce_for_schema(
        &self,
        schema_type: Option<&str>,
    ) -> std::borrow::Cow<'_, MetadataValue> {
        use std::borrow::Cow;
        match (self, schema_type) {
            (MetadataValue::Int(n), Some("keyword" | "text" | "uuid" | "bool")) => {
                Cow::Owned(MetadataValue::String(n.to_string()))
            }
            (MetadataValue::Float(f), Some("keyword" | "text" | "uuid" | "bool")) => {
                Cow::Owned(MetadataValue::String(f.to_string()))
            }
            // No coercion needed: borrow (no clone on the upload hot path).
            _ => Cow::Borrowed(self),
        }
    }
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
                    } else if n.as_u64().is_some() {
                        // u64 above i64::MAX: there is no lossless integer variant,
                        // and `as_f64()` would round (e.g. a 64-bit ID). Keep the
                        // exact digits as a string so exact-match filters still work.
                        fields.push((key, MetadataValue::String(n.to_string())));
                    } else if let Some(f) = n.as_f64() {
                        fields.push((key, MetadataValue::Float(f)));
                    } else {
                        // Not representable at all (shouldn't happen for valid JSON).
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
    fn u64_above_i64_max_kept_as_exact_string() {
        // u64::MAX doesn't fit i64 and rounds as f64; it must be kept as exact
        // digits so a 64-bit ID still matches exactly (regression: the old
        // `as_f64` path rounded 18446744073709551615 -> 18446744073709552000).
        let big = u64::MAX;
        let m = parse_metadata_from_json(serde_json::json!({ "n": big })).unwrap();
        match &m.fields[0].1 {
            MetadataValue::String(s) => assert_eq!(s, "18446744073709551615"),
            other => panic!("expected exact String for huge u64, got {:?}", other),
        }
    }

    #[test]
    fn coerce_for_schema_stringifies_numeric_keyword() {
        // A numeric value under a keyword-declared field must become a string so
        // strict engines (Milvus VarChar, Weaviate text, Qdrant keyword) accept
        // and match it. Numeric-declared fields keep their typed value.
        assert!(matches!(
            MetadataValue::Int(42).coerce_for_schema(Some("keyword")).as_ref(),
            MetadataValue::String(s) if s == "42"
        ));
        assert!(matches!(
            MetadataValue::Float(3.5).coerce_for_schema(Some("text")).as_ref(),
            MetadataValue::String(s) if s == "3.5"
        ));
        assert!(matches!(
            MetadataValue::Int(42)
                .coerce_for_schema(Some("int"))
                .as_ref(),
            MetadataValue::Int(42)
        ));
        assert!(matches!(
            MetadataValue::Float(3.5)
                .coerce_for_schema(Some("float"))
                .as_ref(),
            MetadataValue::Float(_)
        ));
        // No schema hint -> leave typed (best-effort numeric).
        assert!(matches!(
            MetadataValue::Int(7).coerce_for_schema(None).as_ref(),
            MetadataValue::Int(7)
        ));
    }
}
