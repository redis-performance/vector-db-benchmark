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
    Labels(Vec<String>),
    Geo { lon: f64, lat: f64 },
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
                    fields.push((key, MetadataValue::String(n.to_string())));
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
