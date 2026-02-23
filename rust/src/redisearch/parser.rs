use std::collections::HashMap;

/// Parsed filter condition: (query_string, params)
pub type FilterCondition = (String, HashMap<String, FilterValue>);

#[derive(Clone, Debug)]
pub enum FilterValue {
    Str(String),
    Int(i64),
    Float(f64),
}

impl FilterValue {
    pub fn to_redis_bytes(&self) -> Vec<u8> {
        match self {
            FilterValue::Str(s) => s.as_bytes().to_vec(),
            FilterValue::Int(n) => n.to_string().into_bytes(),
            FilterValue::Float(f) => f.to_string().into_bytes(),
        }
    }
}

/// RediSearch condition parser — mirrors the Python RedisConditionParser.
/// Converts benchmark's internal metadata conditions into FT.SEARCH filter syntax.
pub struct RedisConditionParser {
    counter: usize,
}

impl RedisConditionParser {
    pub fn new() -> Self {
        Self { counter: 0 }
    }

    /// Parse meta_conditions dict into (filter_string, params).
    /// Returns None if no conditions.
    pub fn parse(
        &mut self,
        meta_conditions: Option<&MetaConditions>,
    ) -> Option<FilterCondition> {
        let mc = meta_conditions?;
        if mc.and_conditions.is_empty() && mc.or_conditions.is_empty() {
            return None;
        }

        let and_subfilters = self.create_condition_subfilters(&mc.and_conditions);
        let or_subfilters = self.create_condition_subfilters(&mc.or_conditions);

        Some(self.build_condition(&and_subfilters, &or_subfilters))
    }

    fn build_condition(
        &self,
        and_subfilters: &[FilterCondition],
        or_subfilters: &[FilterCondition],
    ) -> FilterCondition {
        let mut clauses = Vec::new();
        let mut all_params = HashMap::new();

        if !and_subfilters.is_empty() {
            let and_clauses: Vec<&str> = and_subfilters.iter().map(|(c, _)| c.as_str()).collect();
            clauses.push(format!("({})", and_clauses.join(" ")));
            for (_, params) in and_subfilters {
                all_params.extend(params.clone());
            }
        }

        if !or_subfilters.is_empty() {
            let or_clauses: Vec<&str> = or_subfilters.iter().map(|(c, _)| c.as_str()).collect();
            clauses.push(format!("({})", or_clauses.join(" | ")));
            for (_, params) in or_subfilters {
                all_params.extend(params.clone());
            }
        }

        (clauses.join(" "), all_params)
    }

    fn create_condition_subfilters(
        &mut self,
        entries: &[ConditionEntry],
    ) -> Vec<FilterCondition> {
        let mut output = Vec::new();
        for entry in entries {
            let condition = match &entry.filter {
                FilterSpec::Match { value } => {
                    self.build_exact_match_filter(&entry.field_name, value)
                }
                FilterSpec::Range { lt, gt, lte, gte } => {
                    self.build_range_filter(&entry.field_name, lt, gt, lte, gte)
                }
                FilterSpec::Geo { lat, lon, radius } => {
                    self.build_geo_filter(&entry.field_name, *lat, *lon, *radius)
                }
            };
            output.push(condition);
        }
        output
    }

    fn build_exact_match_filter(
        &mut self,
        field_name: &str,
        value: &MatchValue,
    ) -> FilterCondition {
        let param_name = format!("{}_{}", field_name, self.counter);
        self.counter += 1;

        match value {
            MatchValue::Str(s) => {
                let clause = format!("@{}:{{${}}}", field_name, param_name);
                let mut params = HashMap::new();
                params.insert(param_name, FilterValue::Str(s.clone()));
                (clause, params)
            }
            MatchValue::Int(n) => {
                let clause = format!("@{}:[${} ${}]", field_name, param_name, param_name);
                let mut params = HashMap::new();
                params.insert(param_name, FilterValue::Int(*n));
                (clause, params)
            }
            MatchValue::Float(f) => {
                let clause = format!("@{}:[${} ${}]", field_name, param_name, param_name);
                let mut params = HashMap::new();
                params.insert(param_name, FilterValue::Float(*f));
                (clause, params)
            }
        }
    }

    fn build_range_filter(
        &mut self,
        field_name: &str,
        lt: &Option<f64>,
        gt: &Option<f64>,
        lte: &Option<f64>,
        gte: &Option<f64>,
    ) -> FilterCondition {
        let param_prefix = format!("{}_{}", field_name, self.counter);
        self.counter += 1;

        let mut params = HashMap::new();
        let mut clauses = Vec::new();

        if let Some(v) = lt {
            let key = format!("{}_lt", param_prefix);
            params.insert(key.clone(), FilterValue::Float(*v));
            clauses.push(format!("@{}:[-inf (${})]", field_name, key));
        }
        if let Some(v) = gt {
            let key = format!("{}_gt", param_prefix);
            params.insert(key.clone(), FilterValue::Float(*v));
            clauses.push(format!("@{}:[(${}  +inf]", field_name, key));
        }
        if let Some(v) = lte {
            let key = format!("{}_lte", param_prefix);
            params.insert(key.clone(), FilterValue::Float(*v));
            clauses.push(format!("@{}:[-inf ${}]", field_name, key));
        }
        if let Some(v) = gte {
            let key = format!("{}_gte", param_prefix);
            params.insert(key.clone(), FilterValue::Float(*v));
            clauses.push(format!("@{}:[${} +inf]", field_name, key));
        }

        (clauses.join(" "), params)
    }

    fn build_geo_filter(
        &mut self,
        field_name: &str,
        lat: f64,
        lon: f64,
        radius: f64,
    ) -> FilterCondition {
        let param_prefix = format!("{}_{}", field_name, self.counter);
        self.counter += 1;

        // Clamp latitude to Redis valid range
        let lat = lat.clamp(-85.05112878, 85.05112878);

        let lon_key = format!("{}_lon", param_prefix);
        let lat_key = format!("{}_lat", param_prefix);
        let radius_key = format!("{}_radius", param_prefix);

        let mut params = HashMap::new();
        params.insert(lon_key.clone(), FilterValue::Float(lon));
        params.insert(lat_key.clone(), FilterValue::Float(lat));
        params.insert(radius_key.clone(), FilterValue::Float(radius));

        let clause = format!(
            "@{}:[${} ${} ${} m]",
            field_name, lon_key, lat_key, radius_key
        );
        (clause, params)
    }
}

// ---- Data structures for parsed conditions ----

#[derive(Clone, Debug)]
pub enum MatchValue {
    Str(String),
    Int(i64),
    Float(f64),
}

#[derive(Clone, Debug)]
pub enum FilterSpec {
    Match {
        value: MatchValue,
    },
    Range {
        lt: Option<f64>,
        gt: Option<f64>,
        lte: Option<f64>,
        gte: Option<f64>,
    },
    Geo {
        lat: f64,
        lon: f64,
        radius: f64,
    },
}

#[derive(Clone, Debug)]
pub struct ConditionEntry {
    pub field_name: String,
    pub filter: FilterSpec,
}

#[derive(Clone, Debug)]
pub struct MetaConditions {
    pub and_conditions: Vec<ConditionEntry>,
    pub or_conditions: Vec<ConditionEntry>,
}

/// Extract MetaConditions from a Python dict (or None).
pub fn extract_meta_conditions(meta_conditions: &Bound<'_, pyo3::PyAny>) -> Option<MetaConditions> {
    if meta_conditions.is_none() {
        return None;
    }

    let and_conditions = extract_condition_list(meta_conditions, "and");
    let or_conditions = extract_condition_list(meta_conditions, "or");

    if and_conditions.is_empty() && or_conditions.is_empty() {
        return None;
    }

    Some(MetaConditions {
        and_conditions,
        or_conditions,
    })
}

fn extract_condition_list(
    meta_conditions: &Bound<'_, pyo3::PyAny>,
    key: &str,
) -> Vec<ConditionEntry> {
    let mut entries = Vec::new();

    let list = match meta_conditions.call_method1("get", (key,)) {
        Ok(v) if !v.is_none() => v,
        _ => return entries,
    };

    let iter = match list.call_method0("__iter__") {
        Ok(it) => it,
        _ => return entries,
    };

    loop {
        match iter.call_method0("__next__") {
            Ok(item) => {
                // item is a dict like {"field_name": {"match": {"value": ...}}}
                let items_iter = match item.call_method0("items") {
                    Ok(items) => match items.call_method0("__iter__") {
                        Ok(it) => it,
                        _ => continue,
                    },
                    _ => continue,
                };

                loop {
                    match items_iter.call_method0("__next__") {
                        Ok(kv) => {
                            let field_name: String = match kv.get_item(0) {
                                Ok(k) => match k.extract() {
                                    Ok(s) => s,
                                    _ => continue,
                                },
                                _ => continue,
                            };
                            let filters = match kv.get_item(1) {
                                Ok(v) => v,
                                _ => continue,
                            };

                            // filters is a dict like {"match": {"value": ...}} or {"range": {...}}
                            let filter_items_iter = match filters.call_method0("items") {
                                Ok(items) => match items.call_method0("__iter__") {
                                    Ok(it) => it,
                                    _ => continue,
                                },
                                _ => continue,
                            };

                            loop {
                                match filter_items_iter.call_method0("__next__") {
                                    Ok(filter_kv) => {
                                        let filter_type: String = match filter_kv.get_item(0) {
                                            Ok(k) => match k.extract() {
                                                Ok(s) => s,
                                                _ => continue,
                                            },
                                            _ => continue,
                                        };
                                        let criteria = match filter_kv.get_item(1) {
                                            Ok(v) => v,
                                            _ => continue,
                                        };

                                        let filter = match filter_type.as_str() {
                                            "match" => {
                                                let val = match criteria.call_method1("get", ("value",)) {
                                                    Ok(v) => v,
                                                    _ => continue,
                                                };
                                                if let Ok(s) = val.extract::<String>() {
                                                    FilterSpec::Match {
                                                        value: MatchValue::Str(s),
                                                    }
                                                } else if let Ok(n) = val.extract::<i64>() {
                                                    FilterSpec::Match {
                                                        value: MatchValue::Int(n),
                                                    }
                                                } else if let Ok(f) = val.extract::<f64>() {
                                                    FilterSpec::Match {
                                                        value: MatchValue::Float(f),
                                                    }
                                                } else {
                                                    continue;
                                                }
                                            }
                                            "range" => {
                                                let lt = extract_optional_f64(&criteria, "lt");
                                                let gt = extract_optional_f64(&criteria, "gt");
                                                let lte = extract_optional_f64(&criteria, "lte");
                                                let gte = extract_optional_f64(&criteria, "gte");
                                                FilterSpec::Range { lt, gt, lte, gte }
                                            }
                                            "geo" => {
                                                let lat = extract_optional_f64(&criteria, "lat")
                                                    .unwrap_or(0.0);
                                                let lon = extract_optional_f64(&criteria, "lon")
                                                    .unwrap_or(0.0);
                                                let radius =
                                                    extract_optional_f64(&criteria, "radius")
                                                        .unwrap_or(0.0);
                                                FilterSpec::Geo { lat, lon, radius }
                                            }
                                            _ => continue,
                                        };

                                        entries.push(ConditionEntry {
                                            field_name: field_name.clone(),
                                            filter,
                                        });
                                    }
                                    Err(_) => break,
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
            }
            Err(_) => break,
        }
    }

    entries
}

fn extract_optional_f64(obj: &Bound<'_, pyo3::PyAny>, key: &str) -> Option<f64> {
    match obj.call_method1("get", (key,)) {
        Ok(v) if !v.is_none() => v.extract::<f64>().ok(),
        _ => None,
    }
}

use pyo3::prelude::*;
