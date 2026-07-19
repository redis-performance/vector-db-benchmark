//! Minimal Vertex AI Vector Search gRPC wire types and synchronous worker.
//!
//! Public endpoints expose `google.cloud.aiplatform.v1.MatchService`, while
//! private VPC/PSC endpoints expose the lower-level
//! `google.cloud.aiplatform.container.v1.MatchService` on port 10000. Keeping
//! the small subset used by pure dense KNN here avoids a build-time `protoc`
//! dependency while remaining wire-compatible with Google's protobufs.

use std::time::Duration;

use tonic::codegen::http::uri::PathAndQuery;
use tonic::metadata::MetadataValue;
use tonic::transport::{Channel, ClientTlsConfig, Endpoint};
use tonic::Request;

#[derive(Clone, PartialEq, prost::Message)]
pub struct IndexDatapoint {
    #[prost(string, tag = "1")]
    pub datapoint_id: String,
    #[prost(float, repeated, tag = "2")]
    pub feature_vector: Vec<f32>,
    // Categorical + numeric filters carried on the query datapoint. Field tags
    // match `google.cloud.aiplatform.v1.IndexDatapoint` (restricts=4,
    // numeric_restricts=6); empty vecs encode to nothing.
    #[prost(message, repeated, tag = "4")]
    pub restricts: Vec<Restriction>,
    #[prost(message, repeated, tag = "6")]
    pub numeric_restricts: Vec<NumericRestriction>,
}

/// `IndexDatapoint.Restriction` — categorical (token) filter.
#[derive(Clone, PartialEq, prost::Message)]
pub struct Restriction {
    #[prost(string, tag = "1")]
    pub namespace: String,
    #[prost(string, repeated, tag = "2")]
    pub allow_list: Vec<String>,
    #[prost(string, repeated, tag = "3")]
    pub deny_list: Vec<String>,
}

/// `IndexDatapoint.NumericRestriction` — numeric filter. The `Value` oneof is
/// modeled as separate optional scalar fields (only one is ever set), which is
/// wire-identical to the proto oneof. `op` is the `Operator` enum as int32.
#[derive(Clone, PartialEq, prost::Message)]
pub struct NumericRestriction {
    #[prost(string, tag = "1")]
    pub namespace: String,
    #[prost(int64, optional, tag = "2")]
    pub value_int: Option<i64>,
    #[prost(float, optional, tag = "3")]
    pub value_float: Option<f32>,
    #[prost(double, optional, tag = "4")]
    pub value_double: Option<f64>,
    #[prost(int32, tag = "5")]
    pub op: i32,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct PublicQuery {
    #[prost(message, optional, tag = "1")]
    pub datapoint: Option<IndexDatapoint>,
    #[prost(int32, tag = "2")]
    pub neighbor_count: i32,
    #[prost(int32, tag = "4")]
    pub approximate_neighbor_count: i32,
    #[prost(double, tag = "5")]
    pub fraction_leaf_nodes_to_search_override: f64,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct PublicFindNeighborsRequest {
    #[prost(string, tag = "1")]
    pub index_endpoint: String,
    #[prost(string, tag = "2")]
    pub deployed_index_id: String,
    #[prost(message, repeated, tag = "3")]
    pub queries: Vec<PublicQuery>,
    #[prost(bool, tag = "4")]
    pub return_full_datapoint: bool,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct PublicNeighbor {
    #[prost(message, optional, tag = "1")]
    pub datapoint: Option<IndexDatapoint>,
    #[prost(double, tag = "2")]
    pub distance: f64,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct PublicNearestNeighbors {
    #[prost(string, tag = "1")]
    pub id: String,
    #[prost(message, repeated, tag = "2")]
    pub neighbors: Vec<PublicNeighbor>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct PublicFindNeighborsResponse {
    #[prost(message, repeated, tag = "1")]
    pub nearest_neighbors: Vec<PublicNearestNeighbors>,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct PrivateMatchRequest {
    #[prost(string, tag = "1")]
    pub deployed_index_id: String,
    #[prost(float, repeated, tag = "2")]
    pub float_val: Vec<f32>,
    #[prost(int32, tag = "3")]
    pub num_neighbors: i32,
    // Categorical filters. Tag matches container `MatchRequest.restricts` (4).
    #[prost(message, repeated, tag = "4")]
    pub restricts: Vec<Namespace>,
    #[prost(int32, tag = "6")]
    pub approx_num_neighbors: i32,
    #[prost(double, tag = "9")]
    pub fraction_leaf_nodes_to_search_override: f64,
    // Numeric filters. Tag matches container `MatchRequest.numeric_restricts` (11).
    #[prost(message, repeated, tag = "11")]
    pub numeric_restricts: Vec<NumericNamespace>,
}

/// Container `Namespace` — categorical filter for the private Match API.
#[derive(Clone, PartialEq, prost::Message)]
pub struct Namespace {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(string, repeated, tag = "2")]
    pub allow_tokens: Vec<String>,
    #[prost(string, repeated, tag = "3")]
    pub deny_tokens: Vec<String>,
}

/// Container `NumericNamespace` — numeric filter for the private Match API.
#[derive(Clone, PartialEq, prost::Message)]
pub struct NumericNamespace {
    #[prost(string, tag = "1")]
    pub name: String,
    #[prost(int64, optional, tag = "2")]
    pub value_int: Option<i64>,
    #[prost(float, optional, tag = "3")]
    pub value_float: Option<f32>,
    #[prost(double, optional, tag = "4")]
    pub value_double: Option<f64>,
    #[prost(int32, tag = "5")]
    pub op: i32,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct PrivateNeighbor {
    #[prost(string, tag = "1")]
    pub id: String,
    #[prost(double, tag = "2")]
    pub distance: f64,
}

#[derive(Clone, PartialEq, prost::Message)]
pub struct PrivateMatchResponse {
    #[prost(message, repeated, tag = "1")]
    pub neighbor: Vec<PrivateNeighbor>,
}

/// Transport-neutral filter model, produced by the query/condition parser and
/// the upload metadata mapper in `vertex.rs`. Serialized to REST JSON there and
/// to the gRPC wire types here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NumericOp {
    Less,
    LessEqual,
    Equal,
    GreaterEqual,
    Greater,
    /// Part of Vertex's `Operator` set; the benchmark filter model has no
    /// not-equal condition, so the parser never emits it.
    #[allow(dead_code)]
    NotEqual,
}

impl NumericOp {
    /// REST `Operator` enum string.
    pub fn as_rest(self) -> &'static str {
        match self {
            NumericOp::Less => "LESS",
            NumericOp::LessEqual => "LESS_EQUAL",
            NumericOp::Equal => "EQUAL",
            NumericOp::GreaterEqual => "GREATER_EQUAL",
            NumericOp::Greater => "GREATER",
            NumericOp::NotEqual => "NOT_EQUAL",
        }
    }

    /// Proto `Operator` enum value (shared by v1 NumericRestriction and the
    /// container NumericNamespace).
    pub fn as_proto(self) -> i32 {
        match self {
            NumericOp::Less => 1,
            NumericOp::LessEqual => 2,
            NumericOp::Equal => 3,
            NumericOp::GreaterEqual => 4,
            NumericOp::Greater => 5,
            NumericOp::NotEqual => 6,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum NumericValue {
    Int(i64),
    Double(f64),
}

#[derive(Clone, PartialEq, Debug)]
pub struct Restrict {
    pub namespace: String,
    pub allow_list: Vec<String>,
}

#[derive(Clone, PartialEq, Debug)]
pub struct NumericRestrict {
    pub namespace: String,
    /// `None` on the upload/datapoint side (a stored value carries no operator);
    /// `Some` on the query side.
    pub op: Option<NumericOp>,
    pub value: NumericValue,
}

#[derive(Clone, PartialEq, Debug, Default)]
pub struct VertexFilter {
    pub restricts: Vec<Restrict>,
    pub numeric_restricts: Vec<NumericRestrict>,
}

impl VertexFilter {
    pub fn is_empty(&self) -> bool {
        self.restricts.is_empty() && self.numeric_restricts.is_empty()
    }

    /// Public v1 wire restrictions (for `IndexDatapoint`).
    fn to_public(&self) -> (Vec<Restriction>, Vec<NumericRestriction>) {
        let restricts = self
            .restricts
            .iter()
            .map(|r| Restriction {
                namespace: r.namespace.clone(),
                allow_list: r.allow_list.clone(),
                deny_list: Vec::new(),
            })
            .collect();
        let numeric = self
            .numeric_restricts
            .iter()
            .map(|n| {
                let mut nr = NumericRestriction {
                    namespace: n.namespace.clone(),
                    op: n.op.map(|o| o.as_proto()).unwrap_or(0),
                    ..Default::default()
                };
                match n.value {
                    NumericValue::Int(i) => nr.value_int = Some(i),
                    NumericValue::Double(d) => nr.value_double = Some(d),
                }
                nr
            })
            .collect();
        (restricts, numeric)
    }

    /// Container wire restrictions (for the private `MatchRequest`).
    fn to_private(&self) -> (Vec<Namespace>, Vec<NumericNamespace>) {
        let restricts = self
            .restricts
            .iter()
            .map(|r| Namespace {
                name: r.namespace.clone(),
                allow_tokens: r.allow_list.clone(),
                deny_tokens: Vec::new(),
            })
            .collect();
        let numeric = self
            .numeric_restricts
            .iter()
            .map(|n| {
                let mut nn = NumericNamespace {
                    name: n.namespace.clone(),
                    op: n.op.map(|o| o.as_proto()).unwrap_or(0),
                    ..Default::default()
                };
                match n.value {
                    NumericValue::Int(i) => nn.value_int = Some(i),
                    NumericValue::Double(d) => nn.value_double = Some(d),
                }
                nn
            })
            .collect();
        (restricts, numeric)
    }
}

#[derive(Clone)]
pub enum VertexGrpcRequest {
    Public(PublicFindNeighborsRequest),
    Private(PrivateMatchRequest),
}

enum WorkerKind {
    Public {
        channel: Channel,
        authorization: MetadataValue<tonic::metadata::Ascii>,
        index_endpoint: String,
        deployed_index_id: String,
    },
    Private {
        channel: Channel,
        deployed_index_id: String,
    },
}

pub struct VertexGrpcWorker {
    runtime: tokio::runtime::Runtime,
    kind: WorkerKind,
}

impl VertexGrpcWorker {
    pub fn public(
        domain: &str,
        token: &str,
        index_endpoint: &str,
        deployed_index_id: &str,
    ) -> Result<Self, String> {
        let runtime = runtime()?;
        let endpoint = Endpoint::from_shared(format!("https://{domain}"))
            .map_err(|e| format!("invalid Vertex public gRPC endpoint: {e}"))?
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(60))
            .tcp_nodelay(true)
            .tls_config(
                ClientTlsConfig::new()
                    .with_native_roots()
                    .domain_name(domain.to_string()),
            )
            .map_err(|e| format!("Vertex public gRPC TLS configuration failed: {e}"))?;
        let channel = runtime
            .block_on(endpoint.connect())
            .map_err(|e| format!("Vertex public gRPC connect failed: {e:?}"))?;
        let authorization = format!("Bearer {token}")
            .parse()
            .map_err(|e| format!("invalid Vertex authorization metadata: {e}"))?;
        Ok(Self {
            runtime,
            kind: WorkerKind::Public {
                channel,
                authorization,
                index_endpoint: index_endpoint.to_string(),
                deployed_index_id: deployed_index_id.to_string(),
            },
        })
    }

    pub fn private(address: &str, deployed_index_id: &str) -> Result<Self, String> {
        let runtime = runtime()?;
        let target = if address.starts_with("http://") {
            address.to_string()
        } else if address.contains(':') {
            format!("http://{address}")
        } else {
            format!("http://{address}:10000")
        };
        let endpoint = Endpoint::from_shared(target)
            .map_err(|e| format!("invalid Vertex private gRPC endpoint: {e}"))?
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(60))
            .tcp_nodelay(true);
        let channel = runtime
            .block_on(endpoint.connect())
            .map_err(|e| format!("Vertex private gRPC connect failed: {e}"))?;
        Ok(Self {
            runtime,
            kind: WorkerKind::Private {
                channel,
                deployed_index_id: deployed_index_id.to_string(),
            },
        })
    }

    pub fn request(
        &self,
        vector: &[f32],
        top: usize,
        fraction_leaf_override: Option<f64>,
        approximate_neighbor_count: Option<i64>,
        filter: Option<&VertexFilter>,
    ) -> VertexGrpcRequest {
        match &self.kind {
            WorkerKind::Public {
                index_endpoint,
                deployed_index_id,
                ..
            } => {
                let (restricts, numeric_restricts) =
                    filter.map(|f| f.to_public()).unwrap_or_default();
                VertexGrpcRequest::Public(PublicFindNeighborsRequest {
                    index_endpoint: index_endpoint.clone(),
                    deployed_index_id: deployed_index_id.clone(),
                    queries: vec![PublicQuery {
                        datapoint: Some(IndexDatapoint {
                            datapoint_id: String::new(),
                            feature_vector: vector.to_vec(),
                            restricts,
                            numeric_restricts,
                        }),
                        neighbor_count: top as i32,
                        approximate_neighbor_count: approximate_neighbor_count.unwrap_or(0) as i32,
                        fraction_leaf_nodes_to_search_override: fraction_leaf_override
                            .unwrap_or(0.0),
                    }],
                    return_full_datapoint: false,
                })
            }
            WorkerKind::Private {
                deployed_index_id, ..
            } => {
                let (restricts, numeric_restricts) =
                    filter.map(|f| f.to_private()).unwrap_or_default();
                VertexGrpcRequest::Private(PrivateMatchRequest {
                    deployed_index_id: deployed_index_id.clone(),
                    float_val: vector.to_vec(),
                    num_neighbors: top as i32,
                    restricts,
                    approx_num_neighbors: approximate_neighbor_count.unwrap_or(0) as i32,
                    fraction_leaf_nodes_to_search_override: fraction_leaf_override.unwrap_or(0.0),
                    numeric_restricts,
                })
            }
        }
    }

    pub fn execute(&mut self, request: VertexGrpcRequest) -> Result<Vec<i64>, String> {
        match (&self.kind, request) {
            (
                WorkerKind::Public {
                    channel,
                    authorization,
                    ..
                },
                VertexGrpcRequest::Public(message),
            ) => self.runtime.block_on(public_find_neighbors(
                channel.clone(),
                authorization.clone(),
                message,
            )),
            (WorkerKind::Private { channel, .. }, VertexGrpcRequest::Private(message)) => self
                .runtime
                .block_on(private_match(channel.clone(), message)),
            _ => Err("Vertex gRPC worker/request transport mismatch".to_string()),
        }
    }
}

fn runtime() -> Result<tokio::runtime::Runtime, String> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|e| format!("failed to build Vertex gRPC runtime: {e}"))
}

async fn public_find_neighbors(
    channel: Channel,
    authorization: MetadataValue<tonic::metadata::Ascii>,
    message: PublicFindNeighborsRequest,
) -> Result<Vec<i64>, String> {
    let mut grpc = tonic::client::Grpc::new(channel);
    grpc.ready()
        .await
        .map_err(|e| format!("Vertex public gRPC service was not ready: {e}"))?;
    let mut request = Request::new(message);
    request
        .metadata_mut()
        .insert("authorization", authorization);
    let response: tonic::Response<PublicFindNeighborsResponse> = grpc
        .unary(
            request,
            PathAndQuery::from_static("/google.cloud.aiplatform.v1.MatchService/FindNeighbors"),
            tonic::codec::ProstCodec::default(),
        )
        .await
        .map_err(|e| format!("Vertex public gRPC FindNeighbors failed: {e}"))?;
    Ok(response
        .into_inner()
        .nearest_neighbors
        .into_iter()
        .next()
        .map(|nearest| {
            nearest
                .neighbors
                .into_iter()
                .filter_map(|neighbor| neighbor.datapoint?.datapoint_id.parse().ok())
                .collect()
        })
        .unwrap_or_default())
}

async fn private_match(channel: Channel, message: PrivateMatchRequest) -> Result<Vec<i64>, String> {
    let mut grpc = tonic::client::Grpc::new(channel);
    grpc.ready()
        .await
        .map_err(|e| format!("Vertex private gRPC service was not ready: {e}"))?;
    let response: tonic::Response<PrivateMatchResponse> = grpc
        .unary(
            Request::new(message),
            PathAndQuery::from_static("/google.cloud.aiplatform.container.v1.MatchService/Match"),
            tonic::codec::ProstCodec::default(),
        )
        .await
        .map_err(|e| format!("Vertex private gRPC Match failed: {e}"))?;
    Ok(response
        .into_inner()
        .neighbor
        .into_iter()
        .filter_map(|neighbor| neighbor.id.parse().ok())
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    #[test]
    fn public_request_uses_official_wire_tags() {
        let message = PublicFindNeighborsRequest {
            index_endpoint: "projects/p/locations/r/indexEndpoints/e".to_string(),
            deployed_index_id: "d".to_string(),
            queries: vec![PublicQuery {
                datapoint: Some(IndexDatapoint {
                    datapoint_id: String::new(),
                    feature_vector: vec![1.0, 2.0],
                    ..Default::default()
                }),
                neighbor_count: 10,
                approximate_neighbor_count: 100,
                fraction_leaf_nodes_to_search_override: 0.07,
            }],
            return_full_datapoint: false,
        };
        let decoded = PublicFindNeighborsRequest::decode(message.encode_to_vec().as_slice())
            .expect("request must round-trip");
        assert_eq!(decoded.queries[0].neighbor_count, 10);
        assert_eq!(decoded.queries[0].approximate_neighbor_count, 100);
    }

    #[test]
    fn private_request_uses_official_wire_tags() {
        let message = PrivateMatchRequest {
            deployed_index_id: "d".to_string(),
            float_val: vec![1.0, 2.0],
            num_neighbors: 10,
            approx_num_neighbors: 100,
            fraction_leaf_nodes_to_search_override: 0.07,
            ..Default::default()
        };
        let decoded = PrivateMatchRequest::decode(message.encode_to_vec().as_slice())
            .expect("request must round-trip");
        assert_eq!(decoded.num_neighbors, 10);
        assert_eq!(decoded.approx_num_neighbors, 100);
    }

    #[test]
    fn public_datapoint_restricts_round_trip() {
        let filter = VertexFilter {
            restricts: vec![Restrict {
                namespace: "color".to_string(),
                allow_list: vec!["red".to_string(), "blue".to_string()],
            }],
            numeric_restricts: vec![NumericRestrict {
                namespace: "size".to_string(),
                op: Some(NumericOp::GreaterEqual),
                value: NumericValue::Int(3),
            }],
        };
        let (restricts, numeric) = filter.to_public();
        let dp = IndexDatapoint {
            datapoint_id: String::new(),
            feature_vector: vec![1.0],
            restricts,
            numeric_restricts: numeric,
        };
        let decoded = IndexDatapoint::decode(dp.encode_to_vec().as_slice()).unwrap();
        assert_eq!(decoded.restricts[0].namespace, "color");
        assert_eq!(decoded.restricts[0].allow_list, vec!["red", "blue"]);
        assert_eq!(decoded.numeric_restricts[0].namespace, "size");
        assert_eq!(decoded.numeric_restricts[0].value_int, Some(3));
        assert_eq!(decoded.numeric_restricts[0].op, 4); // GREATER_EQUAL
    }

    #[test]
    fn private_request_restricts_round_trip() {
        let filter = VertexFilter {
            restricts: vec![Restrict {
                namespace: "color".to_string(),
                allow_list: vec!["red".to_string()],
            }],
            numeric_restricts: vec![NumericRestrict {
                namespace: "size".to_string(),
                op: Some(NumericOp::LessEqual),
                value: NumericValue::Double(7.5),
            }],
        };
        let (restricts, numeric) = filter.to_private();
        let msg = PrivateMatchRequest {
            deployed_index_id: "d".to_string(),
            float_val: vec![1.0],
            num_neighbors: 10,
            restricts,
            numeric_restricts: numeric,
            ..Default::default()
        };
        let decoded = PrivateMatchRequest::decode(msg.encode_to_vec().as_slice()).unwrap();
        assert_eq!(decoded.restricts[0].name, "color");
        assert_eq!(decoded.restricts[0].allow_tokens, vec!["red"]);
        assert_eq!(decoded.numeric_restricts[0].name, "size");
        assert_eq!(decoded.numeric_restricts[0].value_double, Some(7.5));
        assert_eq!(decoded.numeric_restricts[0].op, 2); // LESS_EQUAL
    }
}
