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
    #[prost(int32, tag = "6")]
    pub approx_num_neighbors: i32,
    #[prost(double, tag = "9")]
    pub fraction_leaf_nodes_to_search_override: f64,
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
    ) -> VertexGrpcRequest {
        match &self.kind {
            WorkerKind::Public {
                index_endpoint,
                deployed_index_id,
                ..
            } => VertexGrpcRequest::Public(PublicFindNeighborsRequest {
                index_endpoint: index_endpoint.clone(),
                deployed_index_id: deployed_index_id.clone(),
                queries: vec![PublicQuery {
                    datapoint: Some(IndexDatapoint {
                        datapoint_id: String::new(),
                        feature_vector: vector.to_vec(),
                    }),
                    neighbor_count: top as i32,
                    approximate_neighbor_count: approximate_neighbor_count.unwrap_or(0) as i32,
                    fraction_leaf_nodes_to_search_override: fraction_leaf_override.unwrap_or(0.0),
                }],
                return_full_datapoint: false,
            }),
            WorkerKind::Private {
                deployed_index_id, ..
            } => VertexGrpcRequest::Private(PrivateMatchRequest {
                deployed_index_id: deployed_index_id.clone(),
                float_val: vector.to_vec(),
                num_neighbors: top as i32,
                approx_num_neighbors: approximate_neighbor_count.unwrap_or(0) as i32,
                fraction_leaf_nodes_to_search_override: fraction_leaf_override.unwrap_or(0.0),
            }),
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
        };
        let decoded = PrivateMatchRequest::decode(message.encode_to_vec().as_slice())
            .expect("request must round-trip");
        assert_eq!(decoded.num_neighbors, 10);
        assert_eq!(decoded.approx_num_neighbors, 100);
    }
}
