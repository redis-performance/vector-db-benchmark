from opensearchpy import NotFoundError

from benchmark.dataset import Dataset
from engine.base_client import IncompatibilityError
from engine.base_client.configure import BaseConfigurator
from engine.base_client.distances import Distance
from engine.clients.opensearch.config import (
    OPENSEARCH_DELETE_INDEX_TIMEOUT,
    OPENSEARCH_INDEX,
    get_opensearch_client,
)


class OpenSearchConfigurator(BaseConfigurator):
    DISTANCE_MAPPING = {
        Distance.L2: "l2",
        Distance.COSINE: "cosinesimil",
        # innerproduct (supported for Lucene in OpenSearch version 2.13 and later)
        Distance.DOT: "innerproduct",
    }
    INDEX_TYPE_MAPPING = {
        "int": "long",
        "geo": "geo_point",
    }

    def __init__(self, host, collection_params: dict, connection_params: dict):
        super().__init__(host, collection_params, connection_params)
        self.client = get_opensearch_client(host, connection_params)

    def clean(self):
        try:
            self.client.indices.delete(
                index=OPENSEARCH_INDEX,
                params={
                    "timeout": OPENSEARCH_DELETE_INDEX_TIMEOUT,
                },
            )
        except NotFoundError:
            pass

    def recreate(self, dataset: Dataset, collection_params):
        # The knn_vector data type supports a vector of floats that can have a dimension count of up to 16,000 for the NMSLIB, Faiss, and Lucene engines, as set by the dimension mapping parameter.
        # Source: https://opensearch.org/docs/latest/search-plugins/knn/approximate-knn/
        if dataset.config.vector_size > 16000:
            raise IncompatibilityError

        nodes_stats_res = self.client.nodes.info(filter_path="nodes.*.roles,nodes.*.os")
        nodes_data = nodes_stats_res.get("nodes")

        data_node_count = 0
        total_processors = 0
        for node_id in nodes_data:
            node_info = nodes_data.get(node_id)
            roles = node_info["roles"]
            os_info = node_info["os"]
            if 'data' in roles:
                data_node_count += 1
                total_processors += int(os_info['allocated_processors'])

        processors_per_node = total_processors // data_node_count

        index_thread_qty = max(1, processors_per_node // 2)

        cluster_settings_body = {
            "persistent": {
                "knn.memory.circuit_breaker.limit": "75%",
                "knn.algo_param.index_thread_qty": index_thread_qty
            }
        }

        self.client.cluster.put_settings(cluster_settings_body)

        index_settings = {
            "knn": True,
            "number_of_replicas": 0,
            "refresh_interval": -1,  # no refresh is required because we index all the data at once
        }
        index_config = collection_params.get("index")

        # if we specify the number_of_shards on the config, enforce it. otherwise use the default
        if index_config is not None and index_config.has_key("number_of_shards"):
            index_settings["number_of_shards"] = 1

        field_config = self._prepare_fields_config(dataset)

        engine = "faiss"
        if field_config == {} or field_config is None:
            engine = "nmslib"

        # Followed the bellow link for tuning for ingestion and querying
        # https://opensearch.org/docs/1.1/search-plugins/knn/performance-tuning/#indexing-performance-tuning
        self.client.indices.create(
            index=OPENSEARCH_INDEX,
            body={
                "settings": {
                    "index": index_settings,
                },
                "mappings": {
                    "properties": {
                        "vector": {
                            "type": "knn_vector",
                            "dimension": dataset.config.vector_size,
                            "method": {
                                **{
                                    "name": "hnsw",
                                    "engine": engine,
                                    "space_type": self.DISTANCE_MAPPING[
                                        dataset.config.distance
                                    ],
                                    "parameters": {
                                        "m": 16,
                                        "ef_construction": 100,
                                    },
                                },
                                **collection_params.get("method"),
                            },
                        },
                        # this doesn't work for nmslib, we need see what to do here, may be remove them
                        **field_config
                    }
                },
            },
            params={
                "timeout": 300,
            },
            cluster_manager_timeout="5m",
        )

    def _prepare_fields_config(self, dataset: Dataset):
        return {
            field_name: {
                # The mapping is used only for several types, as some of them
                # overlap with the ones used internally.
                "type": self.INDEX_TYPE_MAPPING.get(field_type, field_type),
                "index": True,
            }
            for field_name, field_type in dataset.config.schema.items()
        }
