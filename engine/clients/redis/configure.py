import redis
import time
from redis import Redis, RedisCluster
from redis.commands.search.field import (
    GeoField,
    NumericField,
    TextField,
    VectorField,
    TagField,
)

from benchmark.dataset import Dataset
from engine.base_client.configure import BaseConfigurator
from engine.base_client.distances import Distance
from engine.clients.redis.config import (
    REDIS_PORT,
    REDIS_AUTH,
    REDIS_USER,
    REDIS_CLUSTER,
    REDIS_KEEP_DOCUMENTS,
)


class RedisConfigurator(BaseConfigurator):
    DISTANCE_MAPPING = {
        Distance.L2: "L2",
        Distance.COSINE: "COSINE",
        Distance.DOT: "IP",
    }
    FIELD_MAPPING = {
        "int": NumericField,
        "keyword": TagField,
        "text": TextField,
        "float": NumericField,
        "geo": GeoField,
    }

    def __init__(self, host, collection_params: dict, connection_params: dict):
        super().__init__(host, collection_params, connection_params)
        redis_constructor = RedisCluster if REDIS_CLUSTER else Redis
        self._is_cluster = True if REDIS_CLUSTER else False
        self.client = redis_constructor(
            host=host, port=REDIS_PORT, password=REDIS_AUTH, username=REDIS_USER
        )

    def clean(self):
        conns = [self.client]
        if self._is_cluster:
            conns = [
                self.client.get_redis_connection(node)
                for node in self.client.get_primaries()
            ]
        for conn in conns:
            index = conn.ft()
            try:
                index.dropindex(delete_documents=(not REDIS_KEEP_DOCUMENTS))
            except redis.ResponseError as e:
                str_err = e.__str__()
                if (
                    "Unknown Index name" not in str_err
                    and "Index does not exist" not in str_err
                ):
                    # google memorystore does not support the DD argument.
                    # in that case we can flushall
                    if "wrong number of arguments for FT.DROPINDEX command" in str_err:
                        print(
                            "Given the FT.DROPINDEX command failed, we're flushing the entire DB..."
                        )
                        if REDIS_KEEP_DOCUMENTS is False:
                            conn.flushall()
                    else:
                        raise e

    def recreate(self, dataset: Dataset, collection_params):
        self.clean()

        payload_fields = [
            self.FIELD_MAPPING[field_type](
                name=field_name,
                sortable=True,
            )
            for field_name, field_type in dataset.config.schema.items()
            if field_type != "keyword"
        ]
        payload_fields += [
            TagField(
                name=field_name,
                separator=";",
                sortable=True,
            )
            for field_name, field_type in dataset.config.schema.items()
            if field_type == "keyword"
        ]
        algorithm_config = {}
        # by default we use hnsw
        algo = collection_params.get("algorithm", "hnsw")
        data_type = collection_params.get("data_type", "float32")
        algorithm_config = collection_params.get(f"{algo}_config", {})
        print(f"Using algorithm {algo} with config {algorithm_config}")
        index_fields = [
            VectorField(
                name="vector",
                algorithm=algo,
                attributes={
                    "TYPE": data_type,
                    "DIM": dataset.config.vector_size,
                    "DISTANCE_METRIC": self.DISTANCE_MAPPING[dataset.config.distance],
                    **algorithm_config,
                },
            )
        ] + payload_fields

        conns = [self.client]
        if self._is_cluster:
            conns = [
                self.client.get_redis_connection(node)
                for node in self.client.get_primaries()
            ]
        for conn in conns:
            search_namespace = conn.ft()
            try:
                search_namespace.create_index(fields=index_fields)
            except redis.ResponseError as e:
                if "Index already exists" not in e.__str__():
                    raise e


if __name__ == "__main__":
    pass
