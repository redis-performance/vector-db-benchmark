import redis
from redis.commands.search.field import GeoField, NumericField, TextField, VectorField, TagField

from benchmark.dataset import Dataset
from engine.base_client.configure import BaseConfigurator
from engine.base_client.distances import Distance
from engine.clients.redis.config import REDIS_PORT, REDIS_AUTH, REDIS_USER, DISABLE_CLEAN, REDIS_KEY_PREFIX


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
        if REDIS_KEY_PREFIX != "":
            print(f"\tUsing a key prefix for this experiment: {REDIS_KEY_PREFIX}")

        self.client = redis.Redis(host=host, port=REDIS_PORT, password=REDIS_AUTH, username=REDIS_USER)

    def clean(self):
        if DISABLE_CLEAN is False:
            index = self.client.ft()
            try:
                index.dropindex(delete_documents=True)
            except redis.ResponseError as e:
                if "Unknown Index name" not in e.__str__():
                    print(e)
        else:
            print(f"\tSkipping clean stage given DISABLE_CLEAN=1")


    def recreate(self, dataset: Dataset, collection_params):
        self.clean()
        search_namespace = self.client.ft()
        payload_fields = [
            self.FIELD_MAPPING[field_type](
                name=field_name,
                sortable=True,
            )
            for field_name, field_type in dataset.config.schema.items() if field_type != 'keyword'
        ]
        payload_fields += [
            TagField(
                name=field_name,
                separator=';',
                sortable=True,
            )
            for field_name, field_type in dataset.config.schema.items() if field_type == 'keyword'
        ]
        try:
            search_namespace.create_index(
                fields=[
                    VectorField(
                        name="vector",
                        algorithm="HNSW",
                        attributes={
                            "TYPE": "FLOAT32",
                            "DIM": dataset.config.vector_size,
                            "DISTANCE_METRIC": self.DISTANCE_MAPPING[
                                dataset.config.distance
                            ],
                            **self.collection_params.get("hnsw_config", {}),
                        },
                    )
                ]
                + payload_fields
            )
        except redis.ResponseError as e:
            if "Index already exists" not in e.__str__():
                raise e
            elif DISABLE_CLEAN is True:
                print("There as an error when creating the index but you've specified DISABLE_CLEAN=1 so we're ignoring it")
            else:
                raise e


if __name__ == "__main__":
    pass
