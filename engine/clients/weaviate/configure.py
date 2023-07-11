from weaviate import Client, AuthApiKey

from benchmark.dataset import Dataset
from engine.base_client.configure import BaseConfigurator
from engine.base_client.distances import Distance
from engine.clients.weaviate.config import WEAVIATE_CLASS_NAME, WEAVIATE_PORT, WEAVIATE_API_KEY


class WeaviateConfigurator(BaseConfigurator):
    DISTANCE_MAPPING = {
        Distance.L2: "l2-squared",
        Distance.COSINE: "cosine",
        Distance.DOT: "dot",
    }
    FIELD_TYPE_MAPPING = {
        "int": "int",
        "keyword": "string",
        "text": "string",
        "float": "number",
        "geo": "geoCoordinates",
    }

    def __init__(self, host, collection_params: dict, connection_params: dict):
        super().__init__(host, collection_params, connection_params)
        if host.startswith("http"):
            url = ""
        else:
            url = "http://"
        url += f"{host}:{connection_params.pop('port', WEAVIATE_PORT)}"
        auth_client_secret = None
        if WEAVIATE_API_KEY is not None:
            auth_client_secret = AuthApiKey(WEAVIATE_API_KEY)

        self.client = Client(url, auth_client_secret, **connection_params)

    def clean(self):
        classes = self.client.schema.get()
        for cl in classes["classes"]:
            if cl["class"] == WEAVIATE_CLASS_NAME:
                self.client.schema.delete_class(WEAVIATE_CLASS_NAME)

    def recreate(self, dataset: Dataset, collection_params):
        self.client.schema.create_class(
            {
                "class": WEAVIATE_CLASS_NAME,
                "vectorizer": "none",
                "properties": [
                    {
                        "name": field_name,
                        "dataType": [
                            self.FIELD_TYPE_MAPPING[field_type],
                        ],
                        "indexInverted": True,
                    }
                    for field_name, field_type in dataset.config.schema.items()
                ],
                "vectorIndexConfig": {
                    **{
                        "vectorCacheMaxObjects": 1000000000,
                        "distance": self.DISTANCE_MAPPING.get(dataset.config.distance),
                    },
                    **collection_params["vectorIndexConfig"],
                },
            }
        )
