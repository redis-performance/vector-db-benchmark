from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from benchmark.dataset import Dataset
from engine.base_client.configure import BaseConfigurator
from engine.base_client.distances import Distance
from engine.clients.qdrant.config import (
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_URL,
    retry_with_exponential_backoff,
)


class QdrantConfigurator(BaseConfigurator):
    DISTANCE_MAPPING = {
        Distance.L2: rest.Distance.EUCLID,
        Distance.COSINE: rest.Distance.COSINE,
        Distance.DOT: rest.Distance.DOT,
    }
    INDEX_TYPE_MAPPING = {
        "int": rest.PayloadSchemaType.INTEGER,
        "keyword": rest.PayloadSchemaType.KEYWORD,
        "text": rest.PayloadSchemaType.TEXT,
        "float": rest.PayloadSchemaType.FLOAT,
        "geo": rest.PayloadSchemaType.GEO,
    }

    def __init__(self, host, collection_params: dict, connection_params: dict):
        super().__init__(host, collection_params, connection_params)
        if QDRANT_URL is None:
            self.client = QdrantClient(
                host=host, api_key=QDRANT_API_KEY, **connection_params
            )
        else:
            self.client = QdrantClient(
                url=QDRANT_URL, api_key=QDRANT_API_KEY, **connection_params
            )

    def clean(self):
        res = self.client.delete_collection(collection_name=QDRANT_COLLECTION_NAME)

    def recreate(self, dataset: Dataset, collection_params):
        retry_with_exponential_backoff(
            self.client.recreate_collection,
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=rest.VectorParams(
                size=dataset.config.vector_size,
                distance=self.DISTANCE_MAPPING.get(dataset.config.distance),
            ),
            **self.collection_params
        )
        self.client.update_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            optimizer_config=rest.OptimizersConfigDiff(
                # indexing_threshold=10000000,
                max_optimization_threads=0,
            ),
        )
        for field_name, field_type in dataset.config.schema.items():
            self.client.create_payload_index(
                collection_name=QDRANT_COLLECTION_NAME,
                field_name=field_name,
                field_schema=self.INDEX_TYPE_MAPPING.get(field_type),
            )
