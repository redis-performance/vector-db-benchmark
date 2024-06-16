from benchmark.dataset import Dataset
from engine.base_client.configure import BaseConfigurator
from engine.base_client.distances import Distance
from engine.clients.azure_ai.config import (
    AZUREAI_API_KEY,
    AZUREAI_API_VERSION,
    AZUREAI_SERVICE_NAME,
    AZUREAI_INDEX_NAME,
    delete_index,
    create_index,
)


class AzureAIConfigurator(BaseConfigurator):
    def __init__(self, host, collection_params: dict, connection_params: dict):
        super().__init__(host, collection_params, connection_params)
        if AZUREAI_API_VERSION is None:
            raise Exception(
                "An api key is required to use Azure AI Search. Specify it via AZUREAI_API_KEY=..."
            )
        self.api_version = AZUREAI_API_VERSION
        self.service_endpoint = f"https://{AZUREAI_SERVICE_NAME}.search.windows.net"

    def clean(self):
        delete_index(
            self.service_endpoint, self.api_version, AZUREAI_INDEX_NAME, AZUREAI_API_KEY
        )

    def recreate(self, dataset: Dataset, collection_params):
        if dataset.config.type == "sparse":
            raise Exception("Sparse vector not implemented.")
        vector_size = dataset.config.vector_size
        distance = dataset.config.distance
        hnsw_config = self.collection_params.get(
            "hnsw_config", {"m": 16, "efConstruction": 64}
        )
        m = hnsw_config["m"]
        efConstruction = hnsw_config["efConstruction"]
        # Index definition
        index_definition = {
            "name": AZUREAI_INDEX_NAME,
            "fields": [
                {
                    "name": "Id",
                    "type": "Edm.String",
                    "key": True,
                    "searchable": False,
                    "filterable": True,
                    "retrievable": True,
                    "sortable": False,
                    "facetable": False,
                },
                {
                    "name": "VectorField",
                    "type": "Collection(Edm.Single)",
                    "searchable": True,
                    "retrievable": True,
                    "dimensions": vector_size,
                    "vectorSearchProfile": "simple-vector-profile",
                },
            ],
            "vectorSearch": {
                "algorithms": [
                    {
                        "name": "simple-hnsw-config",
                        "kind": "hnsw",
                        "hnswParameters": {
                            "m": m,
                            "efConstruction": efConstruction,
                            "metric": distance,
                        },
                    }
                ],
                "profiles": [
                    {"name": "simple-vector-profile", "algorithm": "simple-hnsw-config"}
                ],
            },
        }
        create_index(
            self.service_endpoint, self.api_version, AZUREAI_API_KEY, index_definition
        )
