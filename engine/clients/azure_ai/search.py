from typing import List, Tuple


from dataset_reader.base_reader import Query
from engine.base_client.search import BaseSearcher
from engine.clients.azure_ai.config import (
    AZUREAI_API_VERSION,
    AZUREAI_SERVICE_NAME,
    AZUREAI_API_KEY,
    AZUREAI_INDEX_NAME,
    search_azure,
)


class AzureAISearcher(BaseSearcher):
    search_params = {}

    @classmethod
    def init_client(cls, host, distance, connection_params: dict, search_params: dict):
        if AZUREAI_API_VERSION is None:
            raise Exception(
                "An api key is required to use Azure AI Search. Specify it via AZUREAI_API_KEY=..."
            )
        cls.search_params = search_params
        cls.api_version = AZUREAI_API_VERSION
        cls.service_endpoint = f"https://{AZUREAI_SERVICE_NAME}.search.windows.net"

    @classmethod
    def search_one(cls, query: Query, top: int) -> List[Tuple[int, float]]:
        query = {
            "count": True,
            "select": "Id",
            "vectorQueries": [
                {
                    "vector": query.vector,
                    "k": top,
                    "fields": "VectorField",
                    "kind": "vector",
                    "exhaustive": True,
                }
            ],
        }
        reply = search_azure(
            cls.service_endpoint,
            AZUREAI_INDEX_NAME,
            cls.api_version,
            AZUREAI_API_KEY,
            query,
        )
        result = [
            (int(value["Id"]), float(value["@search.score"])) for value in reply["value"]
        ]

        return result
