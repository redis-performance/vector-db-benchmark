from typing import List, Tuple
from engine.base_client.search import BaseSearcher
from engine.clients.databricks.config import DATABRICKS_TOKEN
from databricks.vector_search.client import VectorSearchClient


class DatabricksSearcher(BaseSearcher):
    search_params = {}
    client = None

    @classmethod
    def init_client(cls, host, distance, connection_params: dict, search_params: dict):
        token = DATABRICKS_TOKEN
        cls.catalog_name = "vector_search_bench"
        cls.schema_name = "vector_search"
        cls.index_name_short = "bench"
        cls.vector_search_endpoint_name = "vector-search-demo-endpoint"
        cls.index_name = f"{cls.catalog_name}.{cls.schema_name}.{cls.index_name_short}"
        cls.client = VectorSearchClient(
            personal_access_token=token,
            disable_notice=True,
        )
        cls.index = cls.client.get_index(index_name=cls.index_name)

    @classmethod
    def search_one(cls, vector, meta_conditions, top) -> List[Tuple[int, float]]:

        # Delta Sync Index with pre-calculated embeddings
        reply = cls.index.similarity_search(
            query_vector=vector,
            columns=["id"],
            num_results=top,
            disable_notice=True,
        )
        results = reply["result"]["data_array"]
        return [(int(result[0]), float(result[1])) for result in results]
