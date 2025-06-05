import time
import requests
from benchmark.dataset import Dataset
from engine.base_client.configure import BaseConfigurator
from engine.base_client.distances import Distance
from engine.clients.databricks.config import DATABRICKS_TOKEN
from databricks.vector_search.client import VectorSearchClient


def wait_until_index_deleted(client, endpoint_name, index_name, timeout=120):
    print(f"Waiting for index {index_name} to be fully deleted...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            client.get_index(endpoint_name=endpoint_name, index_name=index_name)
            print("Still deleting...")
        except Exception as e:
            # Check if it's a 404 Not Found
            if (
                isinstance(e.__cause__, requests.HTTPError)
                and e.__cause__.response.status_code == 404
            ):
                print("Index fully deleted.")
                return
            # elif "RESOURCE_DOES_NOT_EXIST" in e.__str__():
            #     continue
            else:
                raise e
        time.sleep(5)
    raise TimeoutError(f"Index {index_name} was not deleted within {timeout} seconds.")


class DatabricksConfigurator(BaseConfigurator):
    def __init__(self, host, collection_params: dict, connection_params: dict):
        super().__init__(host, collection_params, connection_params)
        token = DATABRICKS_TOKEN
        self.catalog_name = "vector_search_bench"
        self.schema_name = "vector_search"
        self.index_name_short = "bench"
        self.vector_search_endpoint_name = "vector-search-demo-endpoint"
        self.index_name = (
            f"{self.catalog_name}.{self.schema_name}.{self.index_name_short}"
        )

        self.client = VectorSearchClient(
            personal_access_token=token,
            disable_notice=True,
        )

    def clean(self):
        # List existing indexes on the endpoint
        print(f"Listing existing vector indices for clean()")
        existing_indexes = self.client.list_indexes(
            name=self.vector_search_endpoint_name
        )
        if "vector_indexes" in existing_indexes:
            existing_indexes = existing_indexes["vector_indexes"]
        else:
            existing_indexes = []
        print(f"Existing vector indices: {existing_indexes}")
        # Check if the index already exists
        if any(
            existing_index["name"] == self.index_name
            for existing_index in existing_indexes
        ):
            print(f"Index {self.index_name} already exists — skipping creation.")
            index = self.client.get_index(index_name=self.index_name)
            self.client.delete_index(
                endpoint_name=self.vector_search_endpoint_name,
                index_name=self.index_name,
            )
            wait_until_index_deleted(
                self.client,
                endpoint_name=self.vector_search_endpoint_name,
                index_name=self.index_name,
            )

    def recreate(self, dataset: Dataset, collection_params):
        if dataset.config.distance != Distance.L2:
            raise Exception(
                "Mosaic only supports L2 distance. Check https://learn.microsoft.com/en-us/azure/databricks/generative-ai/vector-search#how-does-mosaic-ai-vector-search-work"
            )
        # self.clean()

        index = self.client.create_direct_access_index(
            endpoint_name=self.vector_search_endpoint_name,
            index_name=self.index_name,
            primary_key="id",
            embedding_dimension=dataset.config.vector_size,
            embedding_vector_column="vector",
            schema={
                "id": "int",
                "vector": "array<float>",
            },
        )
        while not index.describe().get("status")["ready"]:
            print("Waiting for index to be ready...")
            time.sleep(5)
        print("Index is ready!")
        print(index.describe())


if __name__ == "__main__":
    pass
