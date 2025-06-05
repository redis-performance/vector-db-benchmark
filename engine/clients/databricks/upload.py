import time
import random
import requests
from typing import List, Optional
from engine.base_client.upload import BaseUploader
from engine.clients.databricks.config import DATABRICKS_TOKEN
from databricks.vector_search.client import VectorSearchClient


class DatabricksUploader(BaseUploader):
    client = None
    host = None
    client_decode = None
    upload_params = {}

    @classmethod
    def init_client(cls, host, distance, connection_params, upload_params):
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
    def upload_batch(
        cls, ids: List[int], vectors: List[list], metadata: Optional[List[dict]]
    ):
        batch = []
        for i in range(len(ids)):
            idx = ids[i]
            vec = vectors[i]
            batch.append({"id": idx, "vector": vec})

        max_retries = 10
        delay = 2  # seconds
        for attempt in range(1, max_retries + 1):
            try:
                cls.index.upsert(batch)
                return  # success
            except Exception as e:
                message = str(e)
                if (
                    "TEMPORARILY_UNAVAILABLE" in message
                    or "status_code 503" in message
                    or isinstance(e.__cause__, requests.HTTPError)
                    and e.__cause__.response.status_code == 503
                ):
                    print(
                        f"[Retry {attempt}] Upsert failed due to service unavailability. Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= random.uniform(1.5, 2.0)  # exponential backoff
                else:
                    raise  # non-retryable error
        raise RuntimeError(
            "Upsert failed after multiple retries due to temporary service issues."
        )

    @classmethod
    def post_upload(cls, _distance):
        while not cls.index.describe().get("status")["ready"]:
            print("Waiting for index to be ready...")
            time.sleep(5)
        print("Index is ready!")
        print(cls.index.describe())
        return {}

    def get_memory_usage(cls):
        return {}
