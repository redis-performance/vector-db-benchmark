from typing import List
import time
from dataset_reader.base_reader import Record
from engine.base_client.upload import BaseUploader

from engine.clients.azure_ai.config import (
    AZUREAI_API_KEY,
    AZUREAI_API_VERSION,
    AZUREAI_SERVICE_NAME,
    AZUREAI_INDEX_NAME,
    add_docs,
    list_indices_statssummary,
)


class AzureAIUploader(BaseUploader):
    client = None
    upload_params = {}

    @classmethod
    def init_client(cls, host, distance, connection_params, upload_params):
        if AZUREAI_API_VERSION is None:
            raise Exception(
                "An api key is required to use Azure AI Search. Specify it via AZUREAI_API_KEY=..."
            )
        cls.api_version = AZUREAI_API_VERSION
        cls.service_endpoint = f"https://{AZUREAI_SERVICE_NAME}.search.windows.net"
        cls.upload_params = upload_params

    @classmethod
    def upload_batch(cls, batch: List[Record]):
        docs = {"value": []}

        for record in batch:
            idx = record.id
            vec = record.vector
            doc = {
                "@search.action": "mergeOrUpload",
                "Id": f"{idx}",
                "VectorField": vec,
            }
            docs["value"].append(doc)

        add_docs(
            cls.service_endpoint,
            cls.api_version,
            AZUREAI_API_KEY,
            AZUREAI_INDEX_NAME,
            docs,
        )

    @classmethod
    def post_upload(cls, _distance, doc_count):
        indexing = True
        sleep_secs = 10
        while indexing is True:
            indexstats = list_indices_statssummary(
                cls.service_endpoint, cls.api_version, AZUREAI_API_KEY
            )
            # {'@odata.context': 'https://vecsim-s2.search.windows.net/$metadata#Collection(Microsoft.Azure.Search.V2023_11_01.IndexStatisticsSummary)', 'value': [{'name': 'idx', 'documentCount': 1183514, 'storageSize': 529452670, 'vectorIndexSize': 183084912}]}
            len_indices = len(indexstats["value"])
            print(f"within clean, detected {len_indices} indices.")
            for index in indexstats["value"]:
                if index["name"] == AZUREAI_INDEX_NAME:
                    print(
                        f"Found existing index with name {AZUREAI_INDEX_NAME}. deleting it..."
                    )
                    print(index)
                    if doc_count is None:
                        print("given doc_count is null, skipping indexing check...")
                        indexing = False
                    else:
                        indexed_docs = index["documentCount"]
                        print(
                            f"checking if indexed docs({indexed_docs}) == doc_count ({doc_count}) of dataset."
                        )
                        if indexed_docs < doc_count:
                            print(
                                f"Indexing still in progress... {indexed_docs} < {doc_count}. Sleeping for {sleep_secs} secs"
                            )
                            time.sleep(sleep_secs)
                        else:
                            indexing = False
                            print("finished indexing...")
                            return {}
        return {}

    def get_memory_usage(cls):
        stats = {}
        indexstats = list_indices_statssummary(
            cls.service_endpoint, cls.api_version, AZUREAI_API_KEY
        )
        # {'@odata.context': 'https://vecsim-s2.search.windows.net/$metadata#Collection(Microsoft.Azure.Search.V2023_11_01.IndexStatisticsSummary)', 'value': [{'name': 'idx', 'documentCount': 1183514, 'storageSize': 529452670, 'vectorIndexSize': 183084912}]}
        len_indices = len(indexstats["value"])
        print(f"within clean, detected {len_indices} indices.")
        for index in indexstats["value"]:
            if index["name"] == AZUREAI_INDEX_NAME:
                print(
                    f"Found existing index with name {AZUREAI_INDEX_NAME}. deleting it..."
                )
                print(index)
                stats = index

        return stats

    @classmethod
    def delete_client(cls):
        if cls.client is not None:
            del cls.client
