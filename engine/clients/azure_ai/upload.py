from typing import List

from dataset_reader.base_reader import Record
from engine.base_client.upload import BaseUploader

from engine.clients.azure_ai.config import (
    AZUREAI_API_KEY,
    AZUREAI_API_VERSION,
    AZUREAI_SERVICE_NAME,
    AZUREAI_INDEX_NAME,
    add_docs,
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
    def post_upload(cls, _distance):
        return {}

    @classmethod
    def delete_client(cls):
        if cls.client is not None:
            del cls.client
