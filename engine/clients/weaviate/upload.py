import uuid
from typing import List, Optional
from weaviate import Client

from engine.base_client.upload import BaseUploader
from engine.clients.weaviate.config import WEAVIATE_CLASS_NAME, setup_client


class WeaviateUploader(BaseUploader):
    client: Client = None
    upload_params = {}
    collection = None

    @classmethod
    def init_client(cls, host, distance, connection_params, upload_params):
        cls.client = setup_client(connection_params, host)
        cls.upload_params = upload_params
        cls.connection_params = connection_params
        cls.collection = cls.client.collections.get(
            WEAVIATE_CLASS_NAME, skip_argument_validation=True
        )

    @classmethod
    def upload_batch(
        cls, ids: List[int], vectors: List[list], metadata: List[Optional[dict]]
    ):
        # Weaviate introduced the batch_size, so it can handle built-in client's
        # multi-threading. That should make the upload faster.
        cls.client.batch.configure(
            batch_size=len(vectors),
            timeout_retries=5,
        )

    @classmethod
    def delete_client(cls):
        if cls.client is not None:
            cls.client.close()
