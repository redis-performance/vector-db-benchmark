import uuid
from typing import List, Optional
from weaviate import WeaviateClient

from engine.base_client.upload import BaseUploader
from engine.clients.weaviate.config import WEAVIATE_CLASS_NAME, setup_client
from weaviate.classes.data import DataObject

class WeaviateUploader(BaseUploader):
    client: WeaviateClient = None
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
        objects = []
        for pos, vector in enumerate(vectors):
            _id = uuid.UUID(ids[pos])
            _property = {}
            if metadata is not None and len(metadata) >= pos:
                _property = metadata[pos]
            objects.append(
                DataObject(properties=_property, vector=vector, uuid=_id)
            )
        if len(objects) > 0:
            cls.collection.data.insert_many(objects)

    @classmethod
    def delete_client(cls):
        if cls.client is not None:
            cls.client.close()
