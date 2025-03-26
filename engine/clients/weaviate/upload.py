import uuid
from typing import List

from weaviate.classes.data import DataObject
from weaviate import WeaviateClient
from dataset_reader.base_reader import Record
from engine.base_client.upload import BaseUploader
from engine.clients.weaviate.config import WEAVIATE_CLASS_NAME, setup_client


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
    def upload_batch(cls, batch: List[Record]):
        objects = []
        for record in batch:
            _id = uuid.UUID(int=record.id)
            _property = record.metadata or {}
            objects.append(
                DataObject(properties=_property, vector=record.vector, uuid=_id)
            )
        if len(objects) > 0:
            cls.collection.data.insert_many(objects)

    @classmethod
    def delete_client(cls):
        if cls.client is not None:
            cls.client.close()
