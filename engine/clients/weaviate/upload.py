import uuid
from typing import List, Optional

from weaviate import Client, AuthApiKey

from engine.base_client.upload import BaseUploader
from engine.clients.weaviate.config import WEAVIATE_CLASS_NAME, WEAVIATE_PORT, WEAVIATE_API_KEY


class WeaviateUploader(BaseUploader):
    client = None
    upload_params = {}

    @classmethod
    def init_client(cls, host, distance, connection_params, upload_params):
        if host.startswith("http"):
            url = ""
        else:
            url = "http://"
        url += f"{host}:{connection_params.pop('port', WEAVIATE_PORT)}"
        auth_client_secret = None
        if WEAVIATE_API_KEY is not None:
            auth_client_secret = AuthApiKey(WEAVIATE_API_KEY)
        cls.client = Client(url, auth_client_secret, **connection_params)

        cls.upload_params = upload_params
        cls.connection_params = connection_params

    @staticmethod
    def _update_geo_data(data_object):
        keys = data_object.keys()
        for key in keys:
            if isinstance(data_object[key], dict):
                if lat := data_object[key].pop("lat", None):
                    data_object[key]["latitude"] = lat
                if lon := data_object[key].pop("lon", None):
                    data_object[key]["longitude"] = lon

        return data_object

    @classmethod
    def upload_batch(
        cls, ids: List[int], vectors: List[list], metadata: List[Optional[dict]]
    ):
        # Weaviate introduced the batch_size, so it can handle built-in client's
        # multi-threading. That should make the upload faster.
        cls.client.batch.configure(
            batch_size=100,
            timeout_retries=3,
        )

        with cls.client.batch as batch:
            for id_, vector, data_object in zip(ids, vectors, metadata):
                data_object = cls._update_geo_data(data_object or {})
                batch.add_data_object(
                    data_object=data_object,
                    class_name=WEAVIATE_CLASS_NAME,
                    uuid=uuid.UUID(int=id_).hex,
                    vector=vector,
                )

            batch.create_objects()
