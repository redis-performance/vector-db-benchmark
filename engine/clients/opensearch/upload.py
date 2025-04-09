import multiprocessing as mp
import uuid
from typing import List, Optional

from opensearchpy import OpenSearch

from engine.base_client.upload import BaseUploader
from engine.clients.opensearch.config import OPENSEARCH_INDEX, get_opensearch_client


class ClosableOpenSearch(OpenSearch):
    def __del__(self):
        self.close()


class OpenSearchUploader(BaseUploader):
    client: OpenSearch = None
    upload_params = {}

    @classmethod
    def get_mp_start_method(cls):
        return "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"

    @classmethod
    def init_client(cls, host, distance, connection_params, upload_params):
        cls.client = get_opensearch_client(host, connection_params)
        cls.upload_params = upload_params

    @classmethod
    def upload_batch(
        cls, ids: List[int], vectors: List[list], metadata: Optional[List[dict]]
    ):
        if metadata is None:
            metadata = [{}] * len(vectors)
        operations = []
        for idx, vector, payload in zip(ids, vectors, metadata):
            vector_id = uuid.UUID(int=idx).hex
            operations.append({"index": {"_id": vector_id}})
            if payload:
                operations.append({"vector": vector, **payload})
            else:
                operations.append({"vector": vector})

        cls.client.bulk(
            index=OPENSEARCH_INDEX,
            body=operations,
            params={
                "timeout": 300,
            },
        )

    @classmethod
    def post_upload(cls, _distance):
        cls.client.indices.forcemerge(
            index=OPENSEARCH_INDEX,
            params={
                "timeout": 300,
            },
        )
        return {}
