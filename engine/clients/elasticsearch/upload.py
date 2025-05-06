import multiprocessing as mp
import uuid
from typing import List

import elastic_transport
from elasticsearch import Elasticsearch, ApiError

from dataset_reader.base_reader import Record
from engine.base_client.upload import BaseUploader
from engine.clients.elasticsearch.config import (
    ELASTIC_INDEX,
    get_es_client,
    _wait_for_es_status,
)


class ClosableElastic(Elasticsearch):
    def __del__(self):
        self.close()


class ElasticUploader(BaseUploader):
    client: Elasticsearch = None
    upload_params = {}

    @classmethod
    def get_mp_start_method(cls):
        return "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"

    @classmethod
    def init_client(cls, host, distance, connection_params, upload_params):
        cls.client = get_es_client(host, connection_params)
        cls.upload_params = upload_params

    @classmethod
    def upload_batch(cls, batch: List[Record]):
        operations = []
        for record in batch:
            vector_id = uuid.UUID(int=record.id).hex
            operations.append({"index": {"_id": vector_id}})
            operations.append({"vector": record.vector, **(record.metadata or {})})

        cls.client.bulk(
            index=ELASTIC_INDEX,
            operations=operations,
        )

    @classmethod
    def post_upload(cls, _distance):
        print("forcing the merge into 1 segment...")
        tries = 30
        for i in range(tries + 1):
            try:
                cls.client.indices.forcemerge(
                    index=ELASTIC_INDEX, wait_for_completion=True, max_num_segments=1
                )
            except (elastic_transport.TlsError, ApiError) as e:
                if i < tries:  # i is zero indexed
                    print(
                        "Received the following error during retry {}/{} while waiting for ES index to be ready... {}".format(
                            i, tries, e.__str__()
                        )
                    )
                    continue
                else:
                    raise
            _wait_for_es_status(cls.client)
            break
        return {}
