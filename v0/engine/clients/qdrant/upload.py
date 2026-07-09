import json
import os
import time
from typing import List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import Batch, CollectionStatus, OptimizersConfigDiff

from engine.base_client.upload import BaseUploader
from engine.clients.qdrant.config import (
    QDRANT_ACCOUNT_ID,
    QDRANT_API_KEY,
    QDRANT_AUTH_TOKEN,
    QDRANT_CLUSTER_ID,
    QDRANT_COLLECTION_NAME,
    QDRANT_MAX_OPTIMIZATION_THREADS,
    QDRANT_URL,
    get_collection_info,
    get_qdrant_cloud_usage,
)


class QdrantUploader(BaseUploader):
    client = None
    upload_params = {}

    @classmethod
    def init_client(cls, host, distance, connection_params, upload_params):
        os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "true"
        os.environ["GRPC_POLL_STRATEGY"] = "epoll,poll"
        if QDRANT_URL is None:
            cls.client = QdrantClient(
                host=host, api_key=QDRANT_API_KEY, prefer_grpc=True, **connection_params
            )
        else:
            cls.client = QdrantClient(
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                prefer_grpc=True,
                **connection_params,
            )
        cls.upload_params = upload_params

    @classmethod
    def upload_batch(
        cls, ids: List[int], vectors: List[list], metadata: Optional[List[dict]]
    ):
        cls.client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=Batch.model_construct(
                ids=ids,
                vectors=vectors,
                payloads=[payload or {} for payload in metadata],
            ),
            wait=False,
        )

    @classmethod
    def post_upload(cls, _distance):
        max_optimization_threads = QDRANT_MAX_OPTIMIZATION_THREADS
        if max_optimization_threads is not None:
            max_optimization_threads = int(max_optimization_threads)
        cls.client.update_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            optimizer_config=OptimizersConfigDiff(
                # indexing_threshold=10_000,
                max_optimization_threads=max_optimization_threads,
            ),
        )

        cls.wait_collection_green()
        return {}

    @classmethod
    def wait_collection_green(cls):
        wait_time = 5.0
        total = 0
        while True:
            time.sleep(wait_time)
            total += wait_time
            collection_info = cls.client.get_collection(QDRANT_COLLECTION_NAME)
            if collection_info.status != CollectionStatus.GREEN:
                continue
            time.sleep(wait_time)
            collection_info = cls.client.get_collection(QDRANT_COLLECTION_NAME)
            if collection_info.status == CollectionStatus.GREEN:
                break
        return total

    @classmethod
    def delete_client(cls):
        if cls.client is not None:
            del cls.client

    def get_memory_usage(cls):
        collection_info = get_collection_info(
            QDRANT_URL, QDRANT_COLLECTION_NAME, QDRANT_API_KEY
        )
        used_memory = {}
        # Extract memory usage information
        if (
            QDRANT_ACCOUNT_ID is not None
            and QDRANT_CLUSTER_ID is not None
            and QDRANT_AUTH_TOKEN is not None
        ):
            print(f"Tring to fetch Qdrant cloud usage from Cluster {QDRANT_CLUSTER_ID}")
            used_memory = get_qdrant_cloud_usage(
                QDRANT_ACCOUNT_ID, QDRANT_CLUSTER_ID, QDRANT_AUTH_TOKEN
            )

        return {
            "used_memory": used_memory,
            "collection_info": collection_info,
        }
