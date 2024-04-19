import time
from typing import List, Optional

import numpy as np
from redis import Redis, RedisCluster
from engine.base_client.upload import BaseUploader
from engine.clients.redis.config import (
    REDIS_PORT,
    REDIS_AUTH,
    REDIS_USER,
    REDIS_CLUSTER,
    REDIS_ALGORITHM,
)
from engine.clients.redis.helper import convert_to_redis_coords


class RedisUploader(BaseUploader):
    client = None
    client_decode = None
    upload_params = {}

    @classmethod
    def init_client(cls, host, distance, connection_params, upload_params):
        redis_constructor = RedisCluster if REDIS_CLUSTER else Redis
        cls.client = redis_constructor(
            host=host, port=REDIS_PORT, password=REDIS_AUTH, username=REDIS_USER
        )
        cls.client_decode = redis_constructor(
            host=host,
            port=REDIS_PORT,
            password=REDIS_AUTH,
            username=REDIS_USER,
            decode_responses=True,
        )
        cls.upload_params = upload_params

    @classmethod
    def upload_batch(
        cls, ids: List[int], vectors: List[list], metadata: Optional[List[dict]]
    ):
        p = cls.client.pipeline(transaction=False)
        for i in range(len(ids)):
            idx = ids[i]
            vec = vectors[i]
            meta = metadata[i] if metadata else {}
            geopoints = {}
            payload = {}
            if meta is not None:
                for k, v in meta.items():
                    # This is a patch for arxiv-titles dataset where we have a list of "labels", and
                    # we want to index all of them under the same TAG field (whose separator is ';').
                    if k == "labels":
                        payload[k] = ";".join(v)
                    if (
                        v is not None
                        and not isinstance(v, dict)
                        and not isinstance(v, list)
                    ):
                        payload[k] = v
                # Redis treats geopoints differently and requires putting them as
                # a comma-separated string with lat and lon coordinates
                geopoints = {
                    k: ",".join(map(str, convert_to_redis_coords(v["lon"], v["lat"])))
                    for k, v in meta.items()
                    if isinstance(v, dict)
                }
            cls.client.hset(
                str(idx),
                mapping={
                    "vector": np.array(vec).astype(np.float32).tobytes(),
                    **payload,
                    **geopoints,
                },
            )
        p.execute()

    @classmethod
    def post_upload(cls, _distance):
        if REDIS_ALGORITHM != "HNSW" and REDIS_ALGORITHM != "FLAT":
            print(f"TODO: FIXME!! Avoiding calling ft.info for {REDIS_ALGORITHM}...")
            return {}
        index_info = cls.client.ft().info()
        # redisearch / memorystore for redis
        if "percent_index" in index_info:
            percent_index = float(index_info["percent_index"])
            while percent_index < 1.0:
                print(
                    "waiting for index to be fully processed. current percent index: {}".format(
                        percent_index * 100.0
                    )
                )
                time.sleep(1)
                percent_index = float(cls.client.ft().info()["percent_index"])
        # memorydb
        if "current_lag" in index_info:
            current_lag = float(index_info["current_lag"])
            while current_lag > 0:
                print(
                    "waiting for index to be fully processed. current current_lag: {}".format(
                        current_lag
                    )
                )
                time.sleep(1)
                current_lag = int(cls.client.ft().info()["current_lag"])
        return {}

    def get_memory_usage(cls):
        used_memory = cls.client_decode.info("memory")["used_memory"]
        index_info = cls.client_decode.ft().info()
        return {"used_memory": used_memory, "index_info": index_info}
