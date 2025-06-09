from typing import List, Optional

import numpy as np
from redis import Redis, RedisCluster

from engine.base_client.upload import BaseUploader
from engine.clients.vectorsets.config import (
    REDIS_AUTH,
    REDIS_CLUSTER,
    REDIS_PORT,
    REDIS_USER,
)
from engine.clients.redis.helper import convert_to_redis_coords


class RedisVsetUploader(BaseUploader):
    client = None
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
        cls._is_cluster = True if REDIS_CLUSTER else False

    @classmethod
    def upload_batch(
        cls, ids: List[int], vectors: List[list], metadata: Optional[List[dict]]
    ):
        upload_params = cls.upload_params
        hnsw_params = upload_params.get("hnsw_config")
        M = hnsw_params.get("M", 16)
        efc = hnsw_params.get("EF_CONSTRUCTION", 200)
        quant = hnsw_params.get("quant", "NOQUANT")
        
        p = cls.client.pipeline(transaction=False)
        for i in range(len(ids)):
            idx = ids[i]
            vec = vectors[i]
            vec = np.array(vec).astype(np.float32).tobytes()
            p.execute_command("VADD", "idx", "FP32", vec, idx, quant, "M", M, "EF", efc, "CAS")
        p.execute()

    @classmethod
    def post_upload(cls, _distance):
        return {}

    def get_memory_usage(cls):
        used_memory = []
        conns = [cls.client_decode]
        if cls._is_cluster:
            conns = [
                cls.client_decode.get_redis_connection(node)
                for node in cls.client_decode.get_primaries()
            ]
        for conn in conns:
            used_memory_shard = conn.info("memory")["used_memory"]
            used_memory.append(used_memory_shard)

        return {"used_memory": sum(used_memory),
                "shards": len(used_memory)}