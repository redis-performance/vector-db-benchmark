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
        cls.upload_params = upload_params

    @classmethod
    def upload_batch(
        cls, ids: List[int], vectors: List[list], metadata: Optional[List[dict]]
    ):
        upload_params = cls.upload_params
        hnsw_params = upload_params.get("hnsw_config")
        M = hnsw_params.get("M", 16)
        efc = hnsw_params.get("EF_CONSTRUCTION", 200)
        quant = hnsw_params.get("quant")
        
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
