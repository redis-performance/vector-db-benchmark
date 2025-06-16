import random
from typing import List, Tuple

import numpy as np
from redis import Redis, RedisCluster


from engine.base_client.search import BaseSearcher
from engine.clients.vectorsets.config import (
    REDIS_AUTH,
    REDIS_CLUSTER,
    REDIS_PORT,
    REDIS_QUERY_TIMEOUT,
    REDIS_USER,
)
from engine.clients.redis.parser import RedisConditionParser


class RedisVsetSearcher(BaseSearcher):
    search_params = {}
    client = None
    parser = RedisConditionParser()

    @classmethod
    def init_client(cls, host, distance, connection_params: dict, search_params: dict):
        redis_constructor = RedisCluster if REDIS_CLUSTER else Redis
        cls.client = redis_constructor(
            host=host, port=REDIS_PORT, password=REDIS_AUTH, username=REDIS_USER
        )
        cls.search_params = search_params
        cls._is_cluster = True if REDIS_CLUSTER else False
        # In the case of CLUSTER API enabled we randomly select the starting primary shard
        # when doing the client initialization to evenly distribute the load among the cluster
        cls.conns = [cls.client]
        if cls._is_cluster:
            cls.conns = [
                cls.client.get_redis_connection(node)
                for node in cls.client.get_primaries()
            ]
        cls._ft = cls.conns[random.randint(0, len(cls.conns)) - 1].ft()

    @classmethod
    def search_one(cls, vector, meta_conditions, top) -> List[Tuple[int, float]]:
        ef = cls.search_params["search_params"]["ef"]
        response = cls.client.execute_command("VSIM", "idx", "FP32", np.array(vector).astype(np.float32).tobytes(), "WITHSCORES", "COUNT", top, "EF", ef)
        # decode responses
        # every even cell is id, every odd is the score
        # scores needs to be 1 - scores since on vector sets 1 is identical, 0 is opposite vector
        ids = [int(response[i]) for i in range(0, len(response), 2)]
        scores = [1 - float(response[i]) for i in range(1, len(response), 2)]
        # we need to return a list of tuples
        # where the first element is the id and the second is the score
        return list(zip(ids, scores))        
