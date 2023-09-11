from typing import List, Tuple

import numpy as np
import redis
from redis.commands.search.query import Query

from engine.base_client.search import BaseSearcher
from engine.clients.redis.config import REDIS_PORT, REDIS_QUERY_TIMEOUT, REDIS_HYBRID_POLICY, REDIS_KEY_PREFIX, REDIS_AUTH, REDIS_USER
from engine.clients.redis.parser import RedisConditionParser


class RedisSearcher(BaseSearcher):
    search_params = {}
    client = None
    parser = RedisConditionParser()

    @classmethod
    def init_client(cls, host, distance, connection_params: dict, search_params: dict):
        cls.client = redis.Redis(host=host, port=REDIS_PORT, db=0, password=REDIS_AUTH, username=REDIS_USER)
        cls.search_params = search_params
        cls.knn_conditions = "EF_RUNTIME $EF"
        if REDIS_HYBRID_POLICY is not None:
            # for HYBRID_POLICY ADHOC_BF we need to remove EF_RUNTIME
            if REDIS_HYBRID_POLICY == "ADHOC_BF":
                cls.knn_conditions = ""
            cls.knn_conditions = f"HYBRID_POLICY {REDIS_HYBRID_POLICY} {cls.knn_conditions}"
    @classmethod
    def search_one(cls, vector, meta_conditions, top) -> List[Tuple[int, float]]:
        conditions = cls.parser.parse(meta_conditions)
        if conditions is None:
            prefilter_condition = "*"
            params = {}
        else:
            prefilter_condition, params = conditions

        q = (
            Query(
                f"{prefilter_condition}=>[KNN $K @vector $vec_param {cls.knn_conditions} AS vector_score]"
            )
            .sort_by("vector_score", asc=False)
            .paging(0, top)
            .return_fields("vector_score")
            .dialect(2)
            .timeout(REDIS_QUERY_TIMEOUT)
        )
        params_dict = {
            "vec_param": np.array(vector).astype(np.float32).tobytes(),
            "K": top,
            "EF": cls.search_params["search_params"]["ef"],
            **params,
        }

        results = cls.client.ft().search(q, query_params=params_dict)

        return [(int(result.id[len(REDIS_KEY_PREFIX):]), float(result.vector_score)) for result in results.docs]
