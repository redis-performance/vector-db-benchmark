import random
from typing import List, Tuple
from ml_dtypes import bfloat16
import numpy as np
from redis import Redis, RedisCluster
from redis.commands.search.query import Query
from engine.base_client.search import BaseSearcher
from engine.clients.redis.config import (
    REDIS_PORT,
    REDIS_QUERY_TIMEOUT,
    REDIS_AUTH,
    REDIS_USER,
    REDIS_CLUSTER,
    REDIS_HYBRID_POLICY,
)

from engine.clients.redis.parser import RedisConditionParser


class RedisSearcher(BaseSearcher):
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
        cls.knn_conditions = ""
        cls.algorithm = cls.search_params.get("algorithm", "hnsw").upper()
        cls.hybrid_policy = REDIS_HYBRID_POLICY

        if cls.algorithm == "HNSW":
            # 'EF_RUNTIME' is irrelevant for 'ADHOC_BF' policy
            if cls.hybrid_policy != "ADHOC_BF":
                cls.knn_conditions = "EF_RUNTIME $EF"

        cls.data_type = "FLOAT32"
        if "search_params" in cls.search_params:
            cls.data_type = (
                cls.search_params["search_params"].get("data_type", "FLOAT32").upper()
            )
        cls.np_data_type = np.float32
        if cls.data_type == "FLOAT64":
            cls.np_data_type = np.float64
        if cls.data_type == "FLOAT16":
            cls.np_data_type = np.float16
        if cls.data_type == "BFLOAT16":
            cls.np_data_type = bfloat16
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
        conditions = cls.parser.parse(meta_conditions)
        hybrid_policy = ""
        if cls.hybrid_policy != "":
            hybrid_policy = f"=>{{$HYBRID_POLICY: {{{cls.hybrid_policy}}}}}"
        if conditions is None:
            prefilter_condition = "*"
            params = {}
        else:
            prefilter_condition, params = conditions

        q = (
            Query(
                f"{prefilter_condition}=>[KNN $K @vector $vec_param {cls.knn_conditions} AS vector_score]{hybrid_policy}"
            )
            .sort_by("vector_score", asc=True)
            .paging(0, top)
            .return_fields("vector_score")
            # performance is optimized for sorting operations on DIALECT 4 in different scenarios.
            # check SORTBY details in https://redis.io/commands/ft.search/
            .dialect(4)
            .timeout(REDIS_QUERY_TIMEOUT)
        )
        params_dict = {
            "vec_param": np.array(vector).astype(cls.np_data_type).tobytes(),
            "K": top,
            **params,
        }
        if cls.algorithm == "HNSW":
            # 'EF_RUNTIME' is irrelevant for 'ADHOC_BF' policy
            if cls.hybrid_policy != "ADHOC_BF":
                params_dict["EF"] = cls.search_params["search_params"]["ef"]
        results = cls._ft.search(q, query_params=params_dict)

        return [(int(result.id), float(result.vector_score)) for result in results.docs]
