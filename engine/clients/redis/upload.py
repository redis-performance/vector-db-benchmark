from typing import List, Optional

import numpy as np
import redis

from engine.base_client.upload import BaseUploader
from engine.clients.redis.config import REDIS_PORT, REDIS_KEY_PREFIX
from engine.clients.redis.helper import convert_to_redis_coords


class RedisUploader(BaseUploader):
    client = None
    upload_params = {}

    @classmethod
    def init_client(cls, host, distance, connection_params, upload_params):
        cls.client = redis.Redis(host=host, port=REDIS_PORT, db=0)
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
            payload = {}
            for k, v in meta.items():
                # This is a patch for arxiv-titles dataset where we have a list of "labels", and
                # we want to index all of them under the same TAG field (whose seperator is ';').
                if k == 'labels':
                    payload[k] = ";".join(v)
                if v is not None and not isinstance(v, dict) and not isinstance(v, list):
                    payload[k] = v
            # Redis treats geopoints differently and requires putting them as
            # a comma-separated string with lat and lon coordinates
            geopoints = {
                k: ",".join(map(str, convert_to_redis_coords(v["lon"], v["lat"])))
                for k, v in meta.items()
                if isinstance(v, dict)
            }
            cls.client.hset(
                REDIS_KEY_PREFIX + str(idx),
                mapping={
                    "vector": np.array(vec).astype(np.float32).tobytes(),
                    **payload,
                    **geopoints,
                },
            )
        p.execute()

    @classmethod
    def post_upload(cls, _distance):
        return {}
