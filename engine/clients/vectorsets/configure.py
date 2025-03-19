import redis
from redis import Redis, RedisCluster

from benchmark.dataset import Dataset
from engine.base_client.configure import BaseConfigurator
from engine.clients.vectorsets.config import (
    REDIS_AUTH,
    REDIS_CLUSTER,
    REDIS_PORT,
    REDIS_USER,
)


class RedisVsetConfigurator(BaseConfigurator):

    def __init__(self, host, collection_params: dict, connection_params: dict):
        super().__init__(host, collection_params, connection_params)
        redis_constructor = RedisCluster if REDIS_CLUSTER else Redis
        self._is_cluster = True if REDIS_CLUSTER else False
        self.client = redis_constructor(
            host=host, port=REDIS_PORT, password=REDIS_AUTH, username=REDIS_USER
        )
        self.client.flushall()

    def clean(self):
        conns = [self.client]
        if self._is_cluster:
            conns = [
                self.client.get_redis_connection(node)
                for node in self.client.get_primaries()
            ]
        for conn in conns:
            index = conn.ft()
            try:
                 conn.flushall()
            except redis.ResponseError as e:
                print(e)

    def recreate(self, dataset: Dataset, collection_params):
        pass


if __name__ == "__main__":
    pass
