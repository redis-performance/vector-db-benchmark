import os

REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_AUTH = os.getenv("REDIS_AUTH", None)
REDIS_USER = os.getenv("REDIS_USER", None)
REDIS_CLUSTER = bool(int(os.getenv("REDIS_CLUSTER", 0)))
REDIS_HYBRID_POLICY = os.getenv("REDIS_HYBRID_POLICY", None)
REDIS_DIALECT = os.getenv("REDIS_DIALECT", "4")

# 60 seconds timeout
REDIS_QUERY_TIMEOUT = int(os.getenv("REDIS_QUERY_TIMEOUT", 60 * 1000))
