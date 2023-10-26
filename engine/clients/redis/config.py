import os

REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_AUTH = os.getenv("REDIS_AUTH", None)
REDIS_USER = os.getenv("REDIS_USER", None)
REDIS_CLUSTER = bool(int(os.getenv("REDIS_CLUSTER", 0)))
REDIS_HYBRID_POLICY = os.getenv("REDIS_HYBRID_POLICY", None)
DISABLE_CLEAN = bool(int(os.getenv("DISABLE_CLEAN", 0)))
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "")

# 90 seconds timeout
REDIS_QUERY_TIMEOUT = int(os.getenv("REDIS_QUERY_TIMEOUT", 90 * 1000))
