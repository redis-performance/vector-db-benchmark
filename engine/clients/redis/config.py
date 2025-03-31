import os

REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_AUTH = os.getenv("REDIS_AUTH", None)
REDIS_USER = os.getenv("REDIS_USER", None)
REDIS_CLUSTER = bool(int(os.getenv("REDIS_CLUSTER", 0)))
# One of BATCHES and ADHOC_BF
# check https://redis.io/docs/latest/develop/interact/search-and-query/advanced-concepts/vectors/#pre-filter-query-attributes-hybrid-approach
REDIS_HYBRID_POLICY = os.getenv("REDIS_HYBRID_POLICY", "")
REDIS_KEEP_DOCUMENTS = bool(os.getenv("REDIS_KEEP_DOCUMENTS", 0))
REDIS_JUST_INDEX = bool(os.getenv("REDIS_JUST_INDEX", 0))
GPU_STATS = bool(int(os.getenv("GPU_STATS", 0)))
GPU_STATS_ENDPOINT = os.getenv("GPU_STATS_ENDPOINT", None)


# 60 seconds timeout
REDIS_QUERY_TIMEOUT = int(os.getenv("REDIS_QUERY_TIMEOUT", 90 * 1000))
