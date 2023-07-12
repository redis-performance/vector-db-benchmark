import os
REDIS_PORT = int(os.getenv("REDIS_PORT",6379))
REDIS_AUTH = os.getenv("REDIS_AUTH",None)
REDIS_USER = os.getenv("REDIS_USER",None)
# 90 seconds timeout
REDIS_QUERY_TIMEOUT = int(os.getenv("REDIS_QUERY_TIMEOUT",90*1000))
