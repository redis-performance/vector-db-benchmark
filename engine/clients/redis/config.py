import os
REDIS_PORT = int(os.getenv("REDIS_PORT",6379))
REDIS_AUTH = os.getenv("REDIS_AUTH",None)
REDIS_USER = os.getenv("REDIS_USER",None)
