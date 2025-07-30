WORKERS=8 SHARD_COUNT=10 REDIS_PORT=14941 host=54.78.191.248
WORKERS=8 SHARD_COUNT=5 REDIS_PORT=12041 host=18.203.186.188
SHARD_COUNT=2 REDIS_PORT=12558 host=54.78.191.248
SHARD_COUNT=1 REDIS_PORT=16556 host=18.203.186.188
WORKERS=8 SHARD_COUNT=10 REDIS_PORT=16318 host=18.203.186.188 # WITH FILTERS
WORKERS=8 SHARD_COUNT=1 REDIS_PORT=13833 host=54.78.191.248 # WITH FILTERS
WORKERS=8 SHARD_COUNT=5 REDIS_PORT=18447 host=54.78.191.248 # WITH FILTERS
WORKERS=8 SHARD_COUNT=2 REDIS_PORT=19459 host=18.203.186.188 # WITH FILTERS
WORKERS=8 SHARD_COUNT=10 REDIS_PORT=15194 host=54.78.191.248 # WITH FILTERS WITH QPF


SHARD_COUNT=1 K_RATIO=1.0 REDIS_PORT=16556 python3 run.py --host 18.203.186.188 --engines redis-hnsw-m-16-ef-128  --datasets arxiv-titles-384-angular-no-filters --parallels 100  --skip-upload --queries 5000

# upload NOFILTERS
WORKERS=8  SHARD_COUNT=10 REDIS_PORT=11926 python3 run.py --host 54.78.191.248 --engines redis-hnsw-m-16-ef-128  --datasets arxiv-titles-384-angular-no-filters --parallels 100  --skip-search
# upload FILTERS
WORKERS=8 SHARD_COUNT=10 REDIS_PORT=15194 python3 run.py --host 54.78.191.248 --engines redis-hnsw-m-16-ef-128  --datasets arxiv-titles-384-angular-filters --parallels 100  --skip-search
#search NOFILTERS
WORKERS=8 SHARD_COUNT=10 REDIS_PORT=19387 K_RATIO=1.0 python3 run.py --host 18.203.186.188 --engines redis-hnsw-m-16-ef-128  --datasets arxiv-titles-384-angular-no-filters --parallels 100  --skip-upload --queries 5000
# search FILTERS
WORKERS=8 SHARD_COUNT=10 REDIS_PORT=15194 K_RATIO=1.0 python3 run.py --host 54.78.191.248 --engines redis-hnsw-m-16-ef-128  --datasets arxiv-titles-384-angular-filters --parallels 100  --skip-upload --queries 5000
# downloaddataset FILTERS
WORKERS=8 SHARD_COUNT=10 REDIS_PORT=16318 K_RATIO=1.0 python3 run.py --host 18.203.186.188 --engines redis-hnsw-m-16-ef-128  --datasets arxiv-titles-384-angular-filters --parallels 100  --skip-upload --skip-search --queries 5000
