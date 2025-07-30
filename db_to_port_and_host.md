HOST1: 18.200.246.132
HOST2: 54.195.17.70

WORKERS=8 SHARD_COUNT=10 REDIS_PORT=15045 host=18.200.246.132 # WITH FILTERS WITH QPF


SHARD_COUNT=1 K_RATIO=1.0 REDIS_PORT=16556 python3 run.py --host 18.203.186.188 --engines redis-hnsw-m-16-ef-128  --datasets arxiv-titles-384-angular-no-filters --parallels 100  --skip-upload --queries 5000

# upload NOFILTERS
WORKERS=8  SHARD_COUNT=10 REDIS_PORT=11926 python3 run.py --host 54.78.191.248 --engines redis-hnsw-m-16-ef-128  --datasets arxiv-titles-384-angular-no-filters --parallels 100  --skip-search
# upload FILTERS
WORKERS=8 SHARD_COUNT=10 REDIS_PORT=15045 python3 run.py --host 18.200.246.132 --engines redis-hnsw-m-16-ef-128  --datasets arxiv-titles-384-angular-filters --parallels 100  --skip-search
# search NOFILTERS
WORKERS=8 SHARD_COUNT=10 REDIS_PORT=19387 K_RATIO=1.0 python3 run.py --host 18.203.186.188 --engines redis-hnsw-m-16-ef-128  --datasets arxiv-titles-384-angular-no-filters --parallels 100  --skip-upload --queries 5000
# search FILTERS
WORKERS=8 SHARD_COUNT=10 REDIS_PORT=15194 K_RATIO=1.0 python3 run.py --host 54.78.191.248 --engines redis-hnsw-m-16-ef-128  --datasets arxiv-titles-384-angular-filters --parallels 100  --skip-upload --queries 5000
# downloaddataset FILTERS
WORKERS=8 SHARD_COUNT=10 REDIS_PORT=16318 K_RATIO=1.0 python3 run.py --host 18.203.186.188 --engines redis-hnsw-m-16-ef-128  --datasets arxiv-titles-384-angular-filters --parallels 100  --skip-upload --skip-search --queries 5000
