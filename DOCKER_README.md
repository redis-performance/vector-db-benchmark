# Redis Vector Database Benchmark

A comprehensive benchmarking tool for vector databases, including Redis (both RediSearch and Vector Sets), Weaviate, Milvus, Qdrant, OpenSearch, Postgres, and others...

In a one-liner cli tool you can get this and much more:

```
docker run --rm --network=host redis/vector-db-benchmark:latest run.py --host localhost --engines vectorsets-fp32-default --datasets glove-100-angular --parallels 100
(...)
================================================================================
BENCHMARK RESULTS SUMMARY
Experiment: vectorsets-fp32-default - glove-100-angular
================================================================================

Precision vs Performance Trade-off:
--------------------------------------------------
Precision  QPS      P50 (ms)   P95 (ms)  
--------------------------------------------------
0.86       1408.3   61.877     107.548   
0.80       2136.3   38.722     69.102    
0.72       2954.3   25.820     48.072    
0.68       3566.5   20.229     38.581    

QPS vs Precision Trade-off - vectorsets-fp32-default - glove-100-angular (up and to the right is better):

  3566 │●                                                           
       │             ●                                              
       │                                                            
  2594 │                                                            
       │                                       ●                    
       │                                                            
  1621 │                                                           ●
       │                                                            
       │                                                            
   648 │                                                            
       │                                                            
       │                                                            
     0 │                                                            
       └────────────────────────────────────────────────────────────
        0.680          0.726          0.772          0.817          
        Precision (0.0 = 0%, 1.0 = 100%)
================================================================================

```

## Quick Start

```bash
# Pull the latest image
docker pull redis/vector-db-benchmark:latest

# Run with help
docker run --rm redis/vector-db-benchmark:latest run.py --help

# Check available datasets
docker run --rm redis/vector-db-benchmark:latest run.py --describe datasets

# Basic Redis benchmark (requires local Redis)
docker run --rm -v $(pwd)/results:/app/results --network=host \
  redis/vector-db-benchmark:latest \
  run.py --host localhost --engines redis-default-simple --dataset random-100
```

## Features

- **42+ Datasets**: Pre-configured datasets from 25 to 1B+ vectors
- **Multiple Engines**: Redis, Qdrant, Weaviate, Milvus, and more
- **Real-time Monitoring**: Live performance metrics during benchmarks
- **Precision Analysis**: Detailed accuracy vs performance trade-offs
- **Easy Discovery**: `--describe` commands for datasets and engines

## Available Tags

- `latest` - Latest development build from update.redisearch branch

## Redis quick start

### Redis 8.2 with RediSearch
```bash
# Start Redis 8.2 with built-in vector support
docker run -d --name redis-test -p 6379:6379 redis:8.2-rc1-bookworm

# Run benchmark
docker run --rm -v $(pwd)/results:/app/results --network=host \
  redis/vector-db-benchmark:latest \
  run.py --host localhost --engines redis-default-simple --dataset glove-25-angular
```


## Common Usage Patterns

### Explore Available Options
```bash
# List all datasets
docker run --rm redis/vector-db-benchmark:latest run.py --describe datasets

# List all engines
docker run --rm redis/vector-db-benchmark:latest run.py --describe engines
```

### Run Benchmarks
```bash
# Quick test with small dataset
docker run --rm -v $(pwd)/results:/app/results --network=host \
  redis/vector-db-benchmark:latest \
  run.py --host localhost --engines redis-default-simple --dataset random-100

# Comprehensive benchmark with multiple configurations
docker run --rm -v $(pwd)/results:/app/results --network=host \
  redis/vector-db-benchmark:latest \
  run.py --host localhost --engines "*redis*" --dataset glove-25-angular

# With Redis authentication
docker run --rm -v $(pwd)/results:/app/results --network=host \
  -e REDIS_AUTH=mypassword -e REDIS_USER=myuser \
  redis/vector-db-benchmark:latest \
  run.py --host localhost --engines redis-default-simple --dataset random-100
```

### Results Analysis
```bash
# View precision summary
jq '.precision_summary' results/*-summary.json

# View detailed results
jq '.search' results/*-summary.json
```

## Volume Mounts

- `/app/results` - Benchmark results (JSON files)
- `/app/datasets` - Dataset storage (optional, auto-downloaded)

## Environment Variables

- `REDIS_HOST` - Redis server hostname (default: localhost)
- `REDIS_PORT` - Redis server port (default: 6379)
- `REDIS_AUTH` - Redis password (default: None)
- `REDIS_USER` - Redis username (default: None)
- `REDIS_CLUSTER` - Enable Redis cluster mode (default: 0)

## Performance Tips

1. **Use `--network=host`** for best performance with local Redis
2. **Mount results volume** to persist benchmark data
3. **Start with small datasets** (random-100, glove-25-angular) for testing
4. **Use wildcard patterns** to test multiple configurations: `--engines "*-m-16-*"`

## Example Output

```json
{
  "precision_summary": {
    "0.91": {
      "qps": 1924.5,
      "p50": 49.828,
      "p95": 58.427
    },
    "0.94": {
      "qps": 1819.9,
      "p50": 51.68,
      "p95": 66.83
    }
  }
}
```

## Support

- **GitHub**: [redis-performance/vector-db-benchmark](https://github.com/redis-performance/vector-db-benchmark)
- **Issues**: Report bugs and feature requests on GitHub
- **Documentation**: Full documentation available in the repository

## License

This project is licensed under the MIT License - see the repository for details.
