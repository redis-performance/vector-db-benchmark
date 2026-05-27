# vector-db-benchmark

[![Docker Pulls](https://img.shields.io/docker/pulls/redis/vector-db-benchmark)](https://hub.docker.com/r/redis/vector-db-benchmark)
[![CI](https://github.com/redis-performance/vector-db-benchmark/actions/workflows/test-basic-functionality-redisearch.yml/badge.svg)](https://github.com/redis-performance/vector-db-benchmark/actions/workflows/test-basic-functionality-redisearch.yml)

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

> [View results](https://redis.io/blog/benchmarking-results-for-vector-databases/)

There are various vector search engines available, and each of them may offer
a different set of features and efficiency. But how do we measure the
performance? There is no clear definition and in a specific case you
may worry about a specific thing, while not paying much attention to other aspects. This
project is a general framework for benchmarking different engines under the
same hardware constraints, so you can choose what works best for you.

Running any benchmark requires choosing an engine, a dataset and defining the
scenario against which it should be tested. A specific scenario may assume
running the server in a single or distributed mode, a different client
implementation and the number of client instances.


## Quick Start

### Quick Start with Docker

The easiest way to run vector-db-benchmark is using Docker. We provide pre-built images on Docker Hub.

```bash
# Pull the latest image
docker pull redis/vector-db-benchmark:latest

# Run with help
docker run --rm redis/vector-db-benchmark:latest run.py --help

# Check which datasets are available
docker run --rm redis/vector-db-benchmark:latest run.py --describe datasets

# Basic Redis benchmark with local Redis
docker run --rm -v $(pwd)/results:/app/results --network=host \
  redis/vector-db-benchmark:latest \
  run.py --host localhost --engines redis-default-simple --datasets glove-25-angular

# At the end of the run, you will find the results in the `results` directory. Lets open the summary one, in the precision summary

$ jq ".precision_summary" results/*-summary.json
{
  "0.91": {
    "qps": 1924.5,
    "p50": 49.828,
    "p95": 58.427
  },
  "0.94": {
    "qps": 1819.9,
    "p50": 51.68,
    "p95": 66.83
  },
  "0.9775": {
    "qps": 1477.8,
    "p50": 65.368,
    "p95": 73.849
  },
  "0.9950": {
    "qps": 1019.8,
    "p50": 95.115,
    "p95": 106.73
  }
}
```

### Using with Redis

For testing with Redis, start a Redis container first:

```bash
# Start Redis container
docker run -d --name redis-test -p 6379:6379 redis:8.2-bookworm

# Run benchmark against Redis

docker run --rm -v $(pwd)/results:/app/results --network=host \
  redis/vector-db-benchmark:latest \
  run.py --host localhost --engines redis-default-simple --datasets random-100

# Or use the convenience script
./docker-run.sh -H localhost -e redis-default-simple -d random-100


# Clean up Redis container when done
docker stop redis-test && docker rm redis-test
```

### Available Docker Images

- **Latest**: `redis/vector-db-benchmark:latest`

For detailed Docker setup and publishing information, see [DOCKER_SETUP.md](DOCKER_SETUP.md).


## Data sets

We have a number of precomputed data sets. All data sets have been pre-split into train/test and include ground truth data for the top-100 nearest neighbors.

| Dataset                                                                                                     | Dimensions |  Train size | Test size | Neighbors | Distance  |
| ----------------------------------------------------------------------------------------------------------- | ---------: |  ---------: | --------: | --------: | --------- |
| **LAION Image Embeddings (512D)**                                                                          |            |             |           |           |           |
| [LAION-1M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/)   |        512 |   1,000,000 |    10,000 |       100 | Cosine    |
| [LAION-10M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/)  |        512 |  10,000,000 |    10,000 |       100 | Cosine    |
| [LAION-20M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/)  |        512 |  20,000,000 |    10,000 |       100 | Cosine    |
| [LAION-40M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/)  |        512 |  40,000,000 |    10,000 |       100 | Cosine    |
| [LAION-100M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/) |        512 | 100,000,000 |    10,000 |       100 | Cosine    |
| [LAION-200M: subset of LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/) |        512 | 200,000,000 |    10,000 |       100 | Cosine    |
| [LAION-400M: from LAION 400M English (image embedings)](https://laion.ai/blog/laion-400-open-dataset/)      |        512 | 400,000,000 |    10,000 |       100 | Cosine    |
| **LAION Image Embeddings (768D)**                                                                          |            |             |           |           |           |
| [LAION-1M: 768D image embeddings](https://laion.ai/blog/laion-400-open-dataset/)                           |        768 |   1,000,000 |    10,000 |       100 | Cosine    |
| [LAION-1B: 768D image embeddings](https://laion.ai/blog/laion-400-open-dataset/)                           |        768 | 1,000,000,000|   10,000 |       100 | Cosine    |
| **Standard Benchmarks**                                                                                    |            |             |           |           |           |
| [GloVe-25: Word vectors](http://ann-benchmarks.com)                                                        |         25 |   1,183,514 |    10,000 |       100 | Cosine    |
| [GloVe-100: Word vectors](http://ann-benchmarks.com)                                                       |        100 |   1,183,514 |    10,000 |       100 | Cosine    |
| [Deep Image-96: CNN image features](http://ann-benchmarks.com)                                             |         96 |   9,990,000 |    10,000 |       100 | Cosine    |
| [GIST-960: Image descriptors](http://ann-benchmarks.com)                                                   |        960 |   1,000,000 |     1,000 |       100 | L2        |
| **Text and Knowledge Embeddings**                                                                          |            |             |           |           |           |
| [DBpedia OpenAI-1M: Knowledge embeddings](https://www.dbpedia.org/)                                       |      1,536 |   1,000,000 |    10,000 |       100 | Cosine    |
| [LAION Small CLIP: Small CLIP embeddings](https://laion.ai/blog/laion-400-open-dataset/)                   |        512 |     100,000 |     1,000 |       100 | Cosine    |
| **Yandex Datasets**                                                                                        |            |             |           |           |           |
| [Yandex T2I: Text-to-image embeddings](https://research.yandex.com/)                                      |        200 |   1,000,000 |   100,000 |       100 | Dot       |
| **Random and Synthetic**                                                                                   |            |             |           |           |           |
| Random-100: Small synthetic dataset                                                                        |        100 |         100 |         9 |         9 | Cosine    |
| Random-100-Euclidean: Small synthetic dataset                                                              |        100 |         100 |         9 |         9 | L2        |
| **Filtered Search Datasets**                                                                               |            |             |           |           |           |
| H&M-2048: Fashion product embeddings (with filters)                                                        |      2,048 |     105,542 |     2,000 |       100 | Cosine    |
| H&M-2048: Fashion product embeddings (no filters)                                                          |      2,048 |     105,542 |     2,000 |       100 | Cosine    |
| ArXiv-384: Academic paper embeddings (with filters)                                                        |        384 |   2,205,995 |    10,000 |       100 | Cosine    |
| ArXiv-384: Academic paper embeddings (no filters)                                                          |        384 |   2,205,995 |    10,000 |       100 | Cosine    |
| Random Match Keyword-100: Synthetic keyword matching (with filters)                                        |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Match Keyword-100: Synthetic keyword matching (no filters)                                          |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Match Int-100: Synthetic integer matching (with filters)                                            |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Match Int-100: Synthetic integer matching (no filters)                                              |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Range-100: Synthetic range queries (with filters)                                                   |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Range-100: Synthetic range queries (no filters)                                                     |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Geo Radius-100: Synthetic geo queries (with filters)                                                |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Geo Radius-100: Synthetic geo queries (no filters)                                                  |        100 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Match Keyword-2048: Large synthetic keyword matching (with filters)                                 |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Match Keyword-2048: Large synthetic keyword matching (no filters)                                   |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Match Int-2048: Large synthetic integer matching (with filters)                                     |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Match Int-2048: Large synthetic integer matching (no filters)                                       |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Range-2048: Large synthetic range queries (with filters)                                            |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Range-2048: Large synthetic range queries (no filters)                                              |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Geo Radius-2048: Large synthetic geo queries (with filters)                                         |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Geo Radius-2048: Large synthetic geo queries (no filters)                                           |      2,048 |     100,000 |     1,000 |       100 | Cosine    |
| Random Match Keyword Small Vocab-256: Small vocabulary keyword matching (with filters)                     |        256 |   1,000,000 |    10,000 |       100 | Cosine    |
| Random Match Keyword Small Vocab-256: Small vocabulary keyword matching (no filters)                       |        256 |   1,000,000 |    10,000 |       100 | Cosine    |


## How to run a benchmark?

Benchmarks are implemented in server-client mode, meaning that the server is
running in a single machine, and the client is running on another.

### Run the server

All engines are served using docker compose. The configuration is in the [servers](./engine/servers/).

To launch the server instance, run the following command:

```bash
cd ./engine/servers/<engine-configuration-name>
docker compose up
```

Containers are expected to expose all necessary ports, so the client can connect to them.

### Run the client

Install dependencies:

```bash
pip install poetry
poetry install
```

Run the benchmark:

```bash
# Basic usage examples
python run.py --engines redis-default-simple --datasets random-100
python run.py --engines redis-default-simple --datasets glove-25-angular
python run.py --engines "*-m-16-*" --datasets "glove-*"

# Using custom engine configurations from a JSON file
python run.py --engines-file custom_engines.json --datasets glove-25-angular

# Get information about available engines (with pattern matching)
python run.py --engines "*redis*" --describe engines --verbose

# Get information about engines from a custom file  
python run.py --engines-file custom_engines.json --describe engines --verbose

# Docker usage (recommended)
docker run --rm -v $(pwd)/results:/app/results --network=host \
  redis/vector-db-benchmark:latest \
  run.py --host localhost --engines redis-default-simple --datasets random-100

# Get help
python run.py --help
```

Command allows you to specify wildcards for engines and datasets.
Results of the benchmarks are stored in the `./results/` directory.

## Using Custom Engine Configurations

The benchmark tool supports two ways to specify which engine configurations to use:

### 1. Pattern Matching (Default)
Use the `--engines` flag with wildcard patterns to select configurations from the `experiments/configurations/` directory:

```bash
python run.py --engines "*redis*" --datasets glove-25-angular
python run.py --engines "qdrant-m-*" --datasets random-100
```

### 2. Custom Configuration File
Use the `--engines-file` flag to specify a JSON file containing custom engine configurations:

```bash
python run.py --engines-file my_engines.json --datasets glove-25-angular
```

The JSON file should contain an array of engine configuration objects. Each configuration must have a `name` field and follow the same structure as configurations in `experiments/configurations/`:

```json
[
  {
    "name": "my-custom-redis-config",
    "engine": "redis",
    "connection_params": {},
    "collection_params": {
      "algorithm": "hnsw",
      "data_type": "FLOAT32",
      "hnsw_config": {
        "M": 16,
        "DISTANCE_METRIC": "L2",
        "EF_CONSTRUCTION": 200
      }
    },
    "search_params": [
      {
        "parallel": 1,
        "top": 10,
        "search_params": {
          "ef": 100,
          "data_type": "FLOAT32"
        }
      }
    ],
    "upload_params": {
      "parallel": 16,
      "data_type": "FLOAT32"
    }
  }
]
```

**Note:** You cannot use both `--engines` and `--engines-file` at the same time.

## How to update benchmark parameters?

Each engine has a configuration file, which is used to define the parameters for the benchmark.
Configuration files are located in the [configuration](./experiments/configurations/) directory.

Each step in the benchmark process is using a dedicated configuration's path:

* `connection_params` - passed to the client during the connection phase.
* `collection_params` - parameters, used to create the collection, indexing parameters are usually defined here.
* `upload_params` - parameters, used to upload the data to the server.
* `search_params` - passed to the client during the search phase. Framework allows multiple search configurations for the same experiment run.

Exact values of the parameters are individual for each engine.

## How to register a dataset?

Datasets are configured in the [datasets/datasets.json](./datasets/datasets.json) file.
Framework will automatically download the dataset and store it in the [datasets](./datasets/) directory.

## How to implement a new engine?

There are a few base classes that you can use to implement a new engine.

* `BaseConfigurator` - defines methods to create collections, setup indexing parameters.
* `BaseUploader` - defines methods to upload the data to the server.
* `BaseSearcher` - defines methods to search the data.

See the examples in the [clients](./engine/clients) directory.

Once all the necessary classes are implemented, you can register the engine in the [ClientFactory](./engine/clients/client_factory.py).

