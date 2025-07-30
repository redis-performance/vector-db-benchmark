import redis
import json
import numpy as np

MAX_INT = 10000000000000000000
# Connect to Redis
r = redis.Redis(host='54.78.191.248', port=13833, decode_responses=False, protocol=3)

queries_folder = '/home/ubuntu/vector-db-benchmark/datasets/arxiv-titles-384-angular/arxiv'
queries_file = f'{queries_folder}/tests.jsonl'

# Load one query from tests.jsonl to test
with open(queries_file) as f:
    first_query = json.loads(f.readline())

print(f"Original query has {len(first_query['closest_ids'])} expected results")

k = 5000
num_shards = 1

# Perform search with k=1000 and shard_k_ratio=1.0
query_vector = first_query['query']
search_results = r.execute_command(
    'FT.SEARCH', 'idx',
    f'*=>[KNN {k} @vector $query_vec AS score]',
    'PARAMS', '2', 'query_vec', np.array(query_vector, dtype=np.float32).tobytes(),
    'SORTBY', 'score',
    'RETURN', '1', 'score',
    'LIMIT', '0', f'{k}',
    'DIALECT', '4'
)

results = search_results[b'results']
results_count = len(results)
assert results_count == k, f"Expected {k} results, got {results_count}"
print(f"example search result: {results[1]}")
print(f"Search returned {results_count} results")

# Extract document IDs from search results (RESP3 format)
baseline_ids = []
for result in results:
    doc_id = result[b'id']
    baseline_ids.append(int(doc_id))

print(f"Extracted {len(baseline_ids)} baseline IDs")
print(f"First 10 IDs: {baseline_ids[:10]}")

def generate_baseline_for_queries(num_queries=MAX_INT):
    baseline_queries = []

    with open(queries_file) as f:
        for line_num, line in enumerate(f):
            query = json.loads(line)
            if line_num >= num_queries:
                break
            if line_num % 100 == 0:
                print(f"Processing query {line_num + 1}...")

            # Perform search for this query
            query_vector = query['query']
            search_results = r.execute_command(
                'FT.SEARCH', 'idx',
                f'*=>[KNN {k} @vector $query_vec AS score]',
                'PARAMS', '2', 'query_vec', np.array(query_vector, dtype=np.float32).tobytes(),
                'SORTBY', 'score',
                'RETURN', '1', 'score',
                'LIMIT', '0', f'{k}',
                'DIALECT', '4'
            )

            # Extract IDs from this search
            results = search_results[b'results']
            baseline_ids = []
            baseline_scores = []
            for result in results:
                doc_id = result[b'id']
                score = float(result[b'extra_attributes'][b'score'])
                baseline_ids.append(int(doc_id))
                baseline_scores.append(score)

            # Create new query with baseline results
            new_query = {
                'query': query['query'],
                'closest_ids': baseline_ids,
                'closest_scores': baseline_scores,
                'conditions': query['conditions']
            }
            baseline_queries.append(new_query)

    # Save baseline queries to new file
    output_file = f'{queries_folder}/tests_baseline_k{k}_{num_shards}_shards.jsonl'
    with open(output_file, 'w') as f:
        for query in baseline_queries:
            f.write(json.dumps(query) + '\n')

    print(f"Saved {len(baseline_queries)} baseline queries to {output_file}")
#
generate_baseline_for_queries(num_queries=5000)
