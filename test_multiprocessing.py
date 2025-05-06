from engine.base_client.search import BaseSearcher
from dataset_reader.base_reader import Query
import time

class TestSearcher(BaseSearcher):
    @classmethod
    def init_client(cls, host, distance, connection_params, search_params):
        pass

    @classmethod
    def search_one(cls, vector, meta_conditions, top):
        return []

    @classmethod
    def _search_one(cls, query, top=None):
        # Add a small delay to simulate real work
        time.sleep(0.001)
        return 1.0, 0.1

    def setup_search(self):
        pass

# Create test queries
queries = [Query(vector=[0.1]*10, meta_conditions=None, expected_result=None) for _ in range(1000)]

# Create a searcher with parallel=10
searcher = TestSearcher('localhost', {}, {'parallel': 10})

# Run the search_all method
start = time.perf_counter()
results = searcher.search_all('cosine', queries)
total_time = time.perf_counter() - start

print(f'Number of queries: {len(results["latencies"])}')
print(f'Total time: {total_time:.6f} seconds')
print(f'Throughput: {results["rps"]:.2f} queries/sec')
