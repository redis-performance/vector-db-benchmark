import functools
import time
from multiprocessing import Process, Queue
from typing import Iterable, List, Optional, Tuple
from itertools import islice

import numpy as np
import tqdm
import os

from dataset_reader.base_reader import Query

DEFAULT_TOP = 10
MAX_QUERIES = int(os.getenv("MAX_QUERIES", -1))



class BaseSearcher:
    MP_CONTEXT = None

    def __init__(self, host, connection_params, search_params):
        self.host = host
        self.connection_params = connection_params
        self.search_params = search_params

    @classmethod
    def init_client(
        cls, host: str, distance, connection_params: dict, search_params: dict
    ):
        raise NotImplementedError()

    @classmethod
    def get_mp_start_method(cls):
        return None

    @classmethod
    def search_one(
        cls, vector: List[float], meta_conditions, top: Optional[int]
    ) -> List[Tuple[int, float]]:
        raise NotImplementedError()

    @classmethod
    def _search_one(cls, query, top: Optional[int] = None):
        if top is None:
            top = (
                len(query.expected_result)
                if query.expected_result is not None and len(query.expected_result) > 0
                else DEFAULT_TOP
            )

        start = time.perf_counter()
        search_res = cls.search_one(query.vector, query.meta_conditions, top)
        end = time.perf_counter()

        precision = 1.0
        if query.expected_result:
            ids = set(x[0] for x in search_res)
            precision = len(ids.intersection(query.expected_result[:top])) / top
        return precision, end - start

    def search_all(
        self,
        distance,
        queries: Iterable[Query],
        num_queries: int = -1,
    ):
        parallel = self.search_params.get("parallel", 1)
        top = self.search_params.get("top", None)
        # setup_search may require initialized client
        self.init_client(
            self.host, distance, self.connection_params, self.search_params
        )
        self.setup_search()

        search_one = functools.partial(self.__class__._search_one, top=top)

        # Convert queries to a list for potential reuse
        queries_list = list(queries)

        # Handle MAX_QUERIES environment variable
        if MAX_QUERIES > 0:
            queries_list = queries_list[:MAX_QUERIES]
            print(f"Limiting queries to [0:{MAX_QUERIES-1}]")

        # Handle num_queries parameter
        if num_queries > 0:
            # If we need more queries than available, use a cycling generator
            if num_queries > len(queries_list) and len(queries_list) > 0:
                print(f"Requested {num_queries} queries but only {len(queries_list)} are available.")
                print(f"Using a cycling generator to efficiently process queries.")

                # Create a cycling generator function
                def cycling_query_generator(queries, total_count):
                    """Generate queries by cycling through the available ones."""
                    count = 0
                    while count < total_count:
                        for query in queries:
                            if count < total_count:
                                yield query
                                count += 1
                            else:
                                break

                # Use the generator instead of creating a full list
                used_queries = cycling_query_generator(queries_list, num_queries)
                # We need to know the total count for the progress bar
                total_query_count = num_queries
            else:
                used_queries = queries_list[:num_queries]
                total_query_count = len(used_queries)
                print(f"Using {num_queries} queries")
        else:
            used_queries = queries_list
            total_query_count = len(used_queries)

        if parallel == 1:
            # Single-threaded execution
            start = time.perf_counter()

            # Create a progress bar with the correct total
            pbar = tqdm.tqdm(total=total_query_count, desc="Processing queries", unit="queries")

            # Process queries with progress updates
            results = []
            for query in used_queries:
                results.append(search_one(query))
                pbar.update(1)

            # Close the progress bar
            pbar.close()

            total_time = time.perf_counter() - start
        else:
            # Dynamically calculate chunk size based on total_query_count
            chunk_size = max(1, total_query_count // parallel)

            # If used_queries is a generator, we need to handle it differently
            if hasattr(used_queries, '__next__'):
                # For generators, we'll create chunks on-the-fly
                query_chunks = []
                remaining = total_query_count
                while remaining > 0:
                    current_chunk_size = min(chunk_size, remaining)
                    chunk = [next(used_queries) for _ in range(current_chunk_size)]
                    query_chunks.append(chunk)
                    remaining -= current_chunk_size
            else:
                # For lists, we can use the chunked_iterable function
                query_chunks = list(chunked_iterable(used_queries, chunk_size))

            # Function to be executed by each worker process
            def worker_function(chunk, result_queue):
                self.__class__.init_client(
                    self.host,
                    distance,
                    self.connection_params,
                    self.search_params,
                )
                self.setup_search()
                results = process_chunk(chunk, search_one)
                result_queue.put(results)

            # Create a queue to collect results
            result_queue = Queue()

            # Create and start worker processes
            processes = []
            for chunk in query_chunks:
                process = Process(target=worker_function, args=(chunk, result_queue))
                processes.append(process)
                process.start()

            # Start measuring time for the critical work
            start = time.perf_counter()

            # Create a progress bar for the total number of queries
            pbar = tqdm.tqdm(total=total_query_count, desc="Processing queries", unit="queries")

            # Collect results from all worker processes
            results = []
            for _ in processes:
                chunk_results = result_queue.get()
                results.extend(chunk_results)
                # Update the progress bar with the number of processed queries in this chunk
                pbar.update(len(chunk_results))

            # Close the progress bar
            pbar.close()

            # Wait for all worker processes to finish
            for process in processes:
                process.join()

            # Stop measuring time for the critical work
            total_time = time.perf_counter() - start

        # Extract precisions and latencies (outside the timed section)
        precisions, latencies = zip(*results)

        self.__class__.delete_client()

        return {
            "total_time": total_time,
            "mean_time": np.mean(latencies),
            "mean_precisions": np.mean(precisions),
            "std_time": np.std(latencies),
            "min_time": np.min(latencies),
            "max_time": np.max(latencies),
            "rps": len(latencies) / total_time,
            "p50_time": np.percentile(latencies, 50),
            "p95_time": np.percentile(latencies, 95),
            "p99_time": np.percentile(latencies, 99),
            "precisions": precisions,
            "latencies": latencies,
        }

    def setup_search(self):
        pass

    def post_search(self):
        pass

    @classmethod
    def delete_client(cls):
        pass


def chunked_iterable(iterable, size):
    """Yield successive chunks of a given size from an iterable."""
    it = iter(iterable)
    while chunk := list(islice(it, size)):
        yield chunk


def process_chunk(chunk, search_one):
    """Process a chunk of queries using the search_one function."""
    # No progress bar in worker processes to avoid cluttering the output
    return [search_one(query) for query in chunk]


def process_chunk_wrapper(chunk, search_one):
    """Wrapper to process a chunk of queries."""
    return process_chunk(chunk, search_one)
