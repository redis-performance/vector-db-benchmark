import functools
import random
import itertools
import time
from multiprocessing import Process, Queue
from typing import Iterable, List, Optional, Tuple
from itertools import islice

import numpy as np
import tqdm
import os
from ml_dtypes import bfloat16

from dataset_reader.base_reader import Query
from engine.base_client.utils import check_data_type

DEFAULT_TOP = 10
MAX_QUERIES = int(os.getenv("MAX_QUERIES", -1))



class BaseSearcher:
    _doc_id_counter = None  # Will be initialized per process
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
    def insert_one(cls, doc_id: int, vector: List[float], meta_conditions):
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

    @classmethod
    def _get_doc_id_counter(cls):
        if cls._doc_id_counter is None:
            # Use process ID to create unique starting point for each worker
            process_id = os.getpid()
            # Each process gets a unique range: 1000000000 + (pid * 1000000)
            start_offset = 1000000000 + (process_id % 1000) * 1000000
            cls._doc_id_counter = itertools.count(start_offset)
        return cls._doc_id_counter

    @classmethod
    def _insert_one(cls, query):
        start = time.perf_counter()

        # Generate unique doc_id with process-safe counter
        doc_id = next(cls._get_doc_id_counter())

        cls.insert_one(str(doc_id), query.vector, query.meta_conditions)
        end = time.perf_counter()
        # No precision metric for inserts, so precision=1.0
        return 1.0, end - start

    def search_all(
        self,
        distance,
        queries: Iterable[Query],
        num_queries: int = -1,
        insert_fraction: float = 0.0,
    ):
        parallel = self.search_params.get("parallel", 1)
        top = self.search_params.get("top", None)
        single_search_params = self.search_params.get("search_params", None)
        if single_search_params:
            data_type = check_data_type(single_search_params.get("data_type", "FLOAT32").upper())
        else:
            data_type = np.float32  # Default data type if not specified
        # setup_search may require initialized client
        self.init_client(
            self.host, distance, self.connection_params, self.search_params
        )
        self.setup_search()

        search_one = functools.partial(self.__class__._search_one, top=top)
        insert_one = functools.partial(self.__class__._insert_one)

        # Convert queries to a list for potential reuse
        # Also, converts query vectors to bytes beforehand, preparing them for sending to client without affecting search time measurements
        queries_list = []
        for query in queries:
            query.vector = np.array(query.vector).astype(data_type).tobytes()
            queries_list.append(query)
        
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
            # Create a progress bar with the correct total
            pbar = tqdm.tqdm(total=total_query_count, desc="Processing queries", unit="queries")

            # Single-threaded execution
            start = time.perf_counter()

            # Process queries with progress updates
            results = []
            total_insert_count = 0
            total_search_count = 0
            all_insert_latencies = []
            all_search_latencies = []
            
            for query in used_queries:
                if random.random() < insert_fraction:
                    precision, latency = insert_one(query)
                    total_insert_count += 1
                    all_insert_latencies.append(latency)
                    results.append(('insert', precision, latency))
                else:
                    precision, latency = search_one(query)
                    total_search_count += 1
                    all_search_latencies.append(latency)
                    results.append(('search', precision, latency))
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

            # Create a queue to collect results
            result_queue = Queue()

            # Create worker processes
            processes = []
            for chunk in query_chunks:
                process = Process(target=worker_function, args=(self, distance, search_one, insert_one, 
                                                                chunk, result_queue, insert_fraction))
                processes.append(process)

            # Start worker processes
            for process in processes:
                process.start()

            # Collect results from all worker processes
            results = []
            total_insert_count = 0
            total_search_count = 0
            all_insert_latencies = []
            all_search_latencies = []
            min_start_time = time.perf_counter()

            for _ in processes:
                proc_start_time, chunk_results, insert_count, search_count, insert_latencies, search_latencies = result_queue.get()
                results.extend(chunk_results)
                total_insert_count += insert_count
                total_search_count += search_count
                all_insert_latencies.extend(insert_latencies)
                all_search_latencies.extend(search_latencies)
                
                # Update min_start_time if necessary
                if proc_start_time < min_start_time:
                    min_start_time = proc_start_time

            # Stop measuring time for the critical work
            total_time = time.perf_counter() - min_start_time

            # Wait for all worker processes to finish
            for process in processes:
                process.join()

        # Extract overall precisions and latencies
        all_precisions = [result[1] for result in results]
        all_latencies = [result[2] for result in results]

        # Calculate search-only precisions (exclude inserts from precision calculation)
        search_precisions = [result[1] for result in results if result[0] == 'search']

        self.__class__.delete_client()

        return {
            # Overall metrics
            "total_time": total_time,
            "total_operations": len(all_latencies),
            "rps": len(all_latencies) / total_time,
            
            # Search metrics
            "search_count": total_search_count,
            "search_rps": total_search_count / total_time if total_search_count > 0 else 0,
            "mean_search_time": np.mean(all_search_latencies) if all_search_latencies else 0,
            "mean_search_precision": np.mean(search_precisions) if search_precisions else 0,
            "p50_search_time": np.percentile(all_search_latencies, 50) if all_search_latencies else 0,
            "p95_search_time": np.percentile(all_search_latencies, 95) if all_search_latencies else 0,
            "p99_search_time": np.percentile(all_search_latencies, 99) if all_search_latencies else 0,
            
            # Insert metrics
            "insert_count": total_insert_count,
            "insert_rps": total_insert_count / total_time if total_insert_count > 0 else 0,
            "mean_insert_time": np.mean(all_insert_latencies) if all_insert_latencies else 0,
            "p50_insert_time": np.percentile(all_insert_latencies, 50) if all_insert_latencies else 0,
            "p95_insert_time": np.percentile(all_insert_latencies, 95) if all_insert_latencies else 0,
            "p99_insert_time": np.percentile(all_insert_latencies, 99) if all_insert_latencies else 0,
            
            # Mixed workload metrics
            "actual_insert_fraction": total_insert_count / len(all_latencies) if len(all_latencies) > 0 else 0,
            "target_insert_fraction": insert_fraction,
            
            # Legacy compatibility (for existing code that expects these)
            "mean_time": np.mean(all_latencies),
            "mean_precisions": np.mean(search_precisions) if search_precisions else 1.0,  # Only search precisions
            "std_time": np.std(all_latencies),
            "min_time": np.min(all_latencies),
            "max_time": np.max(all_latencies),
            "p50_time": np.percentile(all_latencies, 50),
            "p95_time": np.percentile(all_latencies, 95),
            "p99_time": np.percentile(all_latencies, 99),
            "precisions": search_precisions,  # Only search precisions
            "latencies": all_latencies,
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

def process_chunk(chunk, search_one, insert_one, insert_fraction):
    results = []
    insert_count = 0
    search_count = 0
    insert_latencies = []
    search_latencies = []
    
    for i, query in enumerate(chunk):
        if random.random() < insert_fraction:
            precision, latency = insert_one(query)
            insert_count += 1
            insert_latencies.append(latency)
            results.append(('insert', precision, latency))
        else:
            precision, latency = search_one(query)
            search_count += 1
            search_latencies.append(latency)
            results.append(('search', precision, latency))
    
    return results, insert_count, search_count, insert_latencies, search_latencies

# Function to be executed by each worker process
def worker_function(self, distance, search_one, insert_one, chunk, result_queue, insert_fraction=0.0):
    self.init_client(
        self.host,
        distance,
        self.connection_params,
        self.search_params,
    )
    self.setup_search()

    start_time = time.perf_counter()
    results, insert_count, search_count, insert_latencies, search_latencies = process_chunk(
        chunk, search_one, insert_one, insert_fraction
    )
    result_queue.put((start_time, results, insert_count, search_count, insert_latencies, search_latencies))