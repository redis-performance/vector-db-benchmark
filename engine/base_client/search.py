import functools
import time
from multiprocessing import get_context
from typing import Iterable, List, Optional, Tuple
from itertools import islice

import numpy as np
import tqdm

from dataset_reader.base_reader import Query

DEFAULT_TOP = 10


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
    ):
        parallel = self.search_params.get("parallel", 1)
        top = self.search_params.get("top", None)

        # setup_search may require initialized client
        self.init_client(
            self.host, distance, self.connection_params, self.search_params
        )
        self.setup_search()

        search_one = functools.partial(self.__class__._search_one, top=top)

        if parallel == 1:
            start = time.perf_counter()
            precisions, latencies = list(
                zip(*[search_one(query) for query in tqdm.tqdm(queries)])
            )
        else:
            ctx = get_context(self.get_mp_start_method())

            def process_initializer():
                """Initialize each process before starting the search."""
                self.__class__.init_client(
                    self.host,
                    distance,
                    self.connection_params,
                    self.search_params,
                )
                self.setup_search()

            # Dynamically chunk the generator
            query_chunks = list(chunked_iterable(queries, max(1, parallel)))

            with ctx.Pool(
                processes=parallel,
                initializer=process_initializer,
            ) as pool:
                if parallel > 10:
                    time.sleep(15)  # Wait for all processes to start
                start = time.perf_counter()
                results = pool.starmap(
                    process_chunk,
                    [(chunk, search_one) for chunk in query_chunks],
                )
                precisions, latencies = zip(*[result for chunk in results for result in chunk])

        total_time = time.perf_counter() - start

        self.__class__.delete_client()

        return {
            "total_time": total_time,
            "mean_time": np.mean(latencies),
            "mean_precisions": np.mean(precisions),
            "std_time": np.std(latencies),
            "min_time": np.min(latencies),
            "max_time": np.max(latencies),
            "rps": len(latencies) / total_time,
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
    return [search_one(query) for query in chunk]
