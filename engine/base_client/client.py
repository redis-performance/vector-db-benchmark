import json
import os
from datetime import datetime
from pathlib import Path
from typing import List
import warnings

from benchmark import ROOT_DIR
from benchmark.dataset import Dataset
from dataset_reader.base_reader import BaseReader
from engine.base_client.configure import BaseConfigurator
from engine.base_client.distances import Distance
from engine.base_client.search import BaseSearcher
from engine.base_client.upload import BaseUploader

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DETAILED_RESULTS = bool(int(os.getenv("DETAILED_RESULTS", False)))
REPETITIONS = int(os.getenv("REPETITIONS", 3))

warnings.filterwarnings("ignore", category=DeprecationWarning)


class BaseClient:
    def __init__(
        self,
        name: str,  # name of the experiment
        engine: str,  # name of the engine
        configurator: BaseConfigurator,
        uploader: BaseUploader,
        searchers: List[BaseSearcher],
    ):
        self.name = name
        self.configurator = configurator
        self.uploader = uploader
        self.searchers = searchers
        self.engine = engine

    def save_search_results(
        self, dataset_name: str, results: dict, search_id: int, search_params: dict
    ):
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        pid = os.getpid()  # Get the current process ID
        experiments_file = (
            f"{self.name}-{dataset_name}-search-{search_id}-{pid}-{timestamp}.json"
        )
        result_path = RESULTS_DIR / experiments_file
        with open(result_path, "w") as out:
            out.write(
                json.dumps(
                    {
                        "params": {
                            "dataset": dataset_name,
                            "experiment": self.name,
                            "engine": self.engine,
                            **search_params,
                        },
                        "results": results,
                    },
                    indent=2,
                )
            )
        return result_path

    def save_upload_results(
        self, dataset_name: str, results: dict, upload_params: dict,upload_start_idx:int,upload_end_idx:int,
    ):
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
        experiments_file = f"{self.name}-{dataset_name}-upload-{upload_start_idx}-{upload_end_idx}-{timestamp}.json"
        with open(RESULTS_DIR / experiments_file, "w") as out:
            upload_stats = {
                "params": {
                    "experiment": self.name,
                    "engine": self.engine,
                    "dataset": dataset_name,
                    "start_idx": upload_start_idx,
                    "end_idx": upload_end_idx,
                    **upload_params,
                },
                "results": results,
            }
            out.write(json.dumps(upload_stats, indent=2))

    def run_experiment(
        self,
        dataset: Dataset,
        skip_upload: bool = False,
        skip_search: bool = False,
        skip_if_exists: bool = True,
        parallels: [int] = [],
        upload_start_idx: int = 0,
        upload_end_idx: int = -1,
        num_queries: int = -1,
        ef_runtime: List[int] = [],
    ):
        execution_params = self.configurator.execution_params(
            distance=dataset.config.distance, vector_size=dataset.config.vector_size
        )
        reader = dataset.get_reader(execution_params.get("normalize", False))

        if skip_if_exists:
            pid = os.getpid()  # Get the current process ID
            glob_pattern = f"{self.name}-{dataset.config.name}-search-*-{pid}-*.json"
            existing_results = list(RESULTS_DIR.glob(glob_pattern))
            if len(existing_results) == len(self.searchers):
                print(
                    f"Skipping run for {self.name} since it already ran {len(self.searchers)} search configs previously"
                )
                return

        if not skip_upload:
            print("Experiment stage: Configure")
            self.configurator.configure(dataset)
            range_max_str = ":"
            if upload_end_idx > 0:
                range_max_str += f"{upload_end_idx}"
            print(f"Experiment stage: Upload. Vector range [{upload_start_idx}{range_max_str}]")
            upload_stats = self.uploader.upload(
                distance=dataset.config.distance, records=reader.read_data(upload_start_idx,upload_end_idx)
            )

            if not DETAILED_RESULTS:
                # Remove verbose stats from upload results
                upload_stats.pop("latencies", None)

            self.save_upload_results(
                dataset.config.name,
                upload_stats,
                upload_params={
                    **self.uploader.upload_params,
                    **self.configurator.collection_params,
                },
                upload_start_idx=upload_start_idx,
                upload_end_idx=upload_end_idx,
            )

        if not skip_search:
            print("Experiment stage: Search")
            for search_id, searcher in enumerate(self.searchers):
                if skip_if_exists:
                    pid = os.getpid()  # Get the current process ID
                    glob_pattern = (
                        f"{self.name}-{dataset.config.name}-search-{search_id}-{pid}-*.json"
                    )
                    existing_results = list(RESULTS_DIR.glob(glob_pattern))
                    print("Pattern", glob_pattern, "Results:", existing_results)
                    if len(existing_results) >= 1:
                        print(
                            f"Skipping search {search_id} as it already exists",
                        )
                        continue

                search_params = {**searcher.search_params}
                ef = "default"
                if "search_params" in search_params:
                    ef = search_params["search_params"].get("ef", "default")
                client_count = search_params.get("parallel", 1)

                # Filter by client count if parallels is specified
                filter_client_count = len(parallels) > 0
                if filter_client_count and (client_count not in parallels):
                    print(f"\tSkipping ef runtime: {ef}; #clients {client_count}")
                    continue

                # Filter by ef runtime if ef_runtime is specified
                filter_ef_runtime = len(ef_runtime) > 0
                if filter_ef_runtime and isinstance(ef, int) and (ef not in ef_runtime):
                    print(f"\tSkipping ef runtime: {ef}; #clients {client_count} (not in ef_runtime filter)")
                    continue

                if (precision := search_params.get("calibration_precision", None)) is not None:
                    top = search_params["top"]
                    calibration_param = search_params["calibration_param"]
                    calibration_value, calibration_precision = calibrate(
                        searcher,
                        calibration_param,
                        top,
                        precision,
                        dataset.config.distance,
                        reader,
                    )
                    print(
                        f"Calibrated {top=} {precision=} {calibration_value=} {calibration_precision=!s}"
                    )
                    searcher.search_params["search_params"][calibration_param] = calibration_value

                for repetition in range(1, REPETITIONS + 1):
                    print(
                        f"\tRunning repetition {repetition} ef runtime: {ef}; #clients {client_count}"
                    )

                    search_stats = searcher.search_all(
                        dataset.config.distance, reader.read_queries(), num_queries
                    )
                    # ensure we specify the client count in the results
                    search_params["parallel"] = client_count
                    if not DETAILED_RESULTS:
                        # Remove verbose stats from search results
                        search_stats.pop("latencies", None)
                        search_stats.pop("precisions", None)

                    self.save_search_results(
                        dataset.config.name, search_stats, search_id, search_params
                    )

        print("Experiment stage: Done")
        print("Results saved to: ", RESULTS_DIR)

    def delete_client(self):
        self.uploader.delete_client()
        self.configurator.delete_client()

        for s in self.searchers:
            s.delete_client()

def calibrate(
    searcher: BaseSearcher,
    calibration_param: str,
    min_value: int,
    precision: float,
    distance: Distance,
    reader: BaseReader,
    max_value: int = 1000,
) -> tuple[int, float]:
    """Calibrate searcher for a given precision."""
    if min_value > max_value:
        raise ValueError(
            f"{min_value=} cannot be greater than {max_value=}"
        )
    lower_bound = min_value
    upper_bound = max_value
    lower_bound_visited = False
    upper_bound_visited = False
    current = (lower_bound + upper_bound) // 2
    previous = current
    current_precision = 0
    while True:
        searcher.search_params["search_params"][calibration_param] = current
        search_stats = searcher.search_all(distance, reader.read_queries())
        previous_precision = current_precision
        current_precision = search_stats["mean_precisions"]
        if current_precision == precision:
            return current, current_precision
        elif current_precision > precision:
            upper_bound = current
            upper_bound_visited = True
        else:
            lower_bound = current
            lower_bound_visited = True
        next_value = (lower_bound + upper_bound) // 2
        if (
            (lower_bound_visited and next_value == lower_bound)
            or (upper_bound_visited and next_value == upper_bound)
        ):
            if abs(previous_precision - precision) < abs(current_precision - precision):
                final_precision = previous_precision
                final_value = previous
            else:
                final_precision = current_precision
                final_value = current
            return final_value, final_precision
        previous = current
        current = next_value
