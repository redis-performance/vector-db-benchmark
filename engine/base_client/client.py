import json
import os
from datetime import datetime
from pathlib import Path
from typing import List
import warnings

from benchmark import ROOT_DIR
from benchmark.dataset import Dataset
from engine.base_client.configure import BaseConfigurator
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
        experiments_file = (
            f"{self.name}-{dataset_name}-search-{search_id}-{timestamp}.json"
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
    ):
        execution_params = self.configurator.execution_params(
            distance=dataset.config.distance, vector_size=dataset.config.vector_size
        )
        reader = dataset.get_reader(execution_params.get("normalize", False))

        if skip_if_exists:
            glob_pattern = f"{self.name}-{dataset.config.name}-search-*-*.json"
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
                    glob_pattern = (
                        f"{self.name}-{dataset.config.name}-search-{search_id}-*.json"
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
                filter_client_count = len(parallels) > 0
                if filter_client_count and (client_count not in parallels):
                    print(f"\tSkipping ef runtime: {ef}; #clients {client_count}")
                    continue
                for repetition in range(1, REPETITIONS + 1):
                    print(
                        f"\tRunning repetition {repetition} ef runtime: {ef}; #clients {client_count}"
                    )

                    search_stats = searcher.search_all(
                        dataset.config.distance, reader.read_queries()
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
