import fnmatch
import traceback
from typing import List

import stopit
import typer

from benchmark.config_read import read_dataset_config, read_engine_configs
from benchmark.dataset import Dataset
from engine.base_client import IncompatibilityError
from engine.clients.client_factory import ClientFactory

app = typer.Typer()


@app.command()
def run(
    engines: List[str] = typer.Option(["*"]),
    datasets: List[str] = typer.Option(["*"]),
    parallels: List[int] = typer.Option([]),
    host: str = "localhost",
    skip_upload: bool = False,
    skip_search: bool = False,
    skip_if_exists: bool = True,
    exit_on_error: bool = True,
    timeout: float = 86400.0,
    upload_start_idx: int = 0,
    upload_end_idx: int = -1,
    queries: int = typer.Option(-1, help="Number of queries to run. If the available queries are fewer, they will be reused."),
    ef_runtime: List[int] = typer.Option([], help="Filter search experiments by ef runtime values. Only experiments with these ef values will be run."),
):
    """
    Example:
        python3 run.py --engines *-m-16-* --engines qdrant-* --datasets glove-*
    """
    all_engines = read_engine_configs()
    all_datasets = read_dataset_config()

    selected_engines = {
        name: config
        for name, config in all_engines.items()
        if any(fnmatch.fnmatch(name, engine) for engine in engines)
    }

    selected_datasets = {
        name: config
        for name, config in all_datasets.items()
        if any(fnmatch.fnmatch(name, dataset) for dataset in datasets)
    }

    for engine_name, engine_config in selected_engines.items():
        for dataset_name, dataset_config in selected_datasets.items():
            print(f"Running experiment: {engine_name} - {dataset_name}")
            client = ClientFactory(host).build_client(engine_config)
            dataset = Dataset(
                dataset_config,
                skip_upload,
                skip_search,
                upload_start_idx,
                upload_end_idx,
            )
            dataset.download()
            try:
                with stopit.ThreadingTimeout(timeout) as tt:
                    client.run_experiment(
                        dataset,
                        skip_upload,
                        skip_search,
                        skip_if_exists,
                        parallels,
                        upload_start_idx,
                        upload_end_idx,
                        queries,
                        ef_runtime,
                    )
                client.delete_client()

                # If the timeout is reached, the server might be still in the
                # middle of some background processing, like creating the index.
                # Next experiment should not be launched. It's better to reset
                # the server state manually.
                if tt.state != stopit.ThreadingTimeout.EXECUTED:
                    print(
                        f"Timed out {engine_name} - {dataset_name}, "
                        f"exceeded {timeout} seconds"
                    )
                    exit(2)
            except IncompatibilityError as e:
                print(f"Skipping {engine_name} - {dataset_name}, incompatible params")
                continue
            except KeyboardInterrupt as e:
                traceback.print_exc()
                exit(1)
            except Exception as e:
                print(f"Experiment {engine_name} - {dataset_name} interrupted")
                traceback.print_exc()
                if exit_on_error:
                    raise e
                continue


if __name__ == "__main__":
    app()
