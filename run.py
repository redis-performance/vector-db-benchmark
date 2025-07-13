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
    describe: str = typer.Option(None, help="Describe available options: 'datasets' or 'engines'. When used, shows information and exits."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed information when using --describe"),
):
    """
    Example:
        python3 run.py --engines *-m-16-* --engines qdrant-* --datasets glove-*
        python3 run.py --describe datasets
        python3 run.py --describe engines --verbose
    """
    # Handle describe option first
    if describe:
        if describe.lower() == "datasets":
            describe_datasets(datasets[0] if datasets != ["*"] else "*", verbose)
            return
        elif describe.lower() == "engines":
            describe_engines(engines[0] if engines != ["*"] else "*", verbose)
            return
        else:
            typer.echo(f"Error: Unknown describe target '{describe}'. Use 'datasets' or 'engines'.", err=True)
            raise typer.Exit(1)

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


def describe_datasets(filter_pattern: str = "*", verbose: bool = False):
    """Display information about available datasets."""
    try:
        all_datasets = read_dataset_config()
    except Exception as e:
        typer.echo(f"Error reading dataset configuration: {e}", err=True)
        raise typer.Exit(1)

    # Filter datasets
    filtered_datasets = {
        name: config
        for name, config in all_datasets.items()
        if fnmatch.fnmatch(name, filter_pattern)
    }

    if not filtered_datasets:
        typer.echo(f"No datasets found matching pattern '{filter_pattern}'")
        return

    typer.echo(f"\n📊 Available Datasets ({len(filtered_datasets)} found)")
    typer.echo("=" * 80)

    # Sort datasets by dimension, then by vector count, then by name
    def get_sort_key(item):
        name, config = item
        # Get dimension (vector_size)
        dimension = config.get('vector_size', 0)
        if dimension == 'N/A':
            dimension = 0

        # Get vector count from config, fallback to 0 if None or missing
        vector_count = config.get('vector_count', 0)
        if vector_count is None:
            vector_count = 0

        return (dimension, vector_count, name.lower())

    sorted_datasets = sorted(filtered_datasets.items(), key=get_sort_key)

    if verbose:
        # Detailed view
        for name, config in sorted_datasets:
            typer.echo(f"\n🔹 {name}")
            typer.echo(f"   Vector Size: {config.get('vector_size', 'N/A')}")
            typer.echo(f"   Distance:    {config.get('distance', 'N/A')}")
            typer.echo(f"   Type:        {config.get('type', 'N/A')}")
            typer.echo(f"   Path:        {config.get('path', 'N/A')}")
            if 'link' in config:
                typer.echo(f"   Download:    {config['link']}")
            if 'schema' in config:
                typer.echo(f"   Schema:      {config['schema']}")
    else:
        # Compact table view with proper columnar formatting
        col_widths = [35, 6, 8, 12, 30, 20]  # Dataset Name, Dims, Distance, Vector Count, Description, Schema
        headers = ["Dataset Name ", "Dims ", "Distance ", "Vector Count ", "Description ", "Schema "]

        # Print headers
        header_line = ""
        for header, width in zip(headers, col_widths):
            header_line += f"{header:<{width}}"
        typer.echo(header_line)
        typer.echo("-" * sum(col_widths))

        for name, config in sorted_datasets:
            dimensions = str(config.get('vector_size', 'N/A'))
            distance = config.get('distance', 'N/A')

            # Get vector count
            vector_count = config.get('vector_count')
            if vector_count is None:
                vector_count_str = 'N/A'
            else:
                # Format large numbers with appropriate suffixes
                if vector_count >= 1000000000:
                    vector_count_str = f"{vector_count / 1000000000:.1f}B"
                elif vector_count >= 1000000:
                    vector_count_str = f"{vector_count / 1000000:.1f}M"
                elif vector_count >= 1000:
                    vector_count_str = f"{vector_count / 1000:.1f}K"
                else:
                    vector_count_str = str(vector_count)

            # Get description from config
            description = config.get('description', 'N/A')

            # Truncate description if too long
            if len(description) > col_widths[4] - 1:
                description = description[:col_widths[4] - 4] + "..."

            # Get schema information - always show field count first
            schema_info = ""
            if 'schema' in config:
                schema = config['schema']
                if isinstance(schema, dict):
                    field_count = len(schema)

                    # Always start with field count
                    if field_count == 0:
                        schema_info = "0 fields"
                    elif field_count == 1:
                        schema_info = "1 field"
                    else:
                        schema_info = f"{field_count} fields"

                    # Try to add details if they fit
                    if field_count > 0:
                        if field_count <= 2:
                            # For small schemas, try to show field names
                            field_names = ", ".join(schema.keys())
                            test_info = f"{schema_info}: {field_names}"
                            if len(test_info) <= col_widths[4] - 1:
                                schema_info = test_info
                        else:
                            # For larger schemas, try to show types
                            field_types = sorted(set(schema.values()))
                            types_str = ", ".join(field_types)
                            test_info = f"{schema_info} ({types_str})"
                            if len(test_info) <= col_widths[4] - 1:
                                schema_info = test_info
                else:
                    schema_info = str(schema)

                # Final truncation if still too long
                if len(schema_info) > col_widths[4] - 1:
                    schema_info = schema_info[:col_widths[4] - 4] + "..."

            # Truncate name if too long
            display_name = name
            if len(display_name) > col_widths[0] - 1:
                display_name = display_name[:col_widths[0] - 4] + "..."

            # Print row with proper column alignment
            row_line = ""
            values = [display_name, dimensions, distance, vector_count_str, description, schema_info]
            for value, width in zip(values, col_widths):
                row_line += f"{value:<{width}}"
            typer.echo(row_line)

    typer.echo(f"\nTotal: {len(filtered_datasets)} datasets")
    if filter_pattern != "*":
        typer.echo(f"Filter: '{filter_pattern}'")
    typer.echo("\nUse --verbose for detailed information")


def describe_engines(filter_pattern: str = "*", verbose: bool = False):
    """Display information about available engines."""
    try:
        all_engines = read_engine_configs()
    except Exception as e:
        typer.echo(f"Error reading engine configuration: {e}", err=True)
        raise typer.Exit(1)

    # Filter engines
    filtered_engines = {
        name: config
        for name, config in all_engines.items()
        if fnmatch.fnmatch(name, filter_pattern)
    }

    if not filtered_engines:
        typer.echo(f"No engines found matching pattern '{filter_pattern}'")
        return

    typer.echo(f"\n🚀 Available Engines ({len(filtered_engines)} found)")
    typer.echo("=" * 80)

    if verbose:
        # Detailed view
        for name, config in sorted(filtered_engines.items()):
            typer.echo(f"\n🔹 {name}")
            typer.echo(f"   Engine:      {config.get('engine', 'N/A')}")
            typer.echo(f"   Module:      {config.get('module', 'N/A')}")
            if 'docker' in config:
                typer.echo(f"   Docker:      {config['docker']}")
            if 'search_params' in config:
                search_params = config['search_params']
                typer.echo(f"   Search Params:")
                for param, values in search_params.items():
                    if isinstance(values, list):
                        typer.echo(f"     {param}: {values}")
                    else:
                        typer.echo(f"     {param}: {values}")
            if 'upload_params' in config:
                upload_params = config['upload_params']
                typer.echo(f"   Upload Params:")
                for param, value in upload_params.items():
                    typer.echo(f"     {param}: {value}")
    else:
        # Compact table view
        typer.echo(f"{'Engine Name':<40} {'Engine Type':<15} {'Module':<25}")
        typer.echo("-" * 80)
        for name, config in sorted(filtered_engines.items()):
            engine_type = config.get('engine', 'N/A')
            module = config.get('module', 'N/A')
            typer.echo(f"{name:<40} {engine_type:<15} {module:<25}")

    typer.echo(f"\nTotal: {len(filtered_engines)} engines")
    if filter_pattern != "*":
        typer.echo(f"Filter: '{filter_pattern}'")
    typer.echo("\nUse --verbose for detailed information")


if __name__ == "__main__":
    app()
