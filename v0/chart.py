import json
import os

import matplotlib.pyplot as plt
import argparse


x_metrics = {"mean_precisions": {"human_label": "Precision"}}
y_metrics = {
    "rps": {"mode": "higher-better", "human_label": "Search Queries per second"},
    "mean_time": {
        "mode": "lower-better",
        "human_label": "Search avg. latency including RTT (seconds)",
    },
    "p50_time": {
        "mode": "lower-better",
        "human_label": "Search p50 latency including RTT (seconds)",
    },
    "p99_time": {
        "mode": "lower-better",
        "human_label": "Search p99 latency including RTT (seconds)",
    },
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="glove-100-angular")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument(
        "-x",
        "--x-axis",
        help="Which metric to use on the X-axis",
        choices=x_metrics.keys(),
        default="mean_precisions",
    )
    parser.add_argument(
        "--x-axis-left",
        default=None,
    )
    parser.add_argument(
        "--x-axis-right",
        default=None,
    )
    parser.add_argument(
        "--x-axis-label",
        default=None,
    )

    parser.add_argument(
        "-y",
        "--y-axis",
        help="Which metric to use on the Y-axis",
        choices=y_metrics.keys(),
        default="rps",
    )
    parser.add_argument(
        "--y-axis-label",
        default=None,
    )

    parser.add_argument(
        "--y-axis-bottom",
        default=None,
    )
    parser.add_argument(
        "--y-axis-top",
        default=None,
    )

    parser.add_argument(
        "--legend",
        default="Redis",
    )

    parser.add_argument(
        "--results", type=str, help="results folder to process", default="results"
    )
    parser.add_argument(
        "--clients", type=int, help="consider results from this client count", default=1
    )
    args = parser.parse_args()
    final_results_map = {}
    x_axis = []
    y_axis = []
    fig, ax = plt.subplots()

    if os.path.exists(args.results):
        print(f"working on dir: {args.results}")
        print("reading first the upload data")
        for filename in os.listdir(args.results):
            f = os.path.join(args.results, filename)
            setup_name = filename.split(args.dataset)[0]
            setup_name = setup_name[0 : len(setup_name) - 1]

            with open(f, "r") as fd:
                try:
                    json_res = json.load(fd)
                except json.decoder.JSONDecodeError as e:
                    error_str = e.__str__()
                    print(
                        f"skipping {filename} given here as an error while processing the file {error_str})"
                    )
                    continue
                parallel = 1
                if "parallel" in json_res["params"]:
                    parallel = json_res["params"]["parallel"]

                if args.clients != parallel:
                    print(
                        f"skipping {filename} given the client count ({parallel}) is different than the one we wish to plot ({args.clients})"
                    )
                    continue
                # query
                if (
                    args.x_axis in json_res["results"]
                    and args.y_axis in json_res["results"]
                ):
                    x_val = json_res["results"][args.x_axis]
                    y_val = json_res["results"][args.y_axis]
                    x_axis.append(x_val)
                    y_axis.append(y_val)

    color = "tab:red"
    ax.scatter(
        x_axis, y_axis, c=color, label=args.legend, marker="^", edgecolors="none"
    )

    ax.legend()
    ax.grid(True)

    x_axis_label = args.x_axis
    if args.x_axis in x_metrics:
        if "human_label" in x_metrics[args.x_axis]:
            x_axis_label = x_metrics[args.x_axis]["human_label"]
    if args.x_axis_label is not None:
        x_axis_label = args.x_axis_label
    plt.xlabel(x_axis_label)

    y_axis_label = args.y_axis
    y_axis_mode = "higher-better"
    if args.y_axis in y_metrics:
        if "human_label" in y_metrics[args.y_axis]:
            y_axis_label = y_metrics[args.y_axis]["human_label"]
        if "mode" in y_metrics[args.y_axis]:
            y_axis_mode = y_metrics[args.y_axis]["mode"]
    if args.y_axis_label is not None:
        y_axis_label = args.y_axis_label
    plt.ylabel(y_axis_label)
    title_string = (
        f"{x_axis_label} vs {y_axis_label} ({y_axis_mode}).\nclients={args.clients}"
    )
    plt.title(title_string)

    x_axis_left, x_axis_right = plt.xlim()
    _, y_axis_top = plt.ylim()
    y_axis_bottom = 0
    if args.y_axis_bottom is not None:
        y_axis_bottom = float(args.y_axis_bottom)
    if args.y_axis_top is not None:
        y_axis_top = float(args.y_axis_top)

    plt.ylim(y_axis_bottom, y_axis_top)

    if args.x_axis_left is not None:
        x_axis_left = float(args.x_axis_left)
    if args.x_axis_right is not None:
        x_axis_right = float(args.x_axis_right)

    plt.xlim(x_axis_left, x_axis_right)

    output_file = f"{args.y_axis}.png"

    if args.output is not None:
        output_file = args.output

    print(f"writing output to {output_file}")

    plt.savefig(output_file)
