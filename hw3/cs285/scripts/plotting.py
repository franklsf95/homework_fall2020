#!/usr/bin/env python3

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="cs285/data")
parser.add_argument("--input_regex", "-i", required=True, help="regex to filter runs")
parser.add_argument(
    "--label_regex",
    default="(.*)_[A-Z]",
    help="regex to extract run labels, must contain exactly one group",
)
parser.add_argument("--avg_series", default="Eval_AverageReturn")
parser.add_argument("--std_series", default="Eval_StdReturn")
parser.add_argument("--markevery", default=10)
parser.add_argument("--xlabel", default="Steps")
parser.add_argument("--ylabel", default="Return")
parser.add_argument("--output_file", "-o", help="output file name")
args = parser.parse_args()


def plot_all(output_file, runs):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.grid()

    for path, run_name in runs:
        label = re.match(args.label_regex, run_name).group(1)
        plot_series(path, ax, label)

    plt.legend()
    plt.savefig(output_file, dpi=300)


def plot_series(path, ax, label):
    ea = EventAccumulator(path)
    ea.Reload()

    avg_ss = ea.Scalars(args.avg_series)
    try:
        std_ss = ea.Scalars(args.std_series)
        assert len(avg_ss) == len(std_ss), (avg_ss, std_ss)
    except KeyError:
        std_ss = None

    data = {
        "steps": [t[1] for t in avg_ss],
        "avg": [t[2] for t in avg_ss],
    }
    if std_ss is not None:
        data["std"] = [t[2] for t in std_ss]

    df = pd.DataFrame(data)
    # Add a duplicate row to preserve the last data marker
    df.loc[len(df.index)] = df.loc[len(df.index) - 1]
    ax.errorbar(
        x=df["steps"],
        y=df["avg"],
        yerr=df["std"] if "std" in df.columns else None,
        elinewidth=1,
        capsize=4,
        linewidth=2,
        marker="o",
        markersize=6,
        markevery=args.markevery,
        errorevery=args.markevery,
        label=label,
    )


def main():
    runs = []
    for subdir in os.scandir(args.data_dir):
        if not subdir.is_dir():
            continue
        if not re.match(args.input_regex, subdir.name):
            continue
        files = [
            f
            for f in os.scandir(subdir.path)
            if (not f.name.startswith(".")) and f.is_file()
        ]
        assert len(files) == 1, files
        runs.append((files[0].path, subdir.name))
    runs.sort()

    print(f"Found {len(runs)} runs:", runs)
    if len(runs) == 0:
        return

    output_file = args.output_file or args.input_regex + ".png"
    plot_all(output_file, runs)


if __name__ == "__main__":
    main()
