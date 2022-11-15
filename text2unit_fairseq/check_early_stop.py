import os
from os.path import dirname, basename
import numpy as np
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--patience", type=int)
    parser.add_argument("--metric_path", required=True)
    args = parser.parse_args()

    with open(args.metric_path, "r") as f:
        for line in f:
            metrics_dict = eval(line.strip("\n"))
            break

    metrics = [float(metrics_dict[epoch]) for epoch in metrics_dict]

    epochs_since_best = len(metrics) - np.argmin(metrics) - 1

    print(f"epochs since best is {epochs_since_best}")

    if epochs_since_best >= args.patience:
        print("writing stop training file")
        write_dir = dirname(args.metric_path)
        os.system(f"touch {write_dir}/stop_training.txt")