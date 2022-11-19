import os
from os.path import dirname, basename
import numpy as np
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--metric_path", required=True)
    args = parser.parse_args()

    with open(args.metric_path, "r") as f:
        for line in f:
            metrics_dict = eval(line.strip("\n"))
            break

    metrics = [float(metrics_dict[epoch]) for epoch in metrics_dict]

    best_epochs = np.argsort(metrics)[:args.num_epochs] + 1

    for i, e in enumerate(best_epochs):
        project_dir = dirname(args.metric_path)
        link_path = f"{project_dir}/checkpoint_best_{i+1}.pt"
        os.system(f"ln -s {project_dir}/checkpoint{e}.pt {link_path}")