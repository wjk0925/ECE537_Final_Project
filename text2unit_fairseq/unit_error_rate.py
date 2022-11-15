from torchmetrics import WordErrorRate
from argparse import ArgumentParser
from os.path import isfile

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--eval_path", required=True)
    parser.add_argument("--write_path", required=True, help="to write results")
    args = parser.parse_args()

    metric = WordErrorRate()

    preds = []
    targets = []

    with open(args.eval_path, "r") as f:
        for i, line in enumerate(f):
            if line[0] == "T":
                targets.append(line.split("\t")[1].strip("\n"))

            if line[0] == "H":
                preds.append(line.split("\t")[2].strip("\n"))
            
    uer = metric(preds, targets)

    if isfile(args.write_path):
        with open(args.write_path) as f:
            for line in f:
                eval_results = eval(line.strip("\n"))
                break
    else:
        eval_results = {}

    eval_results[args.epoch] = uer.item()

    with open(args.write_path, "w") as f:
        f.write(str(eval_results) + "\n")


