import os
from os.path import join, isdir, isfile
import json
import numpy as np
from argparse import ArgumentParser
import torchmetrics

if __name__ == '__main__':
    parser = ArgumentParser(description='Calculating WER and CNR')
    parser.add_argument('--preds_path', required=True)
    parser.add_argument('--targets_path', required=True)
    args = parser.parse_args()
    
    wer_metric = torchmetrics.WordErrorRate()
    cer_metric = torchmetrics.CharErrorRate()
    
    with open(args.preds_path) as f:
        preds_dict = json.load(f)
        
    with open(args.targets_path) as f:
        targets_dict = json.load(f)
        
    preds = []
    targets = []
    
    for key in targets_dict:
        preds.append(preds_dict[key])
        targets.append(targets_dict[key])
        
    wer = wer_metric(preds, targets)
    cer = cer_metric(preds, targets)
    
    with open(args.preds_path.replace(".json", "metric.txt"), "w") as f:
        f.write(f"WER: {wer}" + "\n")
        f.write(f"CER: {cer}" + "\n")
        
       
    
    