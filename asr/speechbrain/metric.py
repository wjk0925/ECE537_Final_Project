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
    parser.add_argument('--ljspeech_test', action="store_true")
    args = parser.parse_args()
    
    ljspeech_test_exclude = ["LJ012-0199.wav", "LJ017-0277.wav", "LJ034-0003.wav",
                             "LJ038-0304.wav", "LJ050-0122.wav", "LJ050-0158.wav",
                             "LJ050-0159.wav", "LJ050-0161.wav", "LJ050-0160.wav",
                             "LJ050-0199.wav", "LJ050-0203.wav", "LJ050-0207.wav",
                             "LJ050-0215.wav", "LJ050-0227.wav", "LJ050-0277.wav",]
    
    wer_metric = torchmetrics.WordErrorRate()
    cer_metric = torchmetrics.CharErrorRate()
    
    with open(args.preds_path) as f:
        preds_dict = json.load(f)
        
    with open(args.targets_path) as f:
        targets_dict = json.load(f)
        
    preds = []
    targets = []
    
    metric_dict = {}
        
    for key in targets_dict:
        if args.ljspeech_test and key in ljspeech_test_exclude:
            continue
        preds.append(preds_dict[key])
        targets.append(targets_dict[key])
        
        key_wer = wer_metric([preds_dict[key]], [targets_dict[key]])
        key_cer = cer_metric([preds_dict[key]], [targets_dict[key]])
        
        metric_dict[key] = {"wer": float(key_wer), "cer": float(key_cer)}
        
    wer = wer_metric(preds, targets)
    cer = cer_metric(preds, targets)
    
    metric_dict["wer"] = float(wer)
    metric_dict["cer"] = float(cer) 
    
    # print(metric_dict)
    
    if args.ljspeech_test:
        metric_path = args.preds_path.replace(".json", f"_ljspeech_metric.json")
    else:
        metric_path = args.preds_path.replace(".json", f"_metric.json")
            
    with open(metric_path, "w") as f:
        json.dump(metric_dict, f)
        
        
    
        
    
        
       
    
    