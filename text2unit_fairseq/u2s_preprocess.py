import os
from os.path import dirname, basename
import numpy as np
import json
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--unit_outputs_path", required=True)
    parser.add_argument("--name2text_path", default="/home/junkaiwu/ECE537_Final_Project/datasets/LJSpeech/ljspeech.json")
    args = parser.parse_args()

    with open(args.name2text_path) as f:
        name2text = json.load(f)
    text2name = {name2text[key]["char"]:key for key in name2text}

    preds = []
    names = []

    w_f = open(args.unit_outputs_path.replace(".txt", "_u2s.txt"), "w")

    with open(args.unit_outputs_path, "r") as f:
        for i, line in enumerate(f):
            if line[0] == "S":
                text = line.split("\t")[1].strip("\n").replace(" ", "").replace("|", " ")
                name = text2name[text]
                
                names.append(name)

            if line[0] == "H":
                pred = line.split("\t")[2].strip("\n")
                preds.append(pred)

    assert len(preds) == len(names)

    for i in range(len(preds)): 
        utter_dict = {}
        utter_dict["audio"] = names[i]
        utter_dict["hubert"] = preds[i]
        w_f.write(str(utter_dict)+"\n")
            
    w_f.close()




        