#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import argparse
import glob
import os
import random

import soundfile
import json


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", metavar="DIR", help="root directory containing flac files to index"
    )

    parser.add_argument(
        "--ext", default="wav", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest",
    )
    parser.add_argument("--transcription_path", required=True, help="path to json file")
    parser.add_argument("--dict_path", required=True, help="path to dictionary")
    return parser


def main(args):
    
    args.dest = os.path.join(args.root, "manifest")
    
    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    with open(args.transcription_path) as f:
        transcription_dict = json.load(f)

    vocab = []
    with open(args.dict_path, "r") as f:
        for line in f:
            vocab.append(line[0])

    print(vocab)

    dir_path = os.path.realpath(args.root)
    search_path = os.path.join(dir_path, "*." + args.ext)
    
    test_ltr = open(os.path.join(args.dest, "test.ltr"), "w")

    with open(os.path.join(args.dest, "test.tsv"), "w") as test_f:
        print(dir_path, file=test_f)

        for fname in glob.iglob(search_path, recursive=True):
            file_path = os.path.realpath(fname)

            if args.path_must_contain and args.path_must_contain not in file_path:
                continue

            frames = soundfile.info(fname).frames
            dest = test_f
            print(
                "{}\t{}".format(os.path.relpath(file_path, dir_path), frames), file=dest
            )
            
            audio_name = os.path.basename(file_path).strip(".wav")
            text = transcription_dict[audio_name]["char"]
            processed_text = ""
            for c in text:
                if c == " ":
                    if processed_text[-1] != " ":
                        processed_text += c
                elif c in vocab:
                    processed_text += c
                elif c.lower() in vocab:
                    processed_text += c.lower()
                elif c.upper() in vocab:
                    processed_text += c.upper()
                elif c in "âà":
                    processed_text += "a"
                elif c in "êéè":
                    processed_text += "e"
                elif c == "ü":
                    processed_text += "u"
                else:
                    if processed_text[-1] != " ":
                        processed_text += " "

            assert len(processed_text) > 0, print(audio_name, text)
            if processed_text[-1] == " ":
                processed_text = processed_text[:-1]
                    
            char = processed_text.replace(" ", "|")
            char = (" ").join(char)

            test_ltr.write(char + "\n")
            
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)