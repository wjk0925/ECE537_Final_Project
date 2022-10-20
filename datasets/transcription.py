import os
from os.path import join, basename, dirname
from argparse import ArgumentParser

def main(args):
    lj_dict = {}
    with open(join(args.lj_dir, "metadata.csv"), "r") as f:
        for i, line in enumerate(f):
            [filename, transcription, _] = line.strip("\n").split("|")
            lj_dict[filename] = transcription

    for split in ["train", "val", "test"]:
        w_f = open(join(args.txt_datasets_dir, split +"_t.txt"), "w")
        with open(join(args.txt_datasets_dir, split+".txt"), "r") as f:
            for line in f:
                line_dict = eval(line.strip("\n"))
                line_dict['transcription'] = lj_dict[basename(line_dict['audio']).strip('.wav')]
                w_line = str(line_dict) + "\n"
                w_f.write(w_line)
        w_f.close()

if __name__ == '__main__':
    parser = ArgumentParser(description='add transcriptions to speech resynthesis txt datasets')
    parser.add_argument('--lj_dir', help='directory for LJSpeech')
    parser.add_argument('--txt_datasets_dir', help="directory containing txt datasets")
    main(parser.parse_args())