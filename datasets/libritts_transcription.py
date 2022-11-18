import os
import glob
from os.path import join, basename, dirname
from argparse import ArgumentParser

def main(args):
    if "test" in args.split:
        split = "test"
    elif "dev" in args.split:
        split = "val"
    else:
        split = "train"

    w_f = open(f"/home/junkaiwu/ECE537_Final_Project/datasets/LibriTTS/hubert/{split}200.txt", "a")

    for audio in glob.glob(f"{args.libritts_dir}/{args.split}/*/*/*.wav"):
        utterance_dict = {}
        utterance_dict["audio"] = basename(audio).replace(".wav", "")

        with open(audio.replace(".wav", ".normalized.txt"), "r") as f:
            for line in f:
                transcription = line.replace("\n", "")
                break
        with open(audio.replace(".wav", ".original.txt"), "r") as f:
            for line in f:
                transcription_raw = line.replace("\n", "")
                break
    
        utterance_dict["transcription"] = transcription
        utterance_dict["transcription_raw"] = transcription_raw
        utterance_dict["split"] = args.split

        w_f.write(str(utterance_dict) + "\n")

    w_f.close()



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--libritts_dir')
    parser.add_argument('--split', choices=["test-clean", "dev-clean", "train-clean-100", "train-clean-360", "train-clean-500"])
    main(parser.parse_args())