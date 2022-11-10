import os
from os.path import isdir, join, basename
from argparse import ArgumentParser
import glob
import soundfile as sf
import librosa

def main(args):
    if not isdir(args.write_dir):
        os.mkdir(args.write_dir)
    for p in glob.glob(join(args.data_root, "*.wav")):
        # audio, sr = librosa.load(p)
        audio_16k, sr = librosa.load(p, sr=16000)
        sf.write(join(args.write_dir, basename(p)), audio_16k, 16000)
        
        
        


if __name__ == '__main__':
    parser = ArgumentParser(description='downsample sample rate to 16kHz')
    parser.add_argument('--data_root', help='root directory')
    parser.add_argument('--write_dir', help="directory to write")
    main(parser.parse_args())
