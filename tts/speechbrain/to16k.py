import os
from os.path import join, isdir, isfile, basename, dirname
import json
import numpy as np
import glob
import soundfile as sf
import librosa
from tqdm import tqdm
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='downsampling')
    parser.add_argument('--audio_dir', required=True)
    args = parser.parse_args()
    
    if not isdir(f"{args.audio_dir}/16k"):
        os.mkdir(f"{args.audio_dir}/16k")
        
    output_dir = f"{args.audio_dir}/16k"
    
    for audio_p in tqdm(glob.glob(f"{args.audio_dir}/*.wav")):
        output_p = output_dir + "/" + basename(audio_p)
        if isfile(output_p):
            continue
            
        audio, sr = librosa.load(audio_p, sr=16000)
        sf.write(output_p, audio, 16000)