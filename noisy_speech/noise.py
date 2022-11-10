import os
from os.path import join, isdir, isfile, basename
import glob
import soundfile as sf
import numpy as np
from argparse import ArgumentParser
from tqdm import tqdm

def dB2scale(dB_target, signal, noise):
    assert signal.shape == noise.shape
    
    cur_r = np.sum(signal**2) / np.sum(noise**2)
    tar_r = 10**(dB_target/10)

    return np.sqrt((tar_r / cur_r))

def main(args):
    os.makedirs(args.output_dir)
    snr_range = args.snr_range
    assert len(snr_range) == 2
    assert snr_range[0] < snr_range[1]
        
    speech_paths = glob.glob(f"{args.speech_dir}/*{args.speech_ext}")
    speech_paths.sort()
    noise_paths = glob.glob(f"{args.noise_dir}/*{args.noise_ext}")
    noise_paths.sort()
    
    print(noise_paths)
    
    np.random.seed(args.random_seed)
    
    for s_p in tqdm(speech_paths):
        n_p = np.random.choice(noise_paths)
        
        speech, s_sr = sf.read(s_p)
        noise, n_sr = sf.read(n_p)
        assert s_sr == n_sr
        
        # assuming each noise audio is longer than speech
        shift = np.random.randint(0, len(noise) - len(speech))
        noise = noise[shift:shift + len(speech)]
        
        snr = np.random.uniform(snr_range[0], snr_range[1])
        r = dB2scale(snr, speech, noise)
        
        mixture = speech + noise/r
        
        mixture_basename = basename(s_p)
        
        sf.write(f"{args.output_dir}/{mixture_basename}", mixture, s_sr)
        
if __name__ == '__main__':
    
    parser = ArgumentParser(description='Adding Noise to a Dataset!')
    
    parser.add_argument('--speech_dir', required=True, help='audio files to transcribe')
    parser.add_argument('--speech_ext', default="wav")
    parser.add_argument('--noise_dir', required=True, help='audio files to transcribe')
    parser.add_argument('--noise_ext', default="wav")
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--snr_range', type=float, nargs='+')
    parser.add_argument('--output_dir', required=True)
    
    main(parser.parse_args())