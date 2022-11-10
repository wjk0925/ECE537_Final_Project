import soundfile as sf
import numpy as np
import glob
import os
from os.path import join, isfile, isdir

if __name__ == '__main__':
    
    if isdir("/projects/bbmx/junkaiwu/LJSpeech-1.1/wavs_16khz"):
        lj_dir = "/projects/bbmx/junkaiwu/LJSpeech-1.1/wavs_16khz"
        
    else:
        lj_dir = "/nobackup/users/junkaiwu/data/LJSpeech-1.1/wavs_16khz"
        
    all_audios = glob.glob(f"{lj_dir}/*.wav")
    
    all_audios.sort()
    np.random.seed(2022)
    np.random.shuffle(all_audios)
    
    for p in all_audios[:50]:
        audio, sr = sf.read(p)
        
        print(p)
        print(len(audio))
        
        print(np.sum(abs(audio)))
        
        print(np.sum(abs(audio[:16000])))
        
        print(np.sum(abs(audio[16000:])))
        
        print(np.sum(abs(audio[-16000:])))
        
        print(np.sum(abs(audio[:-16000])))
        
        print(audio[12:16])
        
        print(audio[145:150])
        
        print(audio[600:605])
        
    