import os
from os.path import join, isdir, isfile, basename
import glob
import time
import json
import soundfile as sf
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from argparse import ArgumentParser
from tqdm import tqdm, trange

from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN

def main(args):
    transcriptions = []
    audios = []
    with open(args.txt_path, "r") as f:
        for line in f:
            utterance_dict = eval(line.replace("\n", ""))
            transcriptions.append(utterance_dict["transcription"])
            audios.append(utterance_dict["audio"])
                
    tacotron2 = Tacotron2.from_hparams(source=args.tacotron_source, savedir="tmpdir_tts", run_opts={"device":"cuda"})
    hifi_gan = HIFIGAN.from_hparams(source=args.hifigan_source, savedir="tmpdir_vocoder", run_opts={"device":"cuda"})
        
    start = time.time()
    
    if args.batch_infer:
        pass
    else:
        for i in trange(len(audios)):
            audio = basename(audios[i])
            transcription = transcriptions[i]
            
            mel_output, mel_length, alignment = tacotron2.encode_text(transcription)
            waveforms = hifi_gan.decode_batch(mel_output)
            
            save_path = f"{args.output_dir}/{audio}"
            sf.write(save_path, waveforms.cpu().numpy().reshape(-1), 22050)
            
    end = time.time()
    
    print(f"Taking {end - start} seconds to sythesize all audios.")
            
            
    
if __name__ == '__main__':
    
    parser = ArgumentParser(description='TTS with SpeechBrain Tacotron + HiFi-GAN')
    
    parser.add_argument('--txt_path', required=True, help='txt path to dataset')
    parser.add_argument('--batch_infer', action="store_true")
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', required=True)
    
    # for loading tts model
    parser.add_argument('--tacotron_source', default="/u/junkaiwu/speechbrain/tts-tacotron2-ljspeech")
    parser.add_argument('--hifigan_source', default="/u/junkaiwu/speechbrain/tts-hifigan-ljspeech")
    
    main(parser.parse_args())
    
    
    


