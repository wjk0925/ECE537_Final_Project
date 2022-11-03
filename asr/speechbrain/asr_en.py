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
from tqdm import tqdm

from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.utils.sf_torchaudio import sf_load, sf_save

def main(args):
    
    os.system("export OMP_NUM_THREADS=1")
        
    asr_model = EncoderDecoderASR.from_hparams(source=args.source, savedir=args.savedir, run_opts={"device":"cuda"})
    
    print("Model Loaded")
    
    audio_files = glob.glob(f"{args.data_dir}/*.{args.ext}")
    
    print(f"Transcribing {len(audio_files)} audio files")
    
    start = time.time()
    
    if args.batch_infer:
        preds_dict = {}
        
        audio_sizes = [os.stat(audio_path).st_size for audio_path in audio_files]
        audio_orders = np.flip(np.argsort(audio_sizes))
        
        batches = int(np.ceil(len(audio_files) / args.batch_size))
        
        for i in tqdm(range(batches)):
            indices = audio_orders[i*args.batch_size:(i+1)*args.batch_size]
            batch_files = []
            for idx in indices:
                batch_files.append(audio_files[idx])
                sigs=[]
                lens=[]
                for file in batch_files:
                    snt, fs = sf_load(file)
                    sigs.append(snt.squeeze())
                    lens.append(snt.shape[1])

                batch = pad_sequence(sigs, batch_first=True, padding_value=0.0)

                lens = torch.Tensor(lens) / batch.shape[1]

                preds = asr_model.transcribe_batch(batch, lens)[0]
                
                for audio_file, pred in zip(batch_files, preds):
                    preds_dict[basename(audio_file)] = pred
                    
        with open(f"{args.data_dir}/preds_batch_{args.batch_size}.json", "w") as write_file:
            json.dump(preds_dict, write_file)
                   
    else:
        preds_dict = {}
        for audio_file in tqdm(audio_files):
            pred = asr_model.transcribe_file(audio_file)
            preds_dict[basename(audio_file)] = pred
            
        with open(f"{args.data_dir}/preds.json", "w") as write_file:
            json.dump(preds_dict, write_file)
            
    end = time.time()
    
    print(f"Taking {end - start} seconds to transcribe all audios.")
            
        
if __name__ == '__main__':
    
    parser = ArgumentParser(description='ASR with SpeechBrain Transformer')
    
    parser.add_argument('--data_dir', required=True, help='audio files to transcribe')
    parser.add_argument('--ext', default="wav")
    parser.add_argument('--batch_infer', action="store_true")
    parser.add_argument('--batch_size', type=int, default=16)
    
    # for loading asr model
    parser.add_argument('--source', default="/home/junkaiwu/speechbrain/asr-transformer-transformerlm-librispeech")
    parser.add_argument('--savedir', default= "/home/junkaiwu/speechbrain/asr-transformer-transformerlm-librispeech")
    
    main(parser.parse_args())
    
    
    

