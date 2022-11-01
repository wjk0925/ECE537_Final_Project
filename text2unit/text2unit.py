from dataset import from_path
from model import TransformerEncoder, TransformerDecoder

from utils.text import symbols

from tqdm import tqdm, trange

import torch

import numpy as np

from argparse import ArgumentParser

import os

from train import cal_acc, decode_transformer_model

from argparse import ArgumentParser

from utils.text import text_to_sequence

# python text2unit.py --text_path ../text2unit2speech/test1.txt --output_dir ../text2unit2speech/v1/test2


if __name__ == '__main__':
    
    parser = ArgumentParser()
        
    parser.add_argument('--text_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--vocab_size', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=297)
    
    args = parser.parse_args()
    
    exp_dir = f"/scratch/bbmx/junkaiwu/text2unit_transformer/hubert{args.vocab_size}_v2"
    
    device = torch.device('cuda') # only single gpu
    src_vocab_size = len(symbols)
    
    encoder = TransformerEncoder(src_vocab_size=src_vocab_size, 
                                     embedding_dim=512, 
                                     num_heads=8, 
                                     num_layers=6, 
                                     dim_feedforward=2048, 
                                     max_len_src=200,
                                     dropout_rate=0.1,
                                     embedding_factor=1.0)

    decoder = TransformerDecoder(trg_vocab_size=args.vocab_size+3,
                                 embedding_dim=512,
                                 num_heads=8,
                                 num_layers=6,
                                 dim_feedforward=2048,
                                 max_len_trg=512,
                                 dropout_rate=0.1,
                                 embedding_factor=1.0)
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    encoder_path = os.path.join(exp_dir, f"encoder_{args.epoch}.pt")
    decoder_path = os.path.join(exp_dir, f"decoder_{args.epoch}.pt")

    assert os.path.isfile(encoder_path)
    assert os.path.isfile(decoder_path)

    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    # training
    encoder.eval()
    decoder.eval()
    
    os.makedirs(args.output_dir)
    
    os.system(f"cp {args.text_path} {args.output_dir}/")
    
    texts = []
    with open(args.text_path, "r") as f:
        for line in f:
            texts.append(line.replace("\n", ""))
            
    w_f = open(f"{args.output_dir}/pred_units.km", "w")
    
    print(len(texts), "sentences in total")
    
    for i, text in enumerate(tqdm(texts)):
        print(text)
        text = np.array(text_to_sequence(text))
        text = np.pad(text, (0, 200 - len(text)))[:, None]
        
        
        text = torch.from_numpy(text).to(device)
        
        pred, _ = decode_transformer_model(encoder, decoder, text, 512, args.vocab_size+3, device)
        
        pred_units = ""
                
        for token in pred[0, :]:
            if token == 1:
                continue
            if token == 2:
                print("stopped!")
                break
            
            pred_units += str(int(token.item()) - 3)
            pred_units += " "
        
        pred_units = pred_units[:-1]
        
        w_f.write(str(i) + "|" + pred_units + "\n")
            
    w_f.close()