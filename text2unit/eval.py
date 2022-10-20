from dataset import from_path
from model import TransformerEncoder, TransformerDecoder

from utils.text import symbols

from tqdm import tqdm, trange

import torch

import numpy as np

from argparse import ArgumentParser

import os

from train import cal_acc, decode_transformer_model


if __name__ == '__main__':
    
    exp_dir = "/scratch/bbmx/junkaiwu/text2unit_transformer/dmodel_512_nheads_8_layers_6_batch_64_clip_1.0_factor_1.0_warmup_4000"
    
    train_txt_path = "/u/junkaiwu/ECE537_Project/datasets/LJSpeech/hubert100/train_t.txt"
    val_txt_path = "/u/junkaiwu/ECE537_Project/datasets/LJSpeech/hubert100/val_t.txt"
    
    train_dataloader = from_path(train_txt_path, 16, "train", 1, is_distributed=False)
    val_dataloader = from_path(val_txt_path, 16, "val", 1, is_distributed=False)
    
    device = torch.device('cuda') # only single gpu
    src_vocab_size = len(symbols)
    
    encoder = TransformerEncoder(src_vocab_size=src_vocab_size, 
                                     embedding_dim=512, 
                                     num_heads=8, 
                                     num_layers=6, 
                                     dim_feedforward=2048, 
                                     max_len_src=500,
                                     dropout_rate=0.1,
                                     embedding_factor=1.0)

    decoder = TransformerDecoder(trg_vocab_size=103,
                                 embedding_dim=512,
                                 num_heads=8,
                                 num_layers=6,
                                 dim_feedforward=2048,
                                 max_len_trg=1000,
                                 dropout_rate=0.1,
                                 embedding_factor=1.0)
    
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    
    for epoch in [261, 192, 129, 66, 33]:
        print(f"Evaluating {epoch}")
        encoder_path = os.path.join(exp_dir, f"encoder_{epoch}.pt")
        decoder_path = os.path.join(exp_dir, f"decoder_{epoch}.pt")
        
        assert os.path.isfile(encoder_path)
        assert os.path.isfile(decoder_path)
        
        encoder.load_state_dict(torch.load(encoder_path))
        decoder.load_state_dict(torch.load(decoder_path))

        # training
        encoder.eval()
        decoder.eval()
        # encoder.train()
        # decoder.train()
        
        for i, data in enumerate(val_dataloader):
            src = data["text"]
            trg = data["unit"]
            src = src.to(device).transpose(0,1) # [max_src_length, batch_size]
            trg = trg.to(device).transpose(0,1) # [max_trg_length, batch_size]
            
            curr_output, curr_predictions = decode_transformer_model(encoder, decoder, src, 800, 103, device)
            
            #print(trg.shape)
            #print(curr_output.shape)
            #print(curr_predictions.shape)
                
            curr_output = curr_output[:, :trg.shape[0]].transpose(0,1)
            
            acc = cal_acc(curr_output[1:], trg[1:], return_arr=True)
            
            print(acc)
            print(torch.mean(acc))
            
            if i == 20:
                break

