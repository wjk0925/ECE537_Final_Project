from dataset import from_path_v2
from model import TransformerEncoder, TransformerDecoder

from utils.text import symbols

from tqdm import tqdm, trange

import torch

import numpy as np

from argparse import ArgumentParser

import os

from train import cal_acc, decode_transformer_model

from argparse import ArgumentParser


if __name__ == '__main__':
    
    parser = ArgumentParser()
    
    parser.add_argument('--vocab_size', type=int, default=100)
    parser.add_argument('--exp_dir', required=True)
    parser.add_argument('--test_txt_path', required=True)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=295)
    parser.add_argument('--output_name', default=None)
    
    
    args = parser.parse_args()
    
    #exp_dir = f"/scratch/bbmx/junkaiwu/text2unit_transformer/hubert{args.vocab_size}_v2"
    
    #test_txt_path = f"/u/junkaiwu/ECE537_Project/datasets/LJSpeech/hubert100/test{args.vocab_size}.txt"
    
    exp_dir = args.exp_dir
    test_txt_path = args.test_txt_path
    
    test_dataloader = from_path_v2(test_txt_path, args.batch_size, split=args.split, max_in_len=200, min_in_len=10, max_out_len=512, num_workers=args.num_workers, is_distributed=False)
    
    
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
    
    print(f"Evaluating {args.epoch}")
    encoder_path = os.path.join(exp_dir, f"encoder_{args.epoch}.pt")
    decoder_path = os.path.join(exp_dir, f"decoder_{args.epoch}.pt")

    assert os.path.isfile(encoder_path)
    assert os.path.isfile(decoder_path)

    encoder.load_state_dict(torch.load(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path))

    # training
    encoder.eval()
    decoder.eval()
    # encoder.train()
    # decoder.train()

    from torchmetrics import WordErrorRate
    metric = WordErrorRate()

    test_preds = []
    test_targets = []

    print("start evaluation")
    print(len(test_dataloader))


    for i, data in enumerate(tqdm(test_dataloader)):
        src = data["text"]
        trg = data["unit"]
        src = src.to(device).transpose(0,1) # [max_src_length, batch_size]
        trg = trg.to(device).transpose(0,1) # [max_trg_length, batch_size]

        curr_output, curr_predictions = decode_transformer_model(encoder, decoder, src, 512, args.vocab_size+3, device)
        curr_output = curr_output.transpose(0,1)
        
        for b_i in range(curr_output.shape[1]):
            pred_str = ""
            target_str = ""

            for token in curr_output[:, b_i]:
                if token == 1:
                    continue
                if token == 2:
                    break
                pred_str += str(int(token.item()) - 3)
                pred_str += " "

            for token in trg[:, b_i]:
                if token == 1:
                    continue
                if token == 2:
                    break
                target_str += str(int(token.item()) - 3)
                target_str += " "


            pred_str = pred_str[:-1]
            target_str = target_str[:-1]

            test_preds.append(pred_str)
            test_targets.append(target_str)

    assert len(test_preds) == len(test_targets)

    test_err_rate = metric(test_preds, test_targets)

    print(f"Test Error Rate is {test_err_rate}")

    if args.output_name is None:
        p_f = open(f"{exp_dir}/{args.epoch}_preds.km", "w")
    else:
        p_f = open(args.output_name, "w")
    #t_f = open(f"{exp_dir}/{args.epoch}_targets.km", "w")

    for i in range(len(test_preds)):
        p_f.write(str(i) + "|" + test_preds[i] + "\n")
        #t_f.write(str(i) + "|" +  test_targets[i] + "\n")

    p_f.write(f"Test Error Rate is {test_err_rate}\n")

    p_f.close()
    #t_f.close()
                
                
        
        

