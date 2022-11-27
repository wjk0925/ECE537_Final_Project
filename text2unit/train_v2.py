from dataset import from_path_v2
from model import TransformerEncoder, TransformerDecoder

from utils.text import symbols

from tqdm import tqdm, trange

import torch

import numpy as np

from argparse import ArgumentParser

import os


# fix the max input len


def cal_acc(predictions, trg, return_arr=False):
    predictions[trg == 0] = 0
    zero_count = torch.sum((trg == 0), dim=0)
    acc = (torch.sum(predictions == trg, dim=0) - zero_count) / (predictions.shape[0] - zero_count)
    
    if return_arr:
        return acc
    else:
        return torch.mean(acc)
    

def decode_transformer_model(encoder, decoder, src, max_decode_len, trg_vocab_size, device):
    """
    Args:
        encoder: Your RnnEncoder object
        decoder: Your RnnDecoder object
        src: [max_src_length, batch_size] the source sentences you wish to translate
        max_decode_len: The maximum desired length (int) of your target translated sentences
        trg_vocab: The Vocab_Lang object for the target language
        device: the device your torch tensors are on (you may need to call x.to(device) for some of your tensors)

    Returns:
        curr_output: [batch_size, max_decode_len] containing your predicted translated sentences
        curr_predictions: [batch_size, max_decode_len, trg_vocab_size] containing the (unnormalized) probabilities of each
            token in your vocabulary at each time step

    Pseudo-code:
    - Obtain encoder output by encoding src sentences
    - For 1 ≤ t ≤ max_decode_len:
        - Obtain dec_input as the best words so far for previous time steps (you can get this from curr_output)
        - Obtain your (unnormalized) prediction probabilities by feeding dec_input and encoder output to decoder
        - Save your (unnormalized) prediction probabilities in curr_predictions at index t
        - Calculate the most likely (highest probability) token and save in curr_output at timestep t
    """
    
    batch_size = src.size(1)
    curr_output = torch.zeros((batch_size, max_decode_len))
    curr_predictions = torch.zeros((batch_size, max_decode_len, trg_vocab_size))

    # We start the decoding with the start token for each example
    curr_output[:, 0] = torch.tensor([[1]] * batch_size).transpose(0,1).squeeze(1)
    curr_output = curr_output.to(device)
    
    enc_output = encoder(src)

    # At each time step, get the best prediction and save it
    with torch.no_grad():
        ended = torch.tensor([False]*batch_size)
        for t in range(1, max_decode_len):
            output = decoder(curr_output[:,:t].transpose(0,1).int(), enc_output)
            output = output[-1]
            curr_predictions[:,t,:] = output
            output[:, 0] = -np.inf
            predictions = torch.argmax(output, dim=1)
            curr_output[:, t] = predictions
            
            ended = ended + (predictions.to(torch.device("cpu")) == 2)
            
            if torch.all(ended):
                # print("Ended Early")
                break
                
    return curr_output, curr_predictions

def main(args):
    
    if args.exp_name is None:
        exp_name = f"dmodel_{args.embedding_dim}_nheads_{args.num_heads}_layers_{args.num_layers}_batch_{args.train_batch_size}_clip_{args.grad_clip}_factor_{args.factor}_warmup_{args.warmup_steps}"
    else:
        exp_name = args.exp_name
        
    os.makedirs(f"{args.ckpt_dir}/{exp_name}", exist_ok=True)

    import wandb
    wandb.init(project="537", name=exp_name, entity=args.wandb_entity)
    wandb.config = args
    
    from torchmetrics import WordErrorRate
    metric = WordErrorRate()

    train_dataloader = from_path_v2(args.train_txt_path, args.train_batch_size, split="train", max_in_len=args.max_len_src, max_out_len=args.max_len_trg, num_workers=args.num_workers, is_distributed=False)
    val_dataloader = from_path_v2(args.val_txt_path, args.val_batch_size, split="val", max_in_len=args.max_len_src, max_out_len=args.max_len_trg, num_workers=args.num_workers, is_distributed=False) # not used for now
    # not in args
    device = torch.device('cuda') # only single gpu
    src_vocab_size = len(symbols)
    # initialize model    
    encoder = TransformerEncoder(src_vocab_size=src_vocab_size, 
                                 embedding_dim=args.embedding_dim, 
                                 num_heads=args.num_heads, 
                                 num_layers=args.num_layers, 
                                 dim_feedforward=args.dim_feedforward, 
                                 max_len_src=args.max_len_src,
                                 dropout_rate=args.input_dropout_rate,
                                 embedding_factor=args.embedding_factor)
    
    decoder = TransformerDecoder(trg_vocab_size=args.trg_vocab_size,
                                 embedding_dim=args.embedding_dim,
                                 num_heads=args.num_heads,
                                 num_layers=args.num_layers,
                                 dim_feedforward=args.dim_feedforward,
                                 max_len_trg=args.max_len_trg,
                                 dropout_rate=args.input_dropout_rate,
                                 embedding_factor=args.embedding_factor)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # training
    encoder.train()
    decoder.train()

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing) # 0 for pad
    transformer_model_params = list(encoder.parameters()) + list(decoder.parameters())
    
    # noam optimizer in transformer paper
    peak_lr = args.embedding_dim**(-0.5) * args.warmup_steps**(-0.5)
    noam = lambda step: (step / args.warmup_steps) * peak_lr if step < args.warmup_steps else args.embedding_dim**(-0.5) * min(step**(-0.5), step * args.warmup_steps**(-1.5))
    optimizer = torch.optim.Adam(transformer_model_params, betas=args.betas, eps=args.eps, lr=args.factor)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, noam)
    
    epoch_bar = trange(args.epochs, desc='Epoch Loss: 0.0 / Epoch Acc: 0.0', leave=True)
    epoch_accs = []
    
    for epoch in epoch_bar:
        
        encoder.train()
        decoder.train()

        losses = []
        accs = []

        step_bar = tqdm(train_dataloader, desc='Step Loss: 0.0 / Step Acc: 0.0', leave=True)

        for data in step_bar:
            # src for text, trg for unit
            src = data["text"]
            trg = data["unit"]
            src = src.to(device).transpose(0,1) # [max_src_length, batch_size]
            trg = trg.to(device).transpose(0,1) # [max_trg_length, batch_size]
            # transformer forward
            enc_out = encoder(src)
            output = decoder(trg[:-1, :], enc_out)

            output_flatten = output.reshape(-1, output.shape[2])
            trg_flatten = trg[1:].reshape(-1)
            
            optimizer.zero_grad()

            loss = criterion(output_flatten, trg_flatten)
            losses.append(loss.item())
            wandb.log({"step_loss": loss.item()})

            loss.backward()
            
            # Clip to avoid exploding grading issues
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=args.grad_clip)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=args.grad_clip)

            optimizer.step()
            scheduler.step()

            output[:,:, 0] = -float("Inf")
            predictions = torch.argmax(output, dim=2)
            acc = cal_acc(predictions, trg[1:])
            accs.append(acc.item())
            wandb.log({"step_acc": acc.item()})

            step_bar.set_description(f'Step Loss: {loss} / Step Acc: {acc}')

        wandb.log({"epoch_loss": np.mean(losses)})
        wandb.log({"epoch_acc": np.mean(accs)})

        epoch_accs.append(np.mean(accs))


        epoch_bar.set_description(f'Epoch Loss: {np.mean(losses)} / Epoch Acc: {np.mean(accs)}')
        
        val_preds = []
        val_targets = []
        
        if (epoch % 5) == 0:
            encoder.eval()
            decoder.eval()
            
            for i, data in enumerate(tqdm(val_dataloader)):
                src = data["text"]
                trg = data["unit"]
                src = src.to(device).transpose(0,1) # [max_src_length, batch_size]
                trg = trg.to(device).transpose(0,1) # [max_trg_length, batch_size]

                curr_output, curr_predictions = decode_transformer_model(encoder, decoder, src, args.max_len_trg, args.trg_vocab_size, device)

                curr_output = curr_output.transpose(0,1)


                for b_i in range(curr_output.shape[1]):
                    pred_str = ""
                    target_str = ""

                    for token in curr_output[:, b_i]:
                        if token == 1:
                            continue
                        if token == 2:
                            break
                        pred_str += str(int(token.item()))
                        pred_str += " "

                    for token in trg[:, b_i]:
                        if token == 1:
                            continue
                        if token == 2:
                            break
                        target_str += str(int(token.item()))
                        target_str += " "


                    pred_str = pred_str[:-1]
                    target_str = target_str[:-1]

                    val_preds.append(pred_str)
                    val_targets.append(target_str)
                    
            assert len(val_preds) == len(val_targets)

            val_err_rate = metric(val_preds, val_targets)
            
            wandb.log({"val_ter": val_err_rate})

            torch.save(encoder.state_dict(), f"/nobackup/users/junkaiwu/outputs/text2unit_transformer/{exp_name}/encoder_{epoch}.pt")
            torch.save(decoder.state_dict(), f"/nobackup/users/junkaiwu/outputs/text2unit_transformer/{exp_name}/decoder_{epoch}.pt")
    
    

if __name__ == '__main__':
    
    # args and hyperparameters
    parser = ArgumentParser(description='Training Transformer for TEXT2UNIT')
    
    # datasets
    parser.add_argument('--train_txt_path', 
                        default="/u/junkaiwu/ECE537_Project/datasets/LJSpeech/hubert100/train_t.txt",
                        help='txt file containing text and unit pairs')
    parser.add_argument('--val_txt_path',
                        default="/u/junkaiwu/ECE537_Project/datasets/LJSpeech/hubert100/val_t.txt",
                        help='txt file containing text and unit pairs')
    parser.add_argument('--trg_vocab_size', type=int, default=103, 
                        help='num of units + <start> <end> <pad>')
    
    
    # transformer
    parser.add_argument('--embedding_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--input_dropout_rate', type=float, default=0.1)
    parser.add_argument('--layer_dropout_rate', type=float, default=0.1)
    parser.add_argument('--max_len_src', type=int, default=200, help='only used for training')
    parser.add_argument('--max_len_trg', type=int, default=512, help="only used for training")
    parser.add_argument('--embedding_factor', type=float, default=1, help="described in transformer paper")
    
    # training
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--val_batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--grad_clip', type=float, default=1)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    
    # optimizer
    parser.add_argument('--factor', type=float, default=1)
    parser.add_argument('--warmup_steps', type=int, default=4000)
    parser.add_argument('--betas', type=float, nargs='+', default=[0.9, 0.98])
    parser.add_argument('--eps', type=float, default=1e-09)
    
    # ckpt & wandb
    parser.add_argument('--ckpt_dir', default="/scratch/bbmx/junkaiwu/text2unit_transformer")
    parser.add_argument('--exp_name', default=None)
    parser.add_argument('--wandb_entity', default="wujunkai")
    
    main(parser.parse_args())
            
        
            
        
        

    
    

    
    
    
    

    

    
    
    