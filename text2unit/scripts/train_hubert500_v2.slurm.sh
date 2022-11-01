#!/bin/bash
#SBATCH --mem=20g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bbmx-delta-gpu
#SBATCH --job-name=train_text2unit_hubert500_v2
#SBATCH --output="train_text2unit_hubert500_v2_%j.out"
#SBATCH --error="train_text2unit_hubert500_v2_%j.err"
#SBATCH --time=24:00:00
#SBATCH --constraint="scratch"
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --no-requeue
#SBATCH --mail-user=junkai2@illinois.edu
#SBATCH --mail-type="BEGIN,END" 

echo "job is starting on `hostname`"

export OMP_NUM_THREADS=1  # if code is not multithreaded, otherwise set to 8 or 16

source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate diffusion

embedding_dim=512
num_heads=8
num_layers=6

train_batch_size=64
factor=1
warmup_steps=4000

grad_clip=1

text2unit_dir="/u/junkaiwu/ECE537_Project/text2unit"

train_txt_path="/u/junkaiwu/ECE537_Project/datasets/LJSpeech/hubert100/train500.txt"
val_txt_path="/u/junkaiwu/ECE537_Project/datasets/LJSpeech/hubert100/val500.txt"
trg_vocab_size=503
exp_name="hubert500_v2"

srun --gres=gpu:1 --ntasks=1  python ${text2unit_dir}/train_v2.py \
    --embedding_dim ${embedding_dim}  \
    --num_workers 16 \
    --num_heads ${num_heads} \
    --num_layers ${num_layers} \
    --train_batch_size ${train_batch_size} \
    --factor ${factor} \
    --warmup_steps ${warmup_steps} \
    --grad_clip ${grad_clip} \
    --epochs 300 \
    --train_txt_path ${train_txt_path} \
    --val_txt_path ${val_txt_path} \
    --trg_vocab_size ${trg_vocab_size} \
    --exp_name ${exp_name}
    



