#!/bin/bash
#SBATCH -J train_text2unit_hubert200_v2_libritts
#SBATCH -o train_text2unit_hubert200_v2_libritts_%j.%N.out
#SBATCH -e train_text2unit_hubert200_v2_libritts_%j.%N.err
#SBATCH --mail-user=junkaiwu@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --cpus-per-task=16
#SBATCH --exclude=node0019
#SBATCH --mem=0

## User python environment
PYTHON_VIRTUAL_ENVIRONMENT=fairseq3
CONDA_ROOT=/nobackup/users/junkaiwu/anaconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

dim_feedforward=4096
embedding_dim=1024
num_heads=8
num_layers=6

train_batch_size=32
val_batch_size=8
factor=1
warmup_steps=8000

grad_clip=1

text2unit_dir="/home/junkaiwu/ECE537_Final_Project/text2unit"

train_txt_path="/home/junkaiwu/ECE537_Final_Project/datasets/LibriTTS_train-clean-100/hubert/train200.txt"
val_txt_path="/home/junkaiwu/ECE537_Final_Project/datasets/LibriTTS_train-clean-100/hubert/val200.txt"
trg_vocab_size=203
exp_name="hubert200_libritts_large_batch${train_batch_size}_warm${warmup_steps}"

srun --gres=gpu:1 --ntasks=1  python ${text2unit_dir}/train_v2.py \
    --dim_feedforward ${dim_feedforward} \
    --embedding_dim ${embedding_dim}  \
    --num_workers 16 \
    --num_heads ${num_heads} \
    --num_layers ${num_layers} \
    --train_batch_size ${train_batch_size} \
    --val_batch_size ${val_batch_size} \
    --factor ${factor} \
    --warmup_steps ${warmup_steps} \
    --grad_clip ${grad_clip} \
    --epochs 300 \
    --train_txt_path ${train_txt_path} \
    --val_txt_path ${val_txt_path} \
    --trg_vocab_size ${trg_vocab_size} \
    --exp_name ${exp_name}
    



