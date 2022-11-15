#!/bin/bash
#SBATCH -J t2u_transformer
#SBATCH -o t2u_transformer_%j.%N.out
#SBATCH -e t2u_transformer_%j.%N.err
#SBATCH --mail-user=junkaiwu@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --qos=sched_level_2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --exclusive

source /nobackup/users/heting/espnet/tools/conda/bin/../etc/profile.d/conda.sh
conda activate babblenet # change it to your conda environment

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

# fixed
arch="transformer_wmt_en_de"
lr=5e-4
warmup_updates=4000
t2u_dir="/home/junkaiwu/ECE537_Final_Project/text2unit_fairseq"

# change these
dataset="ljspeech_hubert200"
dropouts=( 0.1 0.3 )
attention_dropouts=( 0.0 0.1 )
max_tokens=( 4096 8192 )


project=${arch}-dataset_${dataset}-dropout_${dropout}-attention_dropout_${attention_dropout}-share

save_dir="${t2u_dir}/outputs/${project}"
mkdir -p ${save_dir}

srun --gres=gpu:4 --ntasks=1 fairseq-train \
    ${t2u_dir}/data-bin/${dataset} \
    --distributed-world-size 4 --fp16 \
    --save-dir ${save_dir} --log-file ${save_dir}/train.log --log-format json \
    --arch ${arch} --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr ${lr} --lr-scheduler inverse_sqrt --warmup-updates ${warmup_updates} \
    --dropout ${dropout} --attention_dropout ${attention_dropout} --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens ${max_tokens} --max_epoch ${max_epoch} --validate-interval 99999 \
    --wandb-project ${project}
