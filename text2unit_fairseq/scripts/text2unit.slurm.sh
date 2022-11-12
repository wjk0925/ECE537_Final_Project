#!/bin/bash
#SBATCH -J t2u
#SBATCH -o t2u_%j.%N.out
#SBATCH -e t2u_%j.%N.err
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
#SBATCH --exclude=node0013,node0010,node0020,node0039,node0040

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

name="ljspeech_hubert200"
project=transformer_iwslt_de_en-${name}
fairseq_root="/home/junkaiwu/fairseq-0.12.2"
t2u_dir="/home/junkaiwu/ECE537_Project/text2unit_fairseq"
save_dir="/nobackup/users/junkaiwu/outputs/t2u/${project}"
mkdir -p ${save_dir}

srun --gres=gpu:4 --ntasks=1 fairseq-train \
    ${t2u_dir}/data-bin/${name} \
    --distributed-world-size 4 --fp16 \
    --save-dir ${save_dir} --log-file ${save_dir}/train.log --log-format json \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --skip-invalid-size-inputs-valid-test \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 10, \"max_len_b\": 5}" \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
	--save-interval-updates 10000 --keep-interval-updates 5 \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric --wandb-project ${project}
