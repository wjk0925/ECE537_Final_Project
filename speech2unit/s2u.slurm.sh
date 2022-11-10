#!/bin/bash
#SBATCH -J s2u_ljspeech
#SBATCH -o s2u_ljspeech_%j.out
#SBATCH -e s2u_ljspeech_%j.err
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
PYTHON_VIRTUAL_ENVIRONMENT=babblenet2
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

CUR_DIR="/nobackup/users/junkaiwu/diffwave/speech2unit"

data_dir="/nobackup/users/junkaiwu/data/LJSpeech-1.1/wavs_16khz"
TYPE="hubert"
ext=".wav"
LAYER=6
CKPT_PATH="/nobackup/users/junkaiwu/diffwave/speech2unit/models/hubert_base_ls960.pt"

vocab_sizes=( 100 200 500 1000 )

for i in ${!vocab_sizes[@]}; do

    vocab_size=${vocab_sizes[$i]}


    ./s2u.sh --data_dir ${data_dir} --TYPE hubert --vocab_size ${vocab_size} --ext .wav --LAYER 6 --CKPT_PATH /nobackup/users/junkaiwu/diffwave/speech2unit/models/hubert_base_ls960.pt
    
done