#!/bin/bash
#SBATCH -J t2u_eval
#SBATCH -o t2u_eval_%j.%N.out
#SBATCH -e t2u_eval_%j.%N.err
#SBATCH --mail-user=junkaiwu@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
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
## PYTHON_VIRTUAL_ENVIRONMENT=babblenet
## CONDA_ROOT=/nobackup/users/junkaiwu/anaconda3

## Activate WMLCE virtual environment
## source ${CONDA_ROOT}/etc/profile.d/conda.sh
## conda activate $PYTHON_VIRTUAL_ENVIRONMENT

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

name="ljspeech_hubert200"
project=transformer_iwslt_de_en-${name}
fairseq_root="/home/junkaiwu/fairseq-0.12.2"
t2u_dir="/home/junkaiwu/ECE537_Final_Project/text2unit_fairseq"
save_dir="/nobackup/users/junkaiwu/outputs/t2u/${project}"

ckpts=( 26 40 31 24 28 37 45 30 42 46 33 27 51 32 48 35 29 44 59 22 )
beams=( 1 3 5 7 )

for i in ${!ckpts[@]}; do
    ckpt=${ckpts[$i]}
    for j in ${!beams[@]}; do
        beam=${beams[$j]}
        srun --gres=gpu:1 --ntasks=1 fairseq-generate ${t2u_dir}/data-bin/${name} \
            --path ${save_dir}/checkpoint${ckpt}.pt \
            --batch-size 128 --beam ${beam} --remove-bpe | tee ${t2u_dir}/data-bin/${name}/results_${ckpt}_${beam}.txt
    
    done
done




