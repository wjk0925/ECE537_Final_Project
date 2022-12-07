#!/bin/bash
#SBATCH -J evaluate_hubert_onLJ
#SBATCH -o evaluate_hubert_onLJ_%j.%N.out
#SBATCH -e evaluate_hubert_onLJ_%j.%N.err
#SBATCH --mail-user=junkaiwu@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --cpus-per-task=20
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

text2unit_dir="/home/junkaiwu/ECE537_Final_Project/text2unit"
vocab_size=200
exp_dir="/home/junkaiwu/ECE537_Final_Project/text2unit/t2u_outputs/hubert200_ljspeech_emb512_heads8_layers6_batch64_warm4000"
split="val"
epochs=( 290 )

for i in ${!epochs [@]}; do
    epoch=${epochs[$i]}
    output_name=${exp_dir}/ljspeech_${split}${epoch}.km

    srun --gres=gpu:1 --ntasks=1 python ${text2unit_dir}/eval_v2.py \
        --vocab_size ${vocab_size} \
        --exp_dir ${exp_dir} \
        --test_txt_path /home/junkaiwu/ECE537_Final_Project/datasets/LJSpeech/hubert/${split}${vocab_size}.txt \
        --num_workers 16 \
        --batch_size 60 \
        --epoch ${epoch} \
        --output_name ${output_name}
    
done



