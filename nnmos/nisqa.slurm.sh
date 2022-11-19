#!/bin/bash
#SBATCH -J nisqa_mos
#SBATCH -o nisqa_mos_%j.%N.out
#SBATCH -e nisqa_mos_%j.%N.err
#SBATCH --mail-user=junkaiwu@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --qos=sched_level_2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --exclusive

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

data_dirs=( "/nobackup/users/junkaiwu/outputs/hubert_hifigan_unit2speech/test100_reconstruct_500000released" "/nobackup/users/junkaiwu/outputs/hubert_hifigan_unit2speech/test200_reconstruct_480000" "/nobackup/users/junkaiwu/outputs/hubert_hifigan_unit2speech/test200_reconstruct_500000" "/nobackup/users/junkaiwu/outputs/hubert_tacotron_unit2speech/test100_reconstruct/16k" "/nobackup/users/junkaiwu/outputs/hubert_tacotron_unit2speech/test200_reconstruct/16k")
nisqa_dir="/home/junkaiwu/NISQA"

for i in ${!data_dirs[@]}; do
    data_dir=${data_dirs[$i]}
        
    srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python ${nisqa_dir}/run_predict.py --mode predict_dir --pretrained_model ${nisqa_dir}/weights/nisqa_tts.tar --data_dir ${data_dir} --num_workers 0 --bs 10 --output_dir ${data_dir}

    mv ${data_dir}/NISQA_results.csv ${data_dir}/NISQA_TTS_results.csv

    srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python ${nisqa_dir}/run_predict.py --mode predict_dir --pretrained_model ${nisqa_dir}/weights/nisqa.tar --data_dir ${data_dir} --num_workers 0 --bs 10 --output_dir ${data_dir}

done
