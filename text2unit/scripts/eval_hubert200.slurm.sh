#!/bin/bash
#SBATCH --mem=20g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA100x4
#SBATCH --account=bbmx-delta-gpu
#SBATCH --job-name=eval_text2unit_hubert200
#SBATCH --output="eval_text2unit_hubert200_%j.out"
#SBATCH --error="eval_text2unit_hubert200_%j.err"
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

text2unit_dir="/u/junkaiwu/ECE537_Project/text2unit"

srun --gres=gpu:1 --ntasks=1  python ${text2unit_dir}/eval.py \
    --vocab_size 200 \
    --num_workers 16 \
    --batch_size 64
    



