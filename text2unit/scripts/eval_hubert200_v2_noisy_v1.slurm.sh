#!/bin/bash
#SBATCH --mem=20g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbmx-delta-gpu
#SBATCH --job-name=eval_text2unit_hubert200_noisy_v1
#SBATCH --output="eval_text2unit_hubert200_noisy_v1_%j.out"
#SBATCH --error="eval_text2unit_hubert200_noisy_v1_%j.err"
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
vocab_size=200
exp_dir="/scratch/bbmx/junkaiwu/text2unit_transformer/hubert${vocab_size}_v2_noisy_v1"
test_txt_path="/u/junkaiwu/ECE537_Project/datasets/LJSpeech/hubert100/test${vocab_size}.txt"

srun --gres=gpu:1 --ntasks=1  python ${text2unit_dir}/eval_v2.py \
    --vocab_size ${vocab_size} \
    --exp_dir ${exp_dir} \
    --test_txt_path ${test_txt_path} \
    --num_workers 16 \
    --batch_size 64 \
    --epoch 295
    



