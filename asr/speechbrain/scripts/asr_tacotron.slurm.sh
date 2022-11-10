#!/bin/bash
#SBATCH --mem=20g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbmx-delta-gpu
#SBATCH --job-name=asr
#SBATCH --output="asr_%j.out"
#SBATCH --error="asr_%j.err"
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
conda activate speechbrain

folders=( "test200_0.666_0.1" "test100_0.666_0.1" "test200_hubert200_v2_290_0.666_0.1" "test100_hubert100_v2_275_0.666_0.1" )

asr_dir="/u/junkaiwu/ECE537_Project/asr/speechbrain"

for i in ${!folders[@]}; do
    folder=${folders[$i]}

    srun --gres=gpu:1 --ntasks=1  python ${asr_dir}/asr_en.py \
        --data_dir /scratch/bbmx/junkaiwu/537/unit2speech/fairseq_tacotron/${folder}/16k \
        --source /u/junkaiwu/speechbrain/asr-transformer-transformerlm-librispeech \
        --savedir /u/junkaiwu/speechbrain/asr-transformer-transformerlm-librispeech
        
    srun --gres=gpu:1 --ntasks=1  python ${asr_dir}/metric.py \
        --preds_path /scratch/bbmx/junkaiwu/537/unit2speech/fairseq_tacotron/${folder}/16k/preds.json \
        --targets_path /scratch/bbmx/junkaiwu/537/unit2speech/fairseq_tacotron/${folder}/16k/targets.json \
        --num 640
        
    rm *.wav
    
done
    



