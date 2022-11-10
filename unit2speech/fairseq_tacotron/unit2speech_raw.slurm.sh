#!/bin/bash
#SBATCH --mem=20g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --account=bbmx-delta-gpu
#SBATCH --job-name=unit2speech_raw
#SBATCH --output="unit2speech_raw_%j.out"
#SBATCH --error="unit2speech_raw_%j.err"
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
conda activate fairseq

QUANTIZED_UNIT_DIR="/u/junkaiwu/ECE537_Project/datasets/LJSpeech/hubert100"
QUANTIZED_UNIT_BASEs=( "test200", "test100" )
UNIT_SIZEs=( 200 100 )

for i in ${!QUANTIZED_UNIT_BASEs[@]}; do
    QUANTIZED_UNIT_BASE=${QUANTIZED_UNIT_BASEs[$i]}
    UNIT_SIZE=${UNIT_SIZEs[$i]}
    srun --gres=gpu:1 --ntasks=1 unit2speech_t.sh \
        --QUANTIZED_UNIT_DIR ${QUANTIZED_UNIT_DIR} \
        --QUANTIZED_UNIT_BASE ${QUANTIZED_UNIT_BASE} \
        --UNIT_SIZE ${UNIT_SIZE}
done