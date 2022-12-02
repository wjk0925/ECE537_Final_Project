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


N_CLUSTERS_list=( 500 1000 )
TYPE="hubert"
CKPT_PATH="/home/junkaiwu/ECE537_Final_Project/speech2unit/models/hubert_base_ls960.pt"
LAYER=6
MANIFEST="/home/junkaiwu/workspace/ulm/examples/ulm/manifest/LibriSpeech100-wavenet/train.tsv"
KM_MODEL_PATH="/home/junkaiwu/ECE537_Final_Project/speech2unit/models/${TYPE}_km${vocab_size}.bin"

fairseq_root="/home/junkaiwu/fairseq-0.12.2"
export PYTHONPATH=${fairseq_root}

for i in ${!N_CLUSTERS_list[@]}; do

    N_CLUSTERS=${N_CLUSTERS_list[$i]}

    srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python /home/junkaiwu/fairseq-0.12.2/examples/textless_nlp/gslm/speech2unit/clustering/cluster_kmeans.py \
        --num_clusters $N_CLUSTERS \
        --feature_type $TYPE \
        --checkpoint_path $CKPT_PATH \
        --layer $LAYER \
        --manifest_path $MANIFEST \
        --out_kmeans_model_path $KM_MODEL_PATH
        
done