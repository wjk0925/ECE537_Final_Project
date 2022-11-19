#!/bin/bash
#SBATCH -J eval_t2u_transformer
#SBATCH -o eval_t2u_transformer_%j.%N.out
#SBATCH -e eval_t2u_transformer_%j.%N.err
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

t2u_dir="/home/junkaiwu/ECE537_Final_Project/text2unit_fairseq"
projects=( "transformer_iwslt_de_en-dataset_ljspeech_hubert200-dropout_0.3-max_tokens_4096-share" "transformer_iwslt_de_en-dataset_ljspeech_hubert100-dropout_0.3-max_tokens_4096-share" )
num_ckpts=5
beams=( 1 3 5 7 )

for i in ${!projects[@]}; do
    project=${projects[$i]}
    project_dir="${t2u_dir}/outputs/${project}"
    #### Get the ckpts to evaluate from valid_uer.txt
    srun --gres=gpu:1 --ntasks=1 --mem=200G -c 16 python ${t2u_dir}/best_ckpts.py \
        --metric_path ${project_dir}/valid_uer.txt --num_ckpts ${num_ckpts}

    for (( ckpt=1; ckpt<=$num_ckpts; ckpt++ ))
    do
        echo "evaluating best ckpt ${ckpt}"
        for k in ${!beams[@]}; do
            beam=${beams[$k]}

            srun --gres=gpu:1 --ntasks=1 --mem=200G -c 16 fairseq-generate ${t2u_dir}/data-bin/${dataset} \
                --path ${project_dir}/checkpoint_best_{ckpt}.pt --gen-subset test --batch-size 128 --beam ${beam} \
                --max-len-a 20 --max-len-b 5 --scoring wer --fp16 | tee ${project_dir}/units_best_ckpt_${ckpt}_test_beam${beam}.txt
        done
    done
done



