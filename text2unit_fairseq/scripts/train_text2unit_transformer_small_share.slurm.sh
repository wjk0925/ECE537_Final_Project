#!/bin/bash
#SBATCH -J train_t2u_transformer
#SBATCH -o train_t2u_transformer_%j.%N.out
#SBATCH -e train_t2u_transformer_%j.%N.err
#SBATCH --mail-user=junkaiwu@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --qos=sched_level_2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --exclusive

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

source /nobackup/users/heting/espnet/tools/conda/bin/../etc/profile.d/conda.sh
conda activate babblenet

# fixed
arch="transformer_iwslt_de_en"
lr=5e-4
warmup_updates=4000
t2u_dir="/home/junkaiwu/ECE537_Final_Project/text2unit_fairseq"

# change these
dataset="ljspeech_hubert200"
dropouts=( 0.3 )
max_tokenss=( 4096 )
validation_interval=25
patience=24

max_epochs=()
for ii in {1..20}
do
   max_epochs[${#max_epochs[@]}]=$((validation_interval*ii))
done

for i in ${!dropouts[@]}; do
    dropout=${dropouts[$i]}
    for j in ${!max_tokenss[@]}; do
        max_tokens=${max_tokenss[$j]}

        project=${arch}-dataset_${dataset}-dropout_${dropout}-max_tokens_${max_tokens}-share

        project_dir="${t2u_dir}/outputs/${project}"
        mkdir -p ${project_dir}

        stop_training_file="${project_dir}/stop_training.txt"
        echo ${stop_training_file}

        for k in ${!max_epochs[@]}; do
            max_epoch=${max_epochs[$k]}

            if [ -f "$stop_training_file" ]; then
                echo "stop training!"
            else
                srun --gres=gpu:4 --ntasks=1 fairseq-train \
                    ${t2u_dir}/data-bin/${dataset} \
                    --distributed-world-size 4 --fp16 \
                    --save-dir ${project_dir} --log-file ${project_dir}/train.log --log-format json \
                    --arch ${arch} --share-decoder-input-output-embed \
                    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
                    --lr ${lr} --lr-scheduler inverse_sqrt --warmup-updates ${warmup_updates} \
                    --dropout ${dropout} --weight-decay 0.0001 \
                    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
                    --max-tokens ${max_tokens} --max-epoch ${max_epoch} --validate-interval 99999 \
                    --wandb-project ${project}
                
                sleep 10

                for (( epoch=$((max_epoch-validation_interval+1)); epoch<=$max_epoch; epoch++ ))
                do
                    echo "evaluating epoch${epoch}"
                    ./eval_t2u.sh --project_dir ${project_dir} --epoch ${epoch} --gen_subset valid --dataset ${dataset} --t2u_dir ${t2u_dir}
                done

                srun --gres=gpu:1 --ntasks=1 --mem=200G -c 16 python ${t2u_dir}/check_early_stop.py \
                    --patience ${patience} \
                    --metric_path ${project_dir}/valid_uer.txt
            fi
            
        done
    done
done

        
