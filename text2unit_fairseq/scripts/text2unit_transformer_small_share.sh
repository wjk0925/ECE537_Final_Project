#!/bin/bash


set -e
set -u
set -o pipefail

# fixed
arch="transformer_iwslt_de_en"
lr=5e-4
warmup_updates=4000
t2u_dir="/home/junkaiwu/ECE537_Final_Project/text2unit_fairseq"

# change these
dataset="ljspeech_hubert200"
dropouts=( 0.1 )
max_tokenss=( 8192 )
validation_interval=3
patience=2

max_epochs=()
for ii in {1..20}
do
   max_epochs[${#max_epochs[@]}]=$((validation_interval*ii))
done

echo ${max_epochs}

for i in ${!dropouts[@]}; do
    dropout=${dropouts[$i]}
    for j in ${!max_tokens[@]}; do
        max_tokens=${max_tokenss[$j]}

        project=${arch}-dataset_${dataset}-dropout_${dropout}-max_tokens_${max_tokens}-share

        project_dir="${t2u_dir}/outputs/${project}"
        mkdir -p ${project_dir}

        stop_training_file="${project_dir}/stop_training.txt"

        for k in ${!max_epochs[@]}; do
            max_epoch=${max_epochs[$k]}

            source /nobackup/users/heting/espnet/tools/conda/bin/../etc/profile.d/conda.sh
            conda activate babblenet # change it to your conda environment

            if [ -f "$stop_training_file" ]; then
                echo "stop training!"
            else
                fairseq-train \
                    ${t2u_dir}/data-bin/${dataset} \
                    --distributed-world-size 4 --fp16 \
                    --save-dir ${project_dir} --log-file ${project_dir}/train.log --log-format json \
                    --arch ${arch} --share-decoder-input-output-embed \
                    --optimizer adam --adam-betas '(0.9, 0.98)' \
                    --lr ${lr} --lr-scheduler inverse_sqrt --warmup-updates ${warmup_updates} \
                    --dropout ${dropout} --attention_dropout ${attention_dropout} --weight-decay 0.0001 \
                    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
                    --max-tokens ${max_tokens} --max_epoch ${max_epoch} --validate-interval 99999 \
                    --wandb-project ${project}

                for (( epoch=$((max_epoch-validation_interval+1)); epoch<=$max_epoch; epoch++ ))
                do 
                    ./eval_t2u.sh --project_dir ${project_dir} --epoch ${epoch} --gen_subset valid --dataset ${dataset} --t2u_dir ${t2u_dir}
                done

                python ${t2u_dir}/check_early_stop.py \
                    --patience ${patience} \
                    --metric_path ${project_dir}/${gen_subset}_uer.txt
            fi
            
        done
    done
done

        
