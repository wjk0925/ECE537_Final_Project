#!/bin/bash
project_dir=
epoch=
gen_subset=
beam=5
dataset="ljspeech_hubert200"
t2u_dir="/home/junkaiwu/ECE537_Final_Project/text2unit_fairseq"

. parse_options.sh || exit 1;

set -e
set -u
set -o pipefail

srun --gres=gpu:1 --ntasks=1 --mem=200G -c 16 fairseq-generate ${t2u_dir}/data-bin/${dataset} \
    --path ${project_dir}/checkpoint${epoch}.pt --gen-subset ${gen_subset} --batch-size 120 --beam ${beam} \
    --max-len-a 20 --max-len-b 5 --scoring wer --fp16 | tee ${project_dir}/units_epoch${epoch}_${gen_subset}_beam${beam}.txt

srun --gres=gpu:1 --ntasks=1 --mem=200G -c 16 ${t2u_dir}/unit_error_rate.py \
    --epoch ${epoch} \
    --eval_path ${project_dir}/units_epoch${epoch}_${gen_subset}_beam${beam}.txt \
    --write_path ${project_dir}/${gen_subset}_uer.txt



