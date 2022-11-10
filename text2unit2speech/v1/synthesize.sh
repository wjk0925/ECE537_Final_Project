#!/bin/bash

TEXT2UNIT_DIR="/u/junkaiwu/ECE537_Project/text2unit"
UNIT2SPEECH_DIR="/u/junkaiwu/workspace/diffwave/src/diffwave"

text_path="/u/junkaiwu/ECE537_Project/text2unit2speech/test3.txt"
output_dir="/u/junkaiwu/ECE537_Project/text2unit2speech/v1/test3_100"
vocab_size=100
epoch=275

diffwave_epoch=833808


python ${TEXT2UNIT_DIR}/text2unit.py --text_path ${text_path} --output_dir ${output_dir} --vocab_size ${vocab_size} --epoch ${epoch}

python ${UNIT2SPEECH_DIR}/unit2speech.py \
    --model_path /scratch/bbmx/junkaiwu/diffwave/ljspeech_onehot${vocab_size}_train_v1/weights-${diffwave_epoch}.pt \
    --train_params /u/junkaiwu/workspace/diffwave/params/ljspeech_hubert_onehot${vocab_size}_upv1_small.json \
    --lc_paths ${output_dir}/pred_units.km \
    --output_dir ${output_dir}
    
    
