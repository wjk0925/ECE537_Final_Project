#!/bin/bash
data_dirs=( "/scratch/bbmx/junkaiwu/data/LJSpeech_Noisy_1/test" )

asr_dir="/u/junkaiwu/ECE537_Project/asr/speechbrain"

for i in ${!data_dirs[@]}; do
    data_dir=${data_dirs[$i]}

    python ${asr_dir}/asr_en.py \
        --data_dir ${data_dir} \
        --source /u/junkaiwu/speechbrain/asr-transformer-transformerlm-librispeech \
        --savedir /u/junkaiwu/speechbrain/asr-transformer-transformerlm-librispeech
        
    python ${asr_dir}/metric.py \
        --preds_path ${data_dir}/preds.json \
        --targets_path ${data_dir}/targets.json \
        --ljspeech_test
        
    rm *.wav
    
done
    



