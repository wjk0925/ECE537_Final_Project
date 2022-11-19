#!/bin/bash
nisqa_dir="/u/junkaiwu/NISQA"
data_dirs=( "/u/junkaiwu/speech-resynthesis/generations/lj_hubert200_004600000" "/u/junkaiwu/speech-resynthesis/generations/lj_hubert100_00500000_released" )
. parse_options.sh || exit 1;

for i in ${!data_dirs[@]}; do
    data_dir=${data_dirs[$i]}
    
    python ${nisqa_dir}/run_predict.py --mode predict_dir --pretrained_model ${nisqa_dir}/weights/nisqa_tts.tar --data_dir ${data_dir} --num_workers 0 --bs 10 --output_dir ${data_dir}

    mv ${data_dir}/NISQA_results.csv ${data_dir}/NISQA_TTS_results.csv

    python ${nisqa_dir}/run_predict.py --mode predict_dir --pretrained_model ${nisqa_dir}/weights/nisqa.tar --data_dir ${data_dir} --num_workers 0 --bs 10 --output_dir ${data_dir}
    
done

