#!/bin/bash
nisqa_dir="/u/junkaiwu/NISQA"
data_dir="/nobackup/users/junkaiwu/outputs/hubert_hifigan_unit2speech/test100_reconstruct_500000released"
. parse_options.sh || exit 1;
    
python ${nisqa_dir}/run_predict.py --mode predict_dir --pretrained_model ${nisqa_dir}/weights/nisqa_tts.tar --data_dir ${data_dir} --num_workers 0 --bs 10 --output_dir ${data_dir}

mv ${data_dir}/NISQA_results.csv ${data_dir}/NISQA_TTS_results.csv

python ${nisqa_dir}/run_predict.py --mode predict_dir --pretrained_model ${nisqa_dir}/weights/nisqa.tar --data_dir ${data_dir} --num_workers 0 --bs 10 --output_dir ${data_dir}

