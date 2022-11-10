#!/bin/bash
txt_path="/u/junkaiwu/ECE537_Project/datasets/LJSpeech/hubert100/test200.txt"
output_dir="/scratch/bbmx/junkaiwu/537/tts/speechbrain_ljspeech_tacotron-hifigan/test"
. parse_options.sh || exit 1;

python ljspeech_tacotron-hifigan.py --txt_path ${txt_path} --output_dir ${output_dir}
    
python to16k.py --audio_dir ${output_dir}