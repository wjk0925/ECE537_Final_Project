#!/bin/bash


text2unit_dir="/u/junkaiwu/ECE537_Project/text2unit"

python ${text2unit_dir}/eval_debug1.py \
    --vocab_size 100 \
    --num_workers 4 \
    --batch_size 64
    
    



