#!/bin/bash
text2unit_dir="/home/junkaiwu/ECE537_Final_Project/text2unit"
vocab_size=200
exp_dir="/home/junkaiwu/ECE537_Final_Project/text2unit/t2u_outputs/hubert200_ljspeech_emb512_heads8_layers6_batch64_warm4000"
split="val"

for (( epoch=0; epoch<=295; epoch+=5 ))
do
    output_name=${exp_dir}/ljspeech_${split}${epoch}.km

    if [ -f "$output_name" ]; then
        echo "exist!"
    else
        python eval_v2.py \
            --vocab_size ${vocab_size} \
            --exp_dir ${exp_dir} \
            --test_txt_path /home/junkaiwu/ECE537_Final_Project/datasets/LJSpeech/hubert/${split}${vocab_size}.txt \
            --num_workers 16 \
            --batch_size 60 \
            --epoch ${epoch} \
            --output_name ${output_name}
    fi
done



