#!/bin/bash
text2unit_dir="/home/junkaiwu/ECE537_Final_Project/text2unit"
vocab_size=200
exp_dir="/home/junkaiwu/ECE537_Final_Project/text2unit/t2u_outputs/hubert200_ljspeech_emb512_heads8_layers6_batch64_warm4000"
split="val"
epoch=290

output_name=${exp_dir}/ljspeech_${split}${epoch}_top1.km

if [ -f "$output_name" ]; then
    echo "exist!"
else
    python eval_v3.py \
        --vocab_size ${vocab_size} \
        --exp_dir ${exp_dir} \
        --test_txt_path /home/junkaiwu/ECE537_Final_Project/datasets/LJSpeech/hubert/${split}${vocab_size}.txt \
        --num_workers 16 \
        --batch_size 20 \
        --epoch ${epoch} \
        --output_name ${output_name} \
	--split ${split} \
        --sampling topk \
        --k 1 \
        --temp 1
fi

ks=( 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 )
temps=( 10 10 10 10 1.0 1.0 1.0 1.0 0.1 0.1 0.1 0.1 0.01 0.01 0.01 0.01 )



for i in ${!ks[@]}; do
    k=${ks[$i]}
    temp=${temps[$i]}
    output_name=${exp_dir}/ljspeech_${split}${epoch}_top${k}_temp${temp}_${i}.km

    if [ -f "$output_name" ]; then
        echo "exist!"
    else
        python eval_v3.py \
            --vocab_size ${vocab_size} \
            --exp_dir ${exp_dir} \
            --test_txt_path /home/junkaiwu/ECE537_Final_Project/datasets/LJSpeech/hubert/${split}${vocab_size}.txt \
            --num_workers 16 \
            --batch_size 20 \
            --epoch ${epoch} \
            --output_name ${output_name} \
	    --split ${split} \
            --sampling topk \
            --k ${k} \
            --temp ${temp}
    fi

done



