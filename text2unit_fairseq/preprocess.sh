#!/bin/bash
name="ljspeech_hubert200_noisy_v1"
. parse_options.sh || exit 1;


fairseq-preprocess --source-lang unit --target-lang char \
    --trainpref /home/junkaiwu/ECE537_Final_Project/text2unit_fairseq/data/${name}/train \
    --validpref /home/junkaiwu/ECE537_Final_Project/text2unit_fairseq/data/${name}/val \
    --testpref /home/junkaiwu/ECE537_Final_Project/text2unit_fairseq/data/${name}/test \
    --destdir /home/junkaiwu/ECE537_Final_Project/text2unit_fairseq/data-bin/${name} \
    --workers 8
