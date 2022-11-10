#!/bin/bash
source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate fairseq

data_dir="/scratch/bbmx/junkaiwu/data/LJSpeech_Noisy_1"
TYPE="hubert"
vocab_size=100
ext=".wav"
LAYER=6
CKPT_PATH="/scratch/bbmx/junkaiwu/537/models/hubert_base_ls960.pt"

. parse_options.sh || exit 1;

FAIRSEQ_ROOT="/u/junkaiwu/fairseq"
MANIFEST="${data_dir}/train.tsv"
OUT_QUANTIZED_FILE="${data_dir}/${TYPE}_l${LAYER}_v${vocab_size}.km"
KM_MODEL_PATH="/scratch/bbmx/junkaiwu/537/models/${TYPE}_km${vocab_size}.bin"

# srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 
PYTHONPATH=${FAIRSEQ_ROOT} python ${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/speech2unit/clustering/quantize_with_kmeans.py \
    --feature_type $TYPE \
    --kmeans_model_path $KM_MODEL_PATH \
    --acoustic_model_path $CKPT_PATH \
    --layer $LAYER \
    --manifest_path $MANIFEST \
    --out_quantized_file_path $OUT_QUANTIZED_FILE \
    --extension ${ext} --vocab_size ${vocab_size}