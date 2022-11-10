#!/bin/bash
FAIRSEQ_ROOT="/u/junkaiwu/fairseq"
FEATURE_TYPE="hubert"
UNIT_SIZE=100
MODELS_PATH="/projects/bbmx/junkaiwu/models/fairseq_tacotron_unit2speech"
QUANTIZED_UNIT_DIR=
QUANTIZED_UNIT_BASE=
OUT_ROOT="/scratch/bbmx/junkaiwu/537/unit2speech/fairseq_tacotron"
SIGMA=0.666
DENOISER_STRENGTH=0.1
. parse_options.sh || exit 1;

TTS_MODEL_PATH="${MODELS_PATH}/${FEATURE_TYPE}${UNIT_SIZE}.pt"
WAVEGLOW_PATH="${MODELS_PATH}/waveglow_256channels_new.pt"
CODE_DICT_PATH="${MODELS_PATH}/code_dict_${FEATURE_TYPE}${UNIT_SIZE}"
QUANTIZED_UNIT_PATH="${QUANTIZED_UNIT_DIR}/${QUANTIZED_UNIT_BASE}"
OUT_DIR="${OUT_ROOT}/${QUANTIZED_UNIT_BASE}_${SIGMA}_${DENOISER_STRENGTH}"


PYTHONPATH=${FAIRSEQ_ROOT}:${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech python ${FAIRSEQ_ROOT}/examples/textless_nlp/gslm/unit2speech/synthesize_audio_from_units.py \
    --tts_model_path ${TTS_MODEL_PATH} \
    --quantized_unit_path "${QUANTIZED_UNIT_PATH}.txt" \
    --feature_type ${FEATURE_TYPE} \
    --out_audio_dir ${OUT_DIR} \
    --waveglow_path  ${WAVEGLOW_PATH} \
    --code_dict_path ${CODE_DICT_PATH} \
    --max_decoder_steps 2000 \
    --sigma ${SIGMA} \
    --denoiser_strength ${DENOISER_STRENGTH}
    
python to16k.py --audio_dir ${OUT_DIR}