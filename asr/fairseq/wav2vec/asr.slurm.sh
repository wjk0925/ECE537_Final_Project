#!/bin/bash
source /nobackup/users/heting/espnet/tools/conda/bin/../etc/profile.d/conda.sh
conda activate ssl_disentangle # change it to your conda environment

root=
. utils/parse_options.sh || exit 1;

FAIRSEQ_ROOT="/home/junkaiwu/workspace/ulm"
dict_path="/home/junkaiwu/workspace/ulm_eval/manifests/librispeech100/dict.ltr.txt"
wav2vec_path="/home/junkaiwu/workspace/ulm_eval/models/asr/wav2vec_big_960h.pt"
lm_path="/home/junkaiwu/workspace/ulm_eval/models/asr/4-gram.bin"
lexicon_path="/home/junkaiwu/workspace/ulm_eval/models/asr/lexicon_ltr.lst"


manifest_dir="${root}/manifest"
results_dir="${root}/asr_outputs"

set -e
set -u
set -o pipefail

mkdir -p ${manifest_dir}
cp ${dict_path} ${manifest_dir}

srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python generate_manifest.py --root ${root}

srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python ${FAIRSEQ_ROOT}/examples/speech_recognition/infer.py  \
    ${manifest_dir} \
    --task audio_finetuning --nbest 1 --path ${wav2vec_path} \
    --gen-subset=test --results-path ${results_dir} \
    --w2l-decoder kenlm --lm-model ${lm_path} \
    --lexicon ${lexicon_path} --word-score -1 \
    --sil-weight 0 --lm-weight 2 --criterion ctc --labels ltr --max-tokens 600000 --remove-bpe letter