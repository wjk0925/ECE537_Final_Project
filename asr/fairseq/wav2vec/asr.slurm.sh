#!/bin/bash
#SBATCH -J asr
#SBATCH -o asr_%j.%N.out
#SBATCH -e asr_%j.%N.err
#SBATCH --mail-user=junkaiwu@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --wait-all-nodes=1
#SBATCH --qos=sched_level_2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --exclude=node0013,node0010,node0020,node0039,node0040

source /nobackup/users/heting/espnet/tools/conda/bin/../etc/profile.d/conda.sh
conda activate ssl_disentangle # change it to your conda environment

ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE


####    Use MPI for communication with Horovod - this can be hard-coded during installation as well.
export HOROVOD_GPU_ALLREDUCE=MPI
export HOROVOD_GPU_ALLGATHER=MPI
export HOROVOD_GPU_BROADCAST=MPI
export NCCL_DEBUG=DEBUG

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

FAIRSEQ_ROOT="/home/junkaiwu/workspace/ulm"
dict_path="/home/junkaiwu/workspace/ulm_eval/manifests/librispeech100/dict.ltr.txt"
wav2vec_path="/home/junkaiwu/workspace/ulm_eval/models/asr/wav2vec_big_960h.pt"
lm_path="/home/junkaiwu/workspace/ulm_eval/models/asr/4-gram.bin"
lexicon_path="/home/junkaiwu/workspace/ulm_eval/models/asr/lexicon_ltr.lst"

roots=( "/nobackup/users/junkaiwu/outputs/hubert_tacotron_unit2speech/preds_26_5/16k" "/nobackup/users/junkaiwu/outputs/hubert_tacotron_unit2speech/preds_31_5/16k" "/nobackup/users/junkaiwu/outputs/hubert_tacotron_unit2speech/preds_40_5/16k" )

manifest_dir="${root}/manifest"
results_dir="${root}/asr_outputs"

mkdir -p ${manifest_dir}
cp ${dict_path} ${manifest_dir}

for i in ${!roots[@]}; do
    root=${roots[$i]}

    srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python generate_manifest.py --root ${root} --transcription_path /home/junkaiwu/ECE537_Final_Project/datasets/LJSpeech/ljspeech.json --dict_path dict.ltr.txt

    srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python ${FAIRSEQ_ROOT}/examples/speech_recognition/infer.py  \
        ${manifest_dir} \
        --task audio_finetuning --nbest 1 --path ${wav2vec_path} \
        --gen-subset=test --results-path ${results_dir} \
        --w2l-decoder kenlm --lm-model ${lm_path} \
        --lexicon ${lexicon_path} --word-score -1 \
        --sil-weight 0 --lm-weight 2 --criterion ctc --labels ltr --max-tokens 600000 --remove-bpe letter | tee ${manifest_dir}/data-bin/tee_results.txt

done