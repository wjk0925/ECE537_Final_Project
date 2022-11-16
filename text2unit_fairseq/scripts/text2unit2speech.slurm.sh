#!/bin/bash
#SBATCH -J t2u2s
#SBATCH -o t2u2s_%j.%N.out
#SBATCH -e t2u2s_%j.%N.err
#SBATCH --mail-user=junkaiwu@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --qos=sched_level_2
#SBATCH --cpus-per-task=16
#SBATCH --exclude=node0019
#SBATCH --mem=0

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

# for text to unit
source /nobackup/users/heting/espnet/tools/conda/bin/../etc/profile.d/conda.sh
conda activate babblenet

project_dir="transformer_iwslt_de_en-dataset_ljspeech_hubert200-dropout_0.3-max_tokens_4096-share"
epochs=()

fairseq_root="/home/junkaiwu/fairseq-0.12.2"
tacotron_dir="/home/junkaiwu/ECE537_Final_Project/unit2speech/fairseq_tacotron"
export PYTHONPATH=${fairseq_root}:${fairseq_root}/examples/textless_nlp/gslm/unit2speech

feature_type="hubert"
models_path="/nobackup/users/junkaiwu/models/fairseq_tacotron_unti2speech"
out_root="/nobackup/users/junkaiwu/outputs/hubert_tacotron_unit2speech"
waveglow_path="${models_path}/waveglow_256channels_new.pt"

sigma=0.666
denoiser_strength=0.1

vocab_size=200
endding="" 
split="test"

ckpts=( 26 31 40 )

for i in ${!ckpts[@]}; do

    ckpt=${ckpts[$i]}

    tts_model_path="${models_path}/${feature_type}${vocab_size}.pt"
    code_dict_path="${models_path}/code_dict_${feature_type}${vocab_size}"
    quantized_unit_path="/home/junkaiwu/ECE537_Final_Project/text2unit_fairseq/data-bin/ljspeech_hubert200/preds_${ckpt}_5.txt"
    out_dir="${out_root}/preds_${ckpt}_5"

    srun --ntasks=1 --exclusive --gres=gpu:1 --mem=200G -c 16 python ${fairseq_root}/examples/textless_nlp/gslm/unit2speech/synthesize_audio_from_units.py \
        --tts_model_path ${tts_model_path} \
        --quantized_unit_path ${quantized_unit_path} \
        --feature_type ${feature_type} \
        --out_audio_dir ${out_dir} \
        --waveglow_path  ${waveglow_path} \
        --code_dict_path ${code_dict_path} \
        --max_decoder_steps 2000 \
        --sigma ${sigma} \
        --denoiser_strength ${denoiser_strength}

    python ${tacotron_dir}/to16k.py --audio_dir ${out_dir}

done

