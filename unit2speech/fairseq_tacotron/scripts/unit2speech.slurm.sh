#!/bin/bash
#SBATCH -J u2s
#SBATCH -o u2s_%j.%N.out
#SBATCH -e u2s_%j.%N.err
#SBATCH --mail-user=junkaiwu@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --cpus-per-task=16
#SBATCH --exclude=node0019
#SBATCH --mem=0

## User python environment
PYTHON_VIRTUAL_ENVIRONMENT=fairseq3
CONDA_ROOT=/nobackup/users/junkaiwu/anaconda3

## Activate WMLCE virtual environment
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT

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

fairseq_root="/home/junkaiwu/fairseq-0.12.2"
tacotron_dir="/home/junkaiwu/ECE537_Final_Project/unit2speech/fairseq_tacotron"
export PYTHONPATH=${fairseq_root}:${fairseq_root}/examples/textless_nlp/gslm/unit2speech

feature_type="hubert"
models_path="/nobackup/users/junkaiwu/models/fairseq_tacotron_unti2speech"
waveglow_path="${models_path}/waveglow_256channels_new.pt"

sigma=0.666
denoiser_strength=0.1

t2u_dir="/home/junkaiwu/ECE537_Final_Project/text2unit_fairseq"
projects=( "transformer_iwslt_de_en-dataset_ljspeech_hubert200-dropout_0.3-max_tokens_4096-share" "transformer_iwslt_de_en-dataset_ljspeech_hubert100-dropout_0.3-max_tokens_4096-share" )
vocab_sizes=( 200 100 )
num_ckpts=5
beams=( 1 3 5 7 )

for i in ${!projects[@]}; do
    project=${projects[$i]}
    project_dir="${t2u_dir}/outputs/${project}"
    vocab_size=${vocab_sizes[$i]}

    tts_model_path="${models_path}/${feature_type}${vocab_size}.pt"
    code_dict_path="${models_path}/code_dict_${feature_type}${vocab_size}"

    for (( ckpt=1; ckpt<=$num_ckpts; ckpt++ ))
    do
        for k in ${!beams[@]}; do
            beam=${beams[$k]}

            quantized_unit_path="${project_dir}/units_best_ckpt_${ckpt}_test_beam${beam}_u2s.txt"
            out_dir="${project_dir}/units_best_ckpt_${ckpt}_test_beam${beam}_u2s"

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
    done
done

