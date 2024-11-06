#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# Change this to the GPU you want to use
# I couldn't get the --constraint option to work...
#SBATCH --gres=gpu:XXX:1
#SBATCH --job-name=whisper_large-v1_demucs
#SBATCH --output=/path/to/your/output/whisper_large-v1_demucs.out
#SBATCH --time=08:00:00 

module load miniconda3/23.11.0
module load ffmpeg/20190305

source ${PWD}/.venv/bin/activate

# Faster Whisper uses a CTranslate2 backend, which requires these CUDA libraries
# I installed them via pip, so we have to add them to the LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PWD}/.venv/lib64/python3.11/site-packages/nvidia/cublas/lib:${PWD}/.venv/lib64/python3.11/site-packages/nvidia/cudnn/lib

python -u scripts/whisper-wer.py --directory /path/to/your/audio --model large-v1 --use_demucs
