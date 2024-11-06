#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --constraint="t4|v100-pcie|v100-sxm2" # Kepler GPUs have been causing me issues
#SBATCH --job-name=whisper_large-v1_demucs
#SBATCH --output=~/music2text/output/whisper_large-v1_demucs.out
#SBATCH --time=08:00:00 

module load miniconda3/23.11.0
module load ffmpeg/20190305

source ~/music2text/.venv/bin/activate

# Faster Whisper uses a CTranslate2 backend, which requires these CUDA libraries
# I installed them via pip, so we have to add them to the LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PWD}/.venv/lib64/python3.11/site-packages/nvidia/cublas/lib:${PWD}/.venv/lib64/python3.11/site-packages/nvidia/cudnn/lib

python whisper-wer.py --directory /path/to/your/audio --model large-v1 --use_demucs
