#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:XXX:1
#SBATCH --job-name=spleeter
#SBATCH --output=/path/to/your/output/spleeter.out
#SBATCH --time=08:00:00 

module load ffmpeg/20190305
module load cuda/11.2

source ${PWD}/.venv/bin/activate

python -u scripts/run-spleeter.py --directory /path/to/your/audio