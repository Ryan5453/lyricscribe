#!/bin/bash
#SBATCH --job-name=lyricscribe_transcribe
#SBATCH --output=/projects/fahey.rya/music2text/logs/transcription/transcribe_%j.out
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Job directory and chunk ID are required"
    echo "Usage: sbatch slurm_transcribe.sh /path/to/job-dir <chunk-id>"
    exit 1
fi

JOB_DIR=$1
CHUNK_ID=$2
echo "Job directory: $JOB_DIR"
echo "Chunk ID: $CHUNK_ID"

module load FFmpeg/7.1.1

export HF_HOME=/projects/fahey.rya/music2text/.cache/huggingface
export TORCH_HOME=/projects/fahey.rya/music2text/.cache/torch
export NEMO_CACHE_DIR=/projects/fahey.rya/music2text/.cache/nemo

cd /projects/fahey.rya/music2text/lyricscribe
source .venv/bin/activate

lyricscribe transcribe run --job-dir "$JOB_DIR" --chunk-id "$CHUNK_ID"
