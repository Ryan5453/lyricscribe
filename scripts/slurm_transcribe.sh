#!/bin/bash
#SBATCH --job-name=lyricscribe_transcribe
#SBATCH --output=/projects/fahey.rya/music2text/logs/transcription/transcribe_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

if [ -z "$1" ]; then
    echo "Error: Job directory not provided"
    echo "Usage: sbatch slurm_transcribe.sh /path/to/job-dir"
    exit 1
fi

JOB_DIR=$1
echo "Job directory: $JOB_DIR"

module load FFmpeg/7.1.1

cd /projects/fahey.rya/music2text/lyricscribe
source .venv/bin/activate

lyricscribe transcribe run --job-dir "$JOB_DIR" --chunk-id 1
