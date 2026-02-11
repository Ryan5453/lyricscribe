#!/bin/bash
#SBATCH --job-name=lyricscribe_separate
#SBATCH --output=/projects/fahey.rya/music2text/logs/separation/separate_%A_%a.out
#SBATCH --array=1-5%4
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

if [ -z "$1" ]; then
    echo "Error: Job directory not provided"
    echo "Usage: sbatch slurm_separate.sh /path/to/job-dir"
    exit 1
fi

JOB_DIR=$1
echo "Job directory: $JOB_DIR"

module load FFmpeg/7.1.1

cd /projects/fahey.rya/music2text/lyricscribe
source .venv/bin/activate

lyricscribe separate run --job-dir "$JOB_DIR" --chunk-id ${SLURM_ARRAY_TASK_ID}