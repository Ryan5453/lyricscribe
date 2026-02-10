#!/bin/bash
#SBATCH --job-name=lyricscribe_separate
#SBATCH --output=/projects/fahey.rya/music2text/logs/separate_%A_%a.out
#SBATCH --error=/projects/fahey.rya/music2text/logs/separate_%A_%a.err
#SBATCH --array=1-5
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Usage: sbatch --array=1-5 slurm_separate.sh /path/to/job.db

echo "Chunk ID: ${SLURM_ARRAY_TASK_ID}"
echo "GPU: $CUDA_VISIBLE_DEVICES"

if [ -z "$1" ]; then
    echo "Error: Database path not provided"
    echo "Usage: sbatch slurm_separate.sh /path/to/job.db"
    exit 1
fi

DB_PATH=$1
echo "Database: $DB_PATH"

module load ffmpeg

cd /projects/fahey.rya/music2text/lyricscribe
source .venv/bin/activate


lyricscribe separate run --db "$DB_PATH" --chunk-id ${SLURM_ARRAY_TASK_ID}