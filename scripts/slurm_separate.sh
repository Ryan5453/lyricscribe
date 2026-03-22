#!/bin/bash
#SBATCH --job-name=lyricscribe_separate
#SBATCH --output=/projects/fahey.rya/music2text/logs/separation/separate_%j.out
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Job directory and chunk ID are required"
    echo "Usage: sbatch slurm_separate.sh /path/to/job-dir <chunk-id>"
    exit 1
fi

JOB_DIR=$1
CHUNK_ID=$2
echo "Job directory: $JOB_DIR"
echo "Chunk ID: $CHUNK_ID"

module load cuda/12.8.0
module load FFmpeg/7.1.1

cd /projects/fahey.rya/music2text/lyricscribe
source .venv/bin/activate

lyricscribe separate run --job-dir "$JOB_DIR" --chunk-id "$CHUNK_ID"
