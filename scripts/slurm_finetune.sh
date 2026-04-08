#!/bin/bash -l
# Use a login shell so Environment Modules are initialized
#SBATCH --job-name=lyricscribe_finetune
#SBATCH --output=/projects/fahey.rya/music2text/logs/finetune/finetune_%j.out
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

# Finetuning job script for SLURM
# Usage: sbatch scripts/slurm_finetune.sh /path/to/job-dir <chunk-id>
#
# This trains one chunk (block of epochs) of a finetuning job.
# Each chunk saves a checkpoint, so you can resume if the job times out.
#
# To run the full training:
#   1. Setup the job: lyricscribe finetune setup ...
#   2. Create a run list: for i in {1..N}; do echo "/path/to/job $i"; done > finetune_jobs.txt
#   3. Use the orchestrator: sbatch scripts/slurm_finetune_orchestrate.sh experiments.txt

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Job directory and chunk ID are required"
    echo "Usage: sbatch scripts/slurm_finetune.sh /path/to/job-dir <chunk-id>"
    exit 1
fi

JOB_DIR=$1
CHUNK_ID=$2
echo "Job directory: $JOB_DIR"
echo "Chunk ID: $CHUNK_ID"

module load cuda/12.8.0
module load FFmpeg/7.1.1

export HF_HOME=/projects/fahey.rya/music2text/.cache/huggingface
export TORCH_HOME=/projects/fahey.rya/music2text/.cache/torch
export NEMO_CACHE_DIR=/projects/fahey.rya/music2text/.cache/nemo
# Synchronous CUDA for debugging device-side asserts (Canary).
# Remove once Canary is working.
export CUDA_LAUNCH_BLOCKING=1

cd /projects/fahey.rya/music2text/lyricscribe
source .venv/bin/activate

set -euo pipefail

echo "Starting finetuning chunk $CHUNK_ID..."
echo "Start time: $(date)"

lyricscribe finetune run --job-dir "$JOB_DIR" --chunk-id "$CHUNK_ID"

echo "Chunk $CHUNK_ID completed successfully"
echo "End time: $(date)"
