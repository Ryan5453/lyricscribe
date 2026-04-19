#!/bin/bash -l
# Use a login shell so Environment Modules are initialized
#SBATCH --job-name=lyricscribe_finetune
#SBATCH --output=/projects/fahey.rya/music2text/logs/finetune/finetune_%j.out
#SBATCH --time=12:00:00
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=56
#SBATCH --mem=240G

# Finetuning job script for SLURM (multigpu, 4 H200s per job).
# Usage: sbatch scripts/slurm_finetune.sh /path/to/job-dir <chunk-id>
#
# Whisper uses torchrun to launch DDP; NeMo/Parakeet picks up multi-GPU
# via torch.cuda.device_count() inside its PL Trainer config.

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Job directory and chunk ID are required"
    echo "Usage: sbatch scripts/slurm_finetune.sh /path/to/job-dir <chunk-id>"
    exit 1
fi

JOB_DIR=$1
CHUNK_ID=$2
echo "Job directory: $JOB_DIR"
echo "Chunk ID: $CHUNK_ID"

module load cuda/13.2.0
module load FFmpeg/7.1.1

export HF_HOME=/projects/fahey.rya/music2text/.cache/huggingface
export TORCH_HOME=/projects/fahey.rya/music2text/.cache/torch
export NEMO_CACHE_DIR=/projects/fahey.rya/music2text/.cache/nemo

cd /projects/fahey.rya/music2text/lyricscribe
source .venv/bin/activate

set -euo pipefail

NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
ARCHITECTURE=$(python3 -c "import json; print(json.load(open('$JOB_DIR/config.json'))['architecture'])")

echo "Starting finetuning chunk $CHUNK_ID ($ARCHITECTURE) on $NUM_GPUS GPU(s)..."
echo "Start time: $(date)"

if [ "$ARCHITECTURE" = "whisper" ] && [ "$NUM_GPUS" -gt 1 ]; then
    # HF Seq2SeqTrainer uses torchrun env vars for DDP.
    torchrun --nproc_per_node="$NUM_GPUS" \
        $(which lyricscribe) finetune run --job-dir "$JOB_DIR" --chunk-id "$CHUNK_ID"
else
    # NeMo/PL Trainer spawns its own DDP processes via the strategy="ddp" config.
    lyricscribe finetune run --job-dir "$JOB_DIR" --chunk-id "$CHUNK_ID"
fi

echo "Chunk $CHUNK_ID completed successfully"
echo "End time: $(date)"
