#!/bin/bash -l
# Fire-and-forget batch-size sweep. Reads real audio files straight
# from a dataset directory (no manifests, no job files, nothing
# persists). Submit on the **same node shape you'll train on** so
# per-rank VRAM matches — DDP bucket size depends on rank count.
#
# Usage:
#   sbatch scripts/slurm_tune_batch.sh <dataset-dir> <filename> <model> [extra args]
#
# Minimal (Parakeet full finetune):
#   sbatch scripts/slurm_tune_batch.sh \
#       /path/to/final_validation \
#       htdemucs_ft_vocals.wav \
#       nvidia/parakeet-tdt-0.6b-v3
#
# Parakeet with frozen encoder (must match production config!):
#   sbatch scripts/slurm_tune_batch.sh \
#       /path/to/final_validation htdemucs_ft_vocals.wav \
#       nvidia/parakeet-tdt-0.6b-v3 \
#       --freeze-encoder
#
# Useful extra args forwarded to ``lyricscribe finetune tune-batch``:
#   --max 64             cap the sweep (faster on small GPUs)
#   --min-gap 1          exact ceiling instead of ±1 slop (one more trial)
#   --max-duration 40    clip length in seconds (must match production)
#   --freeze-encoder     Canary/Parakeet only; must match production

#SBATCH --job-name=lyricscribe_tune_batch
#SBATCH --output=/projects/fahey.rya/music2text/logs/tune_batch/tune_%j.out
#SBATCH --time=4:00:00
#SBATCH --partition=multigpu
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G

if [ $# -lt 3 ]; then
    echo "Usage: sbatch scripts/slurm_tune_batch.sh <dataset-dir> <filename> <model> [extra args]"
    exit 1
fi

DATASET_DIR=$1
FILENAME=$2
MODEL=$3
shift 3
EXTRA_ARGS="$@"

echo "dataset:  $DATASET_DIR"
echo "filename: $FILENAME"
echo "model:    $MODEL"
echo "extra:    $EXTRA_ARGS"
echo "node:     $(scontrol show job $SLURM_JOB_ID 2>/dev/null | grep -E 'NumNodes|NumCPUs|Gres' | xargs)"
echo "start:    $(date)"

module load cuda/13.2.0
module load FFmpeg/7.1.1

export HF_HOME=/projects/fahey.rya/music2text/.cache/huggingface
export TORCH_HOME=/projects/fahey.rya/music2text/.cache/torch
export NEMO_CACHE_DIR=/projects/fahey.rya/music2text/.cache/nemo

cd /projects/fahey.rya/music2text/lyricscribe
source .venv/bin/activate

set -euo pipefail

lyricscribe finetune tune-batch \
    --dataset-dir "$DATASET_DIR" \
    --filename "$FILENAME" \
    --model "$MODEL" \
    $EXTRA_ARGS

echo "end: $(date)"
