#!/bin/bash -l
# Use a login shell so Environment Modules are initialized
#SBATCH --job-name=lyricscribe_align
#SBATCH --output=/projects/fahey.rya/music2text/logs/align/align_%A_%a.out
#SBATCH --time=08:00:00
#SBATCH --partition=short
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G

# Run MFA word-level alignment as a SLURM array, 1 task per shard.
# Alignments are written back into each song's lyrics.json (populates the
# ``alignment`` field). No separate alignments/ directory.
#
# Usage:
#   sbatch --array=0-49 scripts/slurm_align.sh <dataset_dir> <filename> [num_chunks]
#
# Example:
#   sbatch --array=0-49 scripts/slurm_align.sh \
#       /projects/fahey.rya/music2text/dataset/final_train \
#       htdemucs_ft_vocals.wav \
#       50
#
# Chunking:
#   ``--array=0-49`` plus ``num_chunks=50`` splits the dataset into 50 shards
#   by round-robin ``song_dirs[chunk_id::num_chunks]``. Each task processes
#   ~(songs / num_chunks) songs with 64 CPUs of MFA parallelism.
#
# Resume:
#   ``lyricscribe dataset align`` skips songs whose lyrics.json already has
#   a non-null ``alignment`` field, so rerunning the array reprocesses only
#   failures and gaps.

set -euo pipefail

if [ -z "${1:-}" ] || [ -z "${2:-}" ]; then
    echo "Error: dataset_dir and filename are required"
    echo "Usage: sbatch --array=0-49 scripts/slurm_align.sh <dataset_dir> <filename> [num_chunks]"
    exit 1
fi

DATASET_DIR="$1"
FILENAME="$2"
NUM_CHUNKS="${3:-${SLURM_ARRAY_TASK_COUNT:-50}}"

CHUNK_ID="${SLURM_ARRAY_TASK_ID:-0}"

mkdir -p /projects/fahey.rya/music2text/logs/align

cd /projects/fahey.rya/music2text/lyricscribe
source .venv/bin/activate

export LYRICSCRIBE_MFA_CONTAINER=/projects/fahey.rya/music2text/mfa.sif

echo "=== MFA align chunk ${CHUNK_ID}/${NUM_CHUNKS} ==="
echo "Dataset:  $DATASET_DIR"
echo "Filename: $FILENAME"
echo "Node:     $(hostname)"
echo "CPUs:     ${SLURM_CPUS_PER_TASK:-unknown}"
date

lyricscribe dataset align \
    --dataset-dir "$DATASET_DIR" \
    --filename "$FILENAME" \
    --container /projects/fahey.rya/music2text/mfa.sif \
    --mfa-root /projects/fahey.rya/music2text/mfa_cache \
    --num-chunks "$NUM_CHUNKS" \
    --chunk-id "$CHUNK_ID"

echo "=== Done chunk ${CHUNK_ID}/${NUM_CHUNKS} ==="
date
