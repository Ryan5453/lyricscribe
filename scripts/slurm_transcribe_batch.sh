#!/bin/bash -l
# Run multiple transcription jobs sequentially in one SLURM allocation.
# Avoids the per-job queue wait when each individual job is short
# (5-15 min). Trade-off: ~30-60s model-reload overhead per job since each
# is a separate `lyricscribe transcribe run` invocation.
#
# Usage:
#   sbatch scripts/slurm_transcribe_batch.sh <jobs.txt>
# Where jobs.txt has one "<job_dir> <chunk_id>" per line.
#SBATCH --job-name=lyricscribe_transcribe_batch
#SBATCH --output=/projects/fahey.rya/music2text/logs/transcription/batch_%j.out
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

set -uo pipefail

if [ -z "${1:-}" ]; then
    echo "Error: jobs file required"
    echo "Usage: sbatch scripts/slurm_transcribe_batch.sh <jobs.txt>"
    exit 1
fi

JOBS_FILE="$1"
if [ ! -f "$JOBS_FILE" ]; then
    echo "Error: jobs file not found: $JOBS_FILE"
    exit 1
fi

module load FFmpeg/7.1.1
export HF_HOME=/projects/fahey.rya/music2text/.cache/huggingface
export TORCH_HOME=/projects/fahey.rya/music2text/.cache/torch
export NEMO_CACHE_DIR=/projects/fahey.rya/music2text/.cache/nemo

cd /projects/fahey.rya/music2text/lyricscribe
source .venv/bin/activate

TOTAL=$(grep -cv '^\s*\(#\|$\)' "$JOBS_FILE")
echo "Batch: $TOTAL jobs from $JOBS_FILE"
echo "Started: $(date)"

DONE=0
FAIL=0
while IFS=' ' read -r job_dir chunk_id; do
    [ -z "$job_dir" ] && continue
    [[ "$job_dir" == \#* ]] && continue

    DONE=$((DONE + 1))
    echo ""
    echo "=== [$DONE/$TOTAL] $(date '+%H:%M:%S') $job_dir chunk $chunk_id ==="

    if lyricscribe transcribe run --job-dir "$job_dir" --chunk-id "$chunk_id"; then
        echo "[$DONE/$TOTAL] OK"
    else
        FAIL=$((FAIL + 1))
        echo "[$DONE/$TOTAL] FAILED (continuing batch)"
    fi
done < "$JOBS_FILE"

echo ""
echo "Batch complete: $DONE jobs run, $FAIL failed"
echo "Finished: $(date)"
