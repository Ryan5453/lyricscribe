#!/bin/bash -l
# Login shell: ensure `module` exists on batch compute nodes (see slurm_transcribe.sh).
#SBATCH --job-name=lyricscribe_orchestrate
#SBATCH --output=/projects/fahey.rya/music2text/logs/transcription/orchestrate_%j.out
#SBATCH --time=48:00:00
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Orchestrator: submits GPU transcription jobs while respecting queue limits.
#
# Usage:
#   sbatch scripts/slurm_orchestrate.sh jobs.txt
#
# Where jobs.txt has one job per line:
#   /path/to/job_dir <chunk_id>
#
# The orchestrator will:
#   1. Reset each job's results before submitting
#   2. Submit up to MAX_SUBMITTED jobs at a time to the gpu partition
#   3. Wait for slots to open before submitting more

set -euo pipefail

MAX_SUBMITTED=8
POLL_INTERVAL=30
# Hardcode path since $0 inside SLURM points to /var/spool/slurmd/...
TRANSCRIBE_SCRIPT="/projects/fahey.rya/music2text/lyricscribe/scripts/slurm_transcribe.sh"

if [ -z "${1:-}" ]; then
    echo "Error: jobs file required"
    echo "Usage: sbatch scripts/slurm_orchestrate.sh <jobs.txt>"
    exit 1
fi

JOBS_FILE="$1"

if [ ! -f "$JOBS_FILE" ]; then
    echo "Error: jobs file not found: $JOBS_FILE"
    exit 1
fi

# Activate environment for reset command
module load FFmpeg/7.1.1
export HF_HOME=/projects/fahey.rya/music2text/.cache/huggingface
export TORCH_HOME=/projects/fahey.rya/music2text/.cache/torch
export NEMO_CACHE_DIR=/projects/fahey.rya/music2text/.cache/nemo

cd /projects/fahey.rya/music2text/lyricscribe
source .venv/bin/activate

# Read all jobs into arrays
JOB_DIRS=()
CHUNK_IDS=()
while IFS=' ' read -r job_dir chunk_id; do
    [ -z "$job_dir" ] && continue
    [[ "$job_dir" == \#* ]] && continue
    JOB_DIRS+=("$job_dir")
    CHUNK_IDS+=("$chunk_id")
done < "$JOBS_FILE"

TOTAL=${#JOB_DIRS[@]}
echo "Orchestrator: $TOTAL jobs to submit (max $MAX_SUBMITTED in queue)"
echo "---"

# Track which job dirs we've already reset
declare -A RESET_DIRS

# Track submitted SLURM job IDs
SLURM_IDS=()

NEXT=0

while true; do
    # Count our currently queued/running GPU jobs
    CURRENT=$(squeue -u "$USER" -p gpu -h 2>/dev/null | wc -l)

    # Submit jobs while we have slots
    while [ "$NEXT" -lt "$TOTAL" ] && [ "$CURRENT" -lt "$MAX_SUBMITTED" ]; do
        JOB_DIR="${JOB_DIRS[$NEXT]}"
        CHUNK_ID="${CHUNK_IDS[$NEXT]}"

        # Submit with retries for transient SLURM errors
        MAX_RETRIES=5
        RETRY_COUNT=0
        while true; do
            if SLURM_OUT=$(sbatch "$TRANSCRIBE_SCRIPT" "$JOB_DIR" "$CHUNK_ID" 2>&1); then
                SLURM_ID=$(echo "$SLURM_OUT" | awk '{print $NF}')
                SLURM_IDS+=("$SLURM_ID")
                echo "[$((NEXT + 1))/$TOTAL] Submitted job $SLURM_ID: $(basename "$(dirname "$JOB_DIR")")/$(basename "$JOB_DIR") chunk $CHUNK_ID"
                break
            else
                RETRY_COUNT=$((RETRY_COUNT + 1))
                if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
                    echo "Error: Failed to submit job after $MAX_RETRIES retries. Output: $SLURM_OUT"
                    exit 1
                fi
                echo "Warning: sbatch failed, retrying in 10s ($RETRY_COUNT/$MAX_RETRIES)..."
                sleep 10
            fi
        done

        NEXT=$((NEXT + 1))
        CURRENT=$((CURRENT + 1))
    done

    # If all submitted, wait for remaining to finish
    if [ "$NEXT" -ge "$TOTAL" ]; then
        REMAINING=$(squeue -u "$USER" -p gpu -h 2>/dev/null | wc -l)
        if [ "$REMAINING" -eq 0 ]; then
            echo "---"
            echo "All $TOTAL jobs completed."
            break
        fi
        echo "Waiting for $REMAINING remaining GPU job(s)..."
    fi

    sleep "$POLL_INTERVAL"
done
