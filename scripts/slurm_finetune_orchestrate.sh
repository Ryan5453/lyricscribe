#!/bin/bash -l
#SBATCH --job-name=lyricscribe_ft_orch
#SBATCH --output=/projects/fahey.rya/music2text/logs/finetune/orchestrate_%j.out
#SBATCH --time=48:00:00
#SBATCH --partition=short
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Finetuning orchestrator: manages sequential chunk chains for multiple experiments.
#
# Unlike the transcription orchestrator (where all chunks are independent), finetuning
# chunks within an experiment must run in order (each chunk resumes from the previous
# checkpoint). This orchestrator submits one chunk per experiment at a time, and only
# advances to the next chunk when the previous one succeeds.
#
# Self-resubmits before the 48h time limit if work remains.
#
# Usage:
#   sbatch scripts/slurm_finetune_orchestrate.sh experiments.txt
#
# Where experiments.txt has one experiment directory per line:
#   ./experiments/parakeet_sep_20260403
#   ./experiments/whisper_mix_20260403
#   ./experiments/canary_multi_20260403
#
# Chunk info is read from each experiment's chunks/ directory.

set -euo pipefail

MAX_GPU_SUBMITTED=8
POLL_INTERVAL=60
SELF_RESUBMIT_BUFFER=3600  # Resubmit 1h before time limit
SCRIPT_PATH="/projects/fahey.rya/music2text/lyricscribe/scripts/slurm_finetune.sh"
SELF_PATH="/projects/fahey.rya/music2text/lyricscribe/scripts/slurm_finetune_orchestrate.sh"

if [ -z "${1:-}" ]; then
    echo "Error: experiments file required"
    echo "Usage: sbatch scripts/slurm_finetune_orchestrate.sh <experiments.txt>"
    exit 1
fi

EXPERIMENTS_FILE="$1"

if [ ! -f "$EXPERIMENTS_FILE" ]; then
    echo "Error: experiments file not found: $EXPERIMENTS_FILE"
    exit 1
fi

module load FFmpeg/7.1.1
export HF_HOME=/projects/fahey.rya/music2text/.cache/huggingface
export TORCH_HOME=/projects/fahey.rya/music2text/.cache/torch
export NEMO_CACHE_DIR=/projects/fahey.rya/music2text/.cache/nemo

cd /projects/fahey.rya/music2text/lyricscribe
source .venv/bin/activate

# Read experiment directories
EXPERIMENTS=()
while IFS= read -r exp_dir; do
    exp_dir=$(echo "$exp_dir" | xargs)  # trim whitespace
    [ -z "$exp_dir" ] && continue
    [[ "$exp_dir" == \#* ]] && continue
    EXPERIMENTS+=("$exp_dir")
done < "$EXPERIMENTS_FILE"

NUM_EXPERIMENTS=${#EXPERIMENTS[@]}
echo "Finetune orchestrator: $NUM_EXPERIMENTS experiments"
echo "---"

# For each experiment, figure out the total number of chunks
declare -A TOTAL_CHUNKS
for exp in "${EXPERIMENTS[@]}"; do
    count=$(ls "$exp/chunks"/chunk_*.json 2>/dev/null | wc -l)
    TOTAL_CHUNKS[$exp]=$count
    echo "  $(basename "$exp"): $count chunks"
done
echo "---"

# Track active SLURM job ID per experiment (0 = none active)
declare -A ACTIVE_JOB

for exp in "${EXPERIMENTS[@]}"; do
    ACTIVE_JOB[$exp]=0
done

START_TIME=$(date +%s)

# --- Helper functions ---

mark_chunk_status() {
    local exp_dir=$1
    local chunk_id=$2
    local new_status=$3
    python3 -c "
import json, sys
path = sys.argv[1] + '/chunks/chunk_' + sys.argv[2] + '.json'
with open(path) as f:
    data = json.load(f)
data['status'] = sys.argv[3]
with open(path, 'w') as f:
    json.dump(data, f, indent=2)
" "$exp_dir" "$chunk_id" "$new_status"
}

# Get the next actionable chunk for an experiment.
# Walks chunks in order and finds the first non-"success" chunk.
#   - "pending"  -> prints chunk ID (ready to submit)
#   - "running"  -> prints "running:<id>" (already submitted, wait for it)
#   - "failed"   -> prints "failed:<id>" (chain is blocked, needs manual retry)
#   - all "success" -> prints empty string (experiment complete)
next_actionable_chunk() {
    local exp_dir=$1
    local total=${TOTAL_CHUNKS[$exp_dir]}
    for ((i = 1; i <= total; i++)); do
        local status
        status=$(python3 -c "
import json, sys
with open(sys.argv[1] + '/chunks/chunk_' + sys.argv[2] + '.json') as f:
    print(json.load(f)['status'])
" "$exp_dir" "$i" 2>/dev/null)
        if [ "$status" = "pending" ]; then
            echo "$i"
            return
        elif [ "$status" = "running" ]; then
            echo "running:$i"
            return
        elif [ "$status" = "failed" ]; then
            echo "failed:$i"
            return
        fi
        # "success" -> continue to next chunk
    done
    echo ""
}

experiment_done() {
    local exp_dir=$1
    local result
    result=$(next_actionable_chunk "$exp_dir")
    [ -z "$result" ]
}

submit_chunk() {
    local exp_dir=$1
    local chunk_id=$2
    local retries=0

    while true; do
        if SLURM_OUT=$(sbatch "$SCRIPT_PATH" "$exp_dir" "$chunk_id" 2>&1); then
            SLURM_ID=$(echo "$SLURM_OUT" | awk '{print $NF}')
            ACTIVE_JOB[$exp_dir]=$SLURM_ID
            mark_chunk_status "$exp_dir" "$chunk_id" "running"
            echo "[$(date '+%H:%M:%S')] Submitted $(basename "$exp_dir") chunk $chunk_id (job $SLURM_ID)"
            return 0
        else
            retries=$((retries + 1))
            if [ "$retries" -ge 5 ]; then
                echo "Error: Failed to submit $(basename "$exp_dir") chunk $chunk_id after 5 retries: $SLURM_OUT"
                return 1
            fi
            echo "Warning: sbatch failed, retrying in 10s ($retries/5)..."
            sleep 10
        fi
    done
}

# Recover from previous orchestrator run: any chunks stuck in "running"
# without an actual SLURM job are orphaned and need to be reset to "pending".
echo "Checking for orphaned running chunks..."
NUM_FINETUNE_JOBS=$(squeue -u "$USER" -p gpu -h -o "%j" 2>/dev/null | grep -c "lyricscribe_finetune" || true)
for exp in "${EXPERIMENTS[@]}"; do
    total=${TOTAL_CHUNKS[$exp]}
    for ((i = 1; i <= total; i++)); do
        status=$(python3 -c "
import json, sys
with open(sys.argv[1] + '/chunks/chunk_' + sys.argv[2] + '.json') as f:
    print(json.load(f)['status'])
" "$exp" "$i" 2>/dev/null)
        if [ "$status" = "running" ]; then
            if [ "$NUM_FINETUNE_JOBS" -gt 0 ]; then
                echo "  $(basename "$exp") chunk $i: marked running and finetune jobs exist in queue, leaving as-is"
            else
                echo "  $(basename "$exp") chunk $i: orphaned (no finetune jobs in queue), resetting to pending"
                mark_chunk_status "$exp" "$i" "pending"
            fi
        fi
    done
done
echo "---"

while true; do
    # Check if we need to self-resubmit before timeout
    ELAPSED=$(( $(date +%s) - START_TIME ))
    # 48h = 172800s, minus buffer
    if [ "$ELAPSED" -ge $((172800 - SELF_RESUBMIT_BUFFER)) ]; then
        echo "---"
        echo "Approaching 48h time limit. Waiting for active GPU jobs to finish before resubmitting..."
        # Wait for any active GPU jobs we submitted to finish, so the new
        # orchestrator's orphan detection sees clean state.
        for exp in "${EXPERIMENTS[@]}"; do
            job_id=${ACTIVE_JOB[$exp]}
            [ "$job_id" = "0" ] && continue
            echo "  Waiting for $(basename "$exp") job $job_id..."
            while squeue -j "$job_id" -h 2>/dev/null | grep -q .; do
                sleep 30
            done
        done
        echo "Resubmitting orchestrator..."
        sbatch "$SELF_PATH" "$EXPERIMENTS_FILE"
        echo "Orchestrator resubmitted. Exiting."
        exit 0
    fi

    # Check for completed GPU jobs
    for exp in "${EXPERIMENTS[@]}"; do
        job_id=${ACTIVE_JOB[$exp]}
        [ "$job_id" = "0" ] && continue

        # Check if job is still in the queue
        if squeue -j "$job_id" -h 2>/dev/null | grep -q .; then
            continue  # Still running/pending
        fi

        # Job finished - check outcome via sacct
        JOB_STATE=$(sacct -j "$job_id" --format=State --noheader -P 2>/dev/null | head -1 | xargs)
        ACTIVE_JOB[$exp]=0

        if [ "$JOB_STATE" = "COMPLETED" ]; then
            echo "[$(date '+%H:%M:%S')] $(basename "$exp") job $job_id completed successfully"
        else
            # TIMEOUT, FAILED, OOM, CANCELLED, etc. — reset any "running"
            # chunks back to "pending" so the orchestrator resubmits them.
            if [ "$JOB_STATE" = "TIMEOUT" ]; then
                echo "[$(date '+%H:%M:%S')] WARNING: $(basename "$exp") job $job_id timed out — resetting chunk to pending"
            else
                echo "[$(date '+%H:%M:%S')] WARNING: $(basename "$exp") job $job_id ended with state: $JOB_STATE — resetting chunk to pending"
            fi
            total=${TOTAL_CHUNKS[$exp]}
            for ((ci = 1; ci <= total; ci++)); do
                cs=$(python3 -c "
import json, sys
with open(sys.argv[1] + '/chunks/chunk_' + sys.argv[2] + '.json') as f:
    print(json.load(f)['status'])
" "$exp" "$ci" 2>/dev/null)
                if [ "$cs" = "running" ]; then
                    mark_chunk_status "$exp" "$ci" "pending"
                    echo "[$(date '+%H:%M:%S')]   Reset $(basename "$exp") chunk $ci to pending"
                fi
            done
        fi
    done

    # Count currently submitted GPU jobs
    GPU_SUBMITTED=$(squeue -u "$USER" -p gpu -h 2>/dev/null | wc -l)

    # Submit new chunks where possible
    for exp in "${EXPERIMENTS[@]}"; do
        [ "$GPU_SUBMITTED" -ge "$MAX_GPU_SUBMITTED" ] && break
        [ "${ACTIVE_JOB[$exp]}" != "0" ] && continue  # Already has a running chunk

        NEXT=$(next_actionable_chunk "$exp")
        [ -z "$NEXT" ] && continue  # All chunks done

        # Chain is blocked by a failed chunk — needs manual intervention
        if [[ "$NEXT" == failed:* ]]; then
            FAILED_ID=${NEXT#failed:}
            echo "[$(date '+%H:%M:%S')] BLOCKED: $(basename "$exp") chunk $FAILED_ID failed. Fix and run: lyricscribe finetune retry --job-dir $exp --chunk-id $FAILED_ID"
            continue
        fi

        # Chunk already submitted (e.g., after orchestrator self-resubmit)
        if [[ "$NEXT" == running:* ]]; then
            continue
        fi

        if submit_chunk "$exp" "$NEXT"; then
            GPU_SUBMITTED=$((GPU_SUBMITTED + 1))
        fi
    done

    # Check if everything is done
    ALL_DONE=true
    SUMMARY=""
    for exp in "${EXPERIMENTS[@]}"; do
        if [ "${ACTIVE_JOB[$exp]}" != "0" ]; then
            ALL_DONE=false
        elif ! experiment_done "$exp"; then
            ALL_DONE=false
        fi

        # Build status summary
        NEXT=$(next_actionable_chunk "$exp")
        if experiment_done "$exp"; then
            SUMMARY="$SUMMARY  $(basename "$exp"): done\n"
        elif [ "${ACTIVE_JOB[$exp]}" != "0" ]; then
            SUMMARY="$SUMMARY  $(basename "$exp"): running (job ${ACTIVE_JOB[$exp]})\n"
        elif [[ "$NEXT" == failed:* ]]; then
            SUMMARY="$SUMMARY  $(basename "$exp"): BLOCKED (chunk ${NEXT#failed:} failed)\n"
        elif [ -n "$NEXT" ]; then
            SUMMARY="$SUMMARY  $(basename "$exp"): waiting (next: chunk $NEXT)\n"
        fi
    done

    if $ALL_DONE; then
        echo "---"
        echo "All $NUM_EXPERIMENTS experiments complete!"
        echo ""
        # Print final summary
        for exp in "${EXPERIMENTS[@]}"; do
            local_status=$(python3 -c "
import json, sys
with open(sys.argv[1] + '/status.json') as f:
    s = json.load(f)
    print(f\"{s['current_epoch']}/{s['max_epochs']} epochs, status: {s['status']}\")
" "$exp" 2>/dev/null)
            echo "  $(basename "$exp"): $local_status"
        done
        exit 0
    fi

    sleep "$POLL_INTERVAL"
done
