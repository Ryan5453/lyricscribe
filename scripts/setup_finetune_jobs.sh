#!/bin/bash -l
#SBATCH --job-name=lyricscribe_ft_setup
#SBATCH --output=/projects/fahey.rya/music2text/logs/finetune/setup_%j.out
#SBATCH --time=08:00:00
#SBATCH --partition=short
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G

# Sets up all finetuning experiments, then launches the orchestrator.

set -euo pipefail

module load FFmpeg/7.1.1
export HF_HOME=/projects/fahey.rya/music2text/.cache/huggingface
export TORCH_HOME=/projects/fahey.rya/music2text/.cache/torch
export NEMO_CACHE_DIR=/projects/fahey.rya/music2text/.cache/nemo

cd /projects/fahey.rya/music2text/lyricscribe
source .venv/bin/activate

DATASET=/projects/fahey.rya/music2text/dataset
EXPDIR=/projects/fahey.rya/music2text/experiments

MODELS=(
    "nvidia/parakeet-tdt-0.6b-v3"
    "openai/whisper-large-v3"
    # "nvidia/canary-1b-v2"  # Canary skipped: too slow to finish by deadline. Re-enable when time allows.
)

# Each entry is a pipe-separated list of filenames. WAV required — MP3
# durations overshoot decoded length and crash Lhotse.
FILENAME_SETS=(
    "htdemucs_ft_vocals.wav"
    "audio.wav"
    "htdemucs_ft_vocals.wav|audio.wav"
)

MAX_EPOCHS="${MAX_EPOCHS:-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0.5}"

for model in "${MODELS[@]}"; do
    for fset in "${FILENAME_SETS[@]}"; do
        # Build --filename flags
        FLAGS=""
        IFS='|' read -ra files <<< "$fset"
        for f in "${files[@]}"; do
            FLAGS="$FLAGS --filename $f"
        done

        echo "=== Setting up: $model / $fset ==="
        lyricscribe finetune setup "$DATASET/final_train" \
            --output-dir "$EXPDIR" \
            --val-dir "$DATASET/final_validation" \
            --model "$model" \
            --max-epochs "$MAX_EPOCHS" \
            --warmup-epochs "$WARMUP_EPOCHS" \
            $FLAGS
    done
done

EXPERIMENTS_FILE=/projects/fahey.rya/music2text/experiments.txt
# Exclude archived experiments in old/ from the orchestrator feed.
find "$EXPDIR" -mindepth 1 -maxdepth 1 -type d ! -name old | sort > "$EXPERIMENTS_FILE"
echo "---"
echo "All experiments set up:"
cat "$EXPERIMENTS_FILE"
echo "---"
echo "Launching orchestrator..."
sbatch scripts/slurm_finetune_orchestrate.sh "$EXPERIMENTS_FILE"
