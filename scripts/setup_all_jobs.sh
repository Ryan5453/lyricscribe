#!/bin/bash
# Defines the job matrix and generates setup commands + run list dynamically.
# Run this script to output the setup environment and generate run_all.txt for the orchestrator.

set -euo pipefail

cd /projects/fahey.rya/music2text/lyricscribe
source .venv/bin/activate

DATASET="${1:-/projects/fahey.rya/music2text/dataset}"
JOBS="${2:-/projects/fahey.rya/music2text/jobs}"

echo "=== Configuration ==="
echo "Dataset directory: $DATASET"
echo "Jobs directory:    $JOBS"
echo "====================="

MODELS=(
    "whisper_large_v3:openai/whisper-large-v3"
    "parakeet_tdt_0.6b_v3:nvidia/parakeet-tdt-0.6b-v3"
    "canary_1b_v2:nvidia/canary-1b-v2"
)

# dataset_dir:chunks:cond_name,filename|cond_name,filename
DATASETS=(
    "final_test:4:mix,audio.mp3|sep,htdemucs_ft_vocals.wav"
    "musdb_alt:1:mix,mixture.wav|sep,htdemucs_ft_vocals.wav|stems,vocals.wav"
    "jam-alt:1:mix,audio.mp3|sep,htdemucs_ft_vocals.wav"
)

# mode_name:flags
MODES=(
    "default:"
    "vad:--vad"
    "chunked:--chunked"
    "vad_chunked:--vad --chunked"
)

RUN_FILE="scripts/run_all.txt"
> "$RUN_FILE"

echo "=== Generating setups and run list ==="

for model_info in "${MODELS[@]}"; do
    model_dir="${model_info%%:*}"
    model_id="${model_info#*:}"
    
    echo "Processing model: $model_dir"
    echo "# === $model_dir ===" >> "$RUN_FILE"
    
    for dataset_info in "${DATASETS[@]}"; do
        dataset_dir=$(echo "$dataset_info" | cut -d: -f1)
        chunks=$(echo "$dataset_info" | cut -d: -f2)
        conditions_str=$(echo "$dataset_info" | cut -d: -f3)
        
        dataset_name="${dataset_dir//-/_}"
        
        IFS='|' read -ra conditions <<< "$conditions_str"
        for cond_info in "${conditions[@]}"; do
            cond_name="${cond_info%,*}"
            filename="${cond_info#*,}"
            
            for mode_info in "${MODES[@]}"; do
                mode_name="${mode_info%%:*}"
                mode_flags="${mode_info#*:}"
                
                job_suffix="${dataset_name}_${cond_name}"
                if [ "$mode_name" != "default" ]; then
                    job_suffix="${job_suffix}_${mode_name}"
                fi
                
                job_path="$JOBS/$model_dir/$job_suffix"
                
                # Run setup command directly
                cmd="lyricscribe transcribe setup $DATASET/$dataset_dir \
                    --job-dir $job_path \
                    --filename $filename \
                    --model $model_id \
                    --chunks $chunks \
                    --lyrics-file lyrics.json"
                
                if [ -n "$mode_flags" ]; then
                    cmd="$cmd $mode_flags"
                fi
                
                eval "$cmd"
                
                # Add chunks to run list
                for ((i=1; i<=chunks; i++)); do
                    echo "$job_path $i" >> "$RUN_FILE"
                done
            done
        done
    done
    echo "" >> "$RUN_FILE"
done

echo "Setup complete! Run list generated at $RUN_FILE"
echo "To orchestrate: sbatch scripts/slurm_transcribe_orchestrate.sh $RUN_FILE"
