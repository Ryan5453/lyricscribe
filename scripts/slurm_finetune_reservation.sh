#!/bin/bash -l
# Multi-GPU finetuning on A100 reservation (fahey.rya_test, d1028)
#
# Usage:
#   sbatch scripts/slurm_finetune_reservation.sh /path/to/job-dir <chunk-id> [num-gpus]
#
# num-gpus defaults to 1. Use 1 or 2 for Ganesan's benchmarks.
# The reservation has 4 x A100-80GB on d1028.

#SBATCH --job-name=lyricscribe_finetune
#SBATCH --output=/projects/fahey.rya/music2text/logs/finetune/finetune_%j.out
#SBATCH --time=8:00:00
#SBATCH --partition=reservation
#SBATCH --reservation=fahey.rya_test
#SBATCH --nodelist=d1028

# Default to 1 GPU. For multi-GPU benchmarks, override on the command line:
#   sbatch --gres=gpu:a100:2 --cpus-per-task=32 --mem=128G scripts/slurm_finetune_reservation.sh ... 2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Job directory and chunk ID are required"
    echo "Usage: sbatch scripts/slurm_finetune_reservation.sh /path/to/job-dir <chunk-id> [num-gpus]"
    exit 1
fi

JOB_DIR=$1
CHUNK_ID=$2
NUM_GPUS=${3:-1}

echo "Job directory: $JOB_DIR"
echo "Chunk ID: $CHUNK_ID"
echo "GPUs requested: $NUM_GPUS"

module load cuda/13.2.0
module load FFmpeg/7.1.1

export HF_HOME=/projects/fahey.rya/music2text/.cache/huggingface
export TORCH_HOME=/projects/fahey.rya/music2text/.cache/torch
export NEMO_CACHE_DIR=/projects/fahey.rya/music2text/.cache/nemo

cd /projects/fahey.rya/music2text/lyricscribe
source .venv/bin/activate

set -euo pipefail

echo "Starting finetuning chunk $CHUNK_ID with $NUM_GPUS GPU(s)..."
echo "Start time: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU: use torchrun to launch DDP processes.
    # HF Seq2SeqTrainer auto-detects WORLD_SIZE/LOCAL_RANK from torchrun.
    # NeMo/PL auto-detects torch.cuda.device_count() in trainer_kwargs.
    torchrun --nproc_per_node="$NUM_GPUS" \
        $(which lyricscribe) finetune run --job-dir "$JOB_DIR" --chunk-id "$CHUNK_ID"
else
    lyricscribe finetune run --job-dir "$JOB_DIR" --chunk-id "$CHUNK_ID"
fi

echo "Chunk $CHUNK_ID completed successfully"
echo "End time: $(date)"
