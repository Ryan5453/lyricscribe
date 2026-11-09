import argparse
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

from lyricscribe.finetune.config import create_finetune_config, resolve_steps
from lyricscribe.finetune.engine import is_main_process, world_size
from lyricscribe.finetune.trainer import run_job

logger = logging.getLogger(__name__)


def main() -> int:
    """
    Run a few training steps at one per-rank batch size and write the peak VRAM (rank 0).

    :return: Process exit code.
    """
    parser = argparse.ArgumentParser(description="Run N training steps and report peak VRAM per rank.")
    parser.add_argument("--manifest", type=Path, required=True, help="Training manifest built by tune_batch.py.")
    parser.add_argument("--model", required=True, help="Model identifier.")
    parser.add_argument("--bs", type=int, required=True, help="Per-rank batch size for this trial.")
    parser.add_argument("--max-duration", type=float, default=30.0, help="Unused; kept for the sweep's command line.")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--output", type=Path, required=True, help="Where rank 0 writes the result JSON.")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format=f"%(asctime)s [rank {os.environ.get('RANK', '0')}] %(levelname)s %(message)s"
    )

    ranks = world_size()
    with open(args.manifest) as f:
        n_windows = sum(1 for line in f if line.strip())
    with tempfile.TemporaryDirectory(prefix="lyricscribe_probe_") as tmp:
        config = create_finetune_config(
            base_model=args.model, train_dir=args.manifest.parent, output_dir=Path(tmp), filenames=["probe"],
            exp_name="probe", batch_size=args.bs, global_batch=args.bs * ranks, max_steps=args.n_steps,
            learning_rate=args.learning_rate, freeze_encoder=args.freeze_encoder, num_workers=2,
        )
        config = resolve_steps(config, n_windows)
        config["train_manifest"] = str(args.manifest)
        logger.info(f"probe {config['architecture']} {args.model} bs={args.bs} world_size={ranks} steps={args.n_steps}")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        run_job(config, Path(tmp), save=False)
        elapsed = time.perf_counter() - t0

    result = {"bs": args.bs, "world_size": ranks, "n_steps": args.n_steps, "wall_seconds": elapsed,
              "mean_step_ms": elapsed / max(args.n_steps, 1) * 1000.0, "gpu": torch.cuda.is_available()}
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        result.update({
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
            "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "total_gb": total / 1024**3,
        })
    if is_main_process():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        logger.info(f"probe result: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
