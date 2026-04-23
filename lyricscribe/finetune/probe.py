"""
Single-trial VRAM probe for finetuning, designed to be invoked via
``torchrun --nproc_per_node=N``.

Runs exactly ``--n-steps`` training steps at ``--bs`` against a
pre-built manifest, replicating production's trainer setup 1:1: same
trainer class, same model load path, same Lhotse vs. standard dataloader
dispatch, same PL strategy, same DDP rank count. Rank 0 writes the peak
VRAM + median step time to ``--output`` as JSON; other ranks exit
silently.

Intended to be driven by :mod:`lyricscribe.finetune.tune_batch`, which
builds the manifest once and invokes this module as a torchrun
subprocess per trial so OOMs cleanly kill the whole DDP group without
poisoning a parent process's CUDA state.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_rank0() -> bool:
    """True on the single rank that owns writing the result file."""
    return int(os.environ.get("RANK", "0")) == 0


def _build_probe_config(
    architecture: str,
    model_name: str,
    batch_value: int,
    freeze_encoder: bool,
    max_duration_s: float,
    learning_rate: float,
) -> dict:
    """
    Build a minimal config dict that :func:`create_trainer` and
    :meth:`setup` expect. Mirrors :func:`create_finetune_config` but
    drops everything the probe doesn't need (output_dir, exp_name,
    max_epochs, epochs_per_job, warmup_epochs) — the probe never
    writes checkpoints or runs more than a few steps, and no fake job
    directory is created on disk.

    *batch_value* is architecture-dependent: for Whisper/Parakeet it is
    the per-rank sample count (``batch_size``), for Canary it is seconds
    of audio per batch (``batch_duration``) since Canary's setup sets
    ``batch_size = None`` and only duration batching controls memory.
    """
    config = {
        "architecture": architecture,
        "base_model": model_name,
        "learning_rate": learning_rate,
        "use_augmentation": True,
        "freeze_encoder": freeze_encoder,
        "num_workers": 2,
        # ``max_duration`` caps the NeMo dataloader's per-sample audio
        # length; match production's 240s so long-clip OOM cases still
        # reproduce in the probe.
        "max_duration": max(max_duration_s, 240.0),
    }
    if architecture == "canary":
        # batch_size is ignored (setup sets it to None); duration is the
        # real memory lever here. Pass a conservative placeholder for
        # batch_size so any stray reference doesn't crash.
        config["batch_size"] = 1
        config["batch_duration"] = batch_value
    else:
        config["batch_size"] = batch_value
        # Keep a reasonable batch_duration default for code paths that
        # read it unconditionally; Parakeet/Whisper ignore this field.
        config["batch_duration"] = 600
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run N training steps and report peak VRAM per rank."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Pre-built NeMo manifest JSONL (built once by tune_batch.py).",
    )
    parser.add_argument("--model", required=True, help="Model identifier.")
    parser.add_argument(
        "--bs",
        type=int,
        required=True,
        help=(
            "Batch parameter for this trial. For Whisper and Parakeet, this "
            "is the per-rank sample batch_size. For Canary, this is "
            "batch_duration in seconds (Canary uses Lhotse duration batching "
            "and ignores batch_size)."
        ),
    )
    parser.add_argument("--max-duration", type=float, default=30.0)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where rank 0 writes the result JSON.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [rank {os.environ.get('RANK', '0')}] %(levelname)s %(message)s",
    )

    import torch

    from lyricscribe.finetune.config import detect_architecture
    from lyricscribe.finetune.trainer import create_trainer

    architecture = detect_architecture(args.model)
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    logger.info(
        f"probe arch={architecture} model={args.model} bs={args.bs} "
        f"world_size={world_size} n_steps={args.n_steps}"
    )

    config = _build_probe_config(
        architecture=architecture,
        model_name=args.model,
        batch_value=args.bs,
        freeze_encoder=args.freeze_encoder,
        max_duration_s=args.max_duration,
        learning_rate=args.learning_rate,
    )

    trainer_obj = create_trainer(config)
    trainer_obj.setup(train_manifest=args.manifest, val_manifest=None)

    # Freeze encoder after setup so the config change is reflected in the
    # optimizer's trainable-parameter list when PL Trainer initializes it.
    if args.freeze_encoder and architecture in ("canary", "parakeet"):
        frozen = sum(1 for _ in trainer_obj.model.encoder.parameters())
        for p in trainer_obj.model.encoder.parameters():
            p.requires_grad = False
        logger.info(f"probe froze {frozen} encoder parameter tensors")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    trainer_obj.train_n_steps(args.n_steps)
    elapsed = time.perf_counter() - t0

    if not torch.cuda.is_available():
        if _is_rank0():
            args.output.write_text(
                json.dumps(
                    {
                        "bs": args.bs,
                        "world_size": world_size,
                        "wall_seconds": elapsed,
                        "gpu": False,
                    }
                )
            )
        return 0

    peak_reserved = torch.cuda.max_memory_reserved()
    peak_allocated = torch.cuda.max_memory_allocated()
    total = torch.cuda.get_device_properties(0).total_memory

    result = {
        "bs": args.bs,
        "world_size": world_size,
        "n_steps": args.n_steps,
        "wall_seconds": elapsed,
        "mean_step_ms": (elapsed / max(args.n_steps, 1)) * 1000.0,
        "peak_reserved_gb": peak_reserved / (1024**3),
        "peak_allocated_gb": peak_allocated / (1024**3),
        "total_gb": total / (1024**3),
    }

    if _is_rank0():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        logger.info(
            f"probe result: bs={args.bs} peak_reserved={result['peak_reserved_gb']:.1f} GiB "
            f"({100 * result['peak_reserved_gb'] / result['total_gb']:.0f}%) "
            f"mean_step={result['mean_step_ms']:.0f} ms"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
