import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_architecture(model_name: str) -> str:
    """
    Map a model identifier to its architecture name.

    :param model_name: Model identifier (e.g. ``nvidia/parakeet-tdt-0.6b-v3``).
    :return: One of ``"whisper"``, ``"canary"``, ``"parakeet"``.
    :raises ValueError: If the name doesn't match any known architecture.
    """
    name = model_name.lower()
    if "whisper" in name:
        return "whisper"
    if "canary" in name:
        return "canary"
    if "parakeet" in name:
        return "parakeet"
    raise ValueError(
        f"Cannot detect architecture from model name '{model_name}'. "
        "Model name must contain 'whisper', 'canary', or 'parakeet'."
    )


def create_finetune_config(
    base_model: str,
    train_dir: Path,
    output_dir: Path,
    filenames: list[str],
    val_dir: Path | None = None,
    **kwargs,
) -> dict:
    """
    Create a finetuning job configuration dictionary.

    Architecture is auto-detected from model name.

    :param base_model: Model identifier (e.g., "nvidia/parakeet-tdt-0.6b-v3")
    :param train_dir: Directory with training songs
    :param output_dir: Directory for experiment outputs
    :param filenames: Audio filenames to train on (multiple = random mix)
    :param val_dir: Directory with validation songs (optional)
    :param kwargs: Additional config parameters
    :return: Configuration dictionary
    """
    if not filenames:
        raise ValueError("At least one --filename is required")

    architecture = detect_architecture(base_model)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    arch_short = architecture[:3].lower()
    filenames_short = "_".join(Path(f).stem for f in filenames)
    exp_name = kwargs.get("exp_name") if kwargs.get("exp_name") else f"{arch_short}_{filenames_short}_{timestamp}"

    # Per-architecture finetune LR defaults. These aren't lumped by
    # training library — each model has its own sensitivity:
    #  - Whisper: HF's ``Seq2SeqTrainer`` + ``fp16``, 244M–1.55B params.
    #    1e-5 is the community-standard finetune LR.
    #  - Canary: 1B encoder-decoder with bf16 attention that can overflow
    #    at hot LRs. NVIDIA's finetune recipes sit at 5e-6 to 1e-5.
    #  - Parakeet: 600M RNNT/TDT. The bf16 RNNT logits can overflow under
    #    aggressive LRs (we saw NaN at 1e-4 on a smoke run), so we
    #    default lower than NVIDIA's published 1e-4 recipe; users can
    #    override back up via ``--learning-rate``.
    default_lr = {
        "whisper": 1e-5,
        "canary": 1e-5,
        "parakeet": 5e-5,
    }[architecture]

    config = {
        "architecture": architecture,
        "base_model": base_model,
        "train_dir": str(train_dir),
        "val_dir": str(val_dir) if val_dir else None,
        "output_dir": str(output_dir),
        "exp_name": exp_name,
        "filenames": filenames,
        "batch_size": kwargs.get("batch_size", 48 if architecture == "parakeet" else 24),
        "max_epochs": kwargs.get("max_epochs", 10),
        "epochs_per_job": kwargs.get("epochs_per_job", 5 if architecture == "whisper" else 3),
        "learning_rate": kwargs.get("learning_rate", default_lr),
        "warmup_epochs": kwargs.get("warmup_epochs", 5),
        "use_augmentation": kwargs.get("use_augmentation", True),
        "batch_duration": kwargs.get("batch_duration", 1440),
        "freeze_encoder": kwargs.get("freeze_encoder", False),
        "eval_subset_size": kwargs.get("eval_subset_size", 200),
        "current_epoch": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    return config


def load_finetune_config(job_dir: Path) -> dict:
    """
    Load finetune config from job directory.
    
    :param job_dir: Path to job directory
    :return: Configuration dictionary
    """
    with open(job_dir / "config.json") as f:
        return json.load(f)


def get_latest_checkpoint(job_dir: Path, config: dict) -> Path | None:
    """
    Find the latest checkpoint in job directory.
    
    :param job_dir: Path to job directory
    :param config: Job configuration
    :return: Path to latest checkpoint or None
    """
    ckpt_dir = Path(config["output_dir"]) / config["exp_name"] / "checkpoints"
    if not ckpt_dir.exists():
        return None

    checkpoints = list({
        p for p in ckpt_dir.iterdir()
        if p.name.startswith("epoch_")
    })
    if not checkpoints:
        return None
    
    def epoch_from_path(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1
    
    return max(checkpoints, key=epoch_from_path)


def get_checkpoint_for_epoch(job_dir: Path, config: dict, epoch: int) -> Path | None:
    """
    Find the checkpoint for a specific epoch.

    :param job_dir: Path to job directory
    :param config: Job configuration
    :param epoch: Epoch number to find
    :return: Path to checkpoint or None
    """
    ckpt_dir = Path(config["output_dir"]) / config["exp_name"] / "checkpoints"
    if not ckpt_dir.exists():
        return None

    for candidate in ckpt_dir.iterdir():
        if candidate.name == f"epoch_{epoch}" or candidate.stem == f"epoch_{epoch}":
            return candidate

    return None


def list_checkpoints(job_dir: Path, config: dict) -> list[tuple[int, Path]]:
    """
    List all available checkpoints sorted by epoch.

    :return: List of (epoch, path) tuples
    """
    ckpt_dir = Path(config["output_dir"]) / config["exp_name"] / "checkpoints"
    if not ckpt_dir.exists():
        return []

    results = []
    for p in ckpt_dir.iterdir():
        if not p.name.startswith("epoch_"):
            continue
        try:
            epoch = int(p.stem.split("_")[1])
            results.append((epoch, p))
        except (IndexError, ValueError):
            continue

    return sorted(results, key=lambda x: x[0])
