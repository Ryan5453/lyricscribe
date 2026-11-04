import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_architecture(model_name: str) -> str:
    """
    Map a model identifier to its architecture name.

    :param model_name: Model identifier (e.g. ``nvidia/parakeet-tdt-0.6b-v3``).
    :return: One of ``"whisper"``, ``"canary"``, ``"parakeet"``,
        ``"qwen3_asr"``, ``"qwen3_omni"``.
    :raises ValueError: If the name doesn't match any known architecture.
    """
    name = model_name.lower().replace("_", "-")
    if "qwen3-asr" in name:
        return "qwen3_asr"
    if "qwen3-omni" in name:
        return "qwen3_omni"
    if "whisper" in name:
        return "whisper"
    if "canary" in name:
        return "canary"
    if "parakeet" in name:
        return "parakeet"
    raise ValueError(
        f"Cannot detect architecture from model name '{model_name}'. "
        "Model name must contain 'whisper', 'canary', 'parakeet', "
        "'qwen3-asr' or 'qwen3-omni'."
    )


_CONFIG_KEYS = {
    "exp_name",
    "batch_size",
    "max_epochs",
    "epochs_per_job",
    "learning_rate",
    "warmup_epochs",
    "use_augmentation",
    "batch_duration",
    "freeze_encoder",
    "eval_subset_size",
    "whisper_timestamp_prob",
    "dedup_max_run",
    "max_unlabeled_vocal_frac",
    "use_alignment",
    "windows_per_song",
    "line_overlap_threshold",
    "ref_selfwer_file",
    "max_ref_self_wer",
    "grad_accum",
    "save_steps",
    "lora_rank",
    "lora_targets",
    "max_train_windows",
    "train_window_seed",
    "audio_domain_mode",
    "manifests_from",
}

_ARCH_DEFAULTS = {
    "qwen3_asr": {
        "batch_size": 4, "grad_accum": 4, "max_epochs": 3, "epochs_per_job": 3,
        "learning_rate": 3e-5, "warmup_epochs": 0.09, "save_steps": 500,
    },
    "qwen3_omni": {
        "batch_size": 1, "grad_accum": 16, "max_epochs": 1, "epochs_per_job": 1,
        "learning_rate": 1e-4, "warmup_epochs": 0.03, "save_steps": 500,
        "lora_rank": 16, "lora_targets": "attn", "max_train_windows": 20000,
    },
}


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
    unknown = set(kwargs) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown finetune config keys: {sorted(unknown)}")

    architecture = detect_architecture(base_model)
    kwargs = {**_ARCH_DEFAULTS.get(architecture, {}), **kwargs}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    arch_short = {"qwen3_asr": "qasr", "qwen3_omni": "qomni"}.get(architecture, architecture[:3].lower())
    filenames_short = "_".join(Path(f).stem for f in filenames)
    exp_name = kwargs.get("exp_name") if kwargs.get("exp_name") else f"{arch_short}_{filenames_short}_{timestamp}"

    default_lr = {
        "whisper": 1e-5,
        "canary": 1e-5,
        "parakeet": 5e-5,
    }.get(architecture)

    config = {
        "architecture": architecture,
        "base_model": base_model,
        "train_dir": str(train_dir),
        "val_dir": str(val_dir) if val_dir else None,
        "output_dir": str(output_dir),
        "exp_name": exp_name,
        "filenames": filenames,
        "batch_size": kwargs.get("batch_size", 16 if architecture == "parakeet" else 24),
        "max_epochs": kwargs.get("max_epochs", 10),
        "epochs_per_job": kwargs.get("epochs_per_job", 5 if architecture == "whisper" else 3),
        "learning_rate": kwargs.get("learning_rate", default_lr),
        "warmup_epochs": kwargs.get("warmup_epochs", 5),
        "use_augmentation": kwargs.get("use_augmentation", True),
        "batch_duration": kwargs.get("batch_duration", 1440),
        "freeze_encoder": kwargs.get("freeze_encoder", False),
        "eval_subset_size": kwargs.get("eval_subset_size", 200),
        # Fraction of Whisper samples trained with timestamp labels.
        # 0 unlearns timestamps and breaks >30s long-form decoding.
        "whisper_timestamp_prob": kwargs.get("whisper_timestamp_prob", 0.5),
        # Cap on consecutive identical label words (0 = off).
        "dedup_max_run": kwargs.get("dedup_max_run", 0),
        # Reject windows whose sung time is >this fraction unlabeled
        # (None = off). Offsets are re-drawn rather than dropped.
        "max_unlabeled_vocal_frac": kwargs.get("max_unlabeled_vocal_frac"),
        "use_alignment": kwargs.get("use_alignment", True),
        "windows_per_song": kwargs.get("windows_per_song", 1),
        "line_overlap_threshold": kwargs.get("line_overlap_threshold", 0.7),
        "ref_selfwer_file": kwargs.get("ref_selfwer_file"),
        "max_ref_self_wer": kwargs.get("max_ref_self_wer"),
        # Audio-domain post-pass over the manifests (see finetune/domains.py).
        # None = random file per song.
        "audio_domain_mode": kwargs.get("audio_domain_mode"),
        # Experiment whose manifests were copied instead of regenerated.
        "manifests_from": kwargs.get("manifests_from"),
        # Random training-window subset (None = every manifest window).
        "max_train_windows": kwargs.get("max_train_windows"),
        "train_window_seed": kwargs.get("train_window_seed", 0),
        "grad_accum": kwargs.get("grad_accum", 1),
        "save_steps": kwargs.get("save_steps", 2000),
        "current_epoch": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if architecture == "qwen3_omni":
        config["lora_rank"] = kwargs.get("lora_rank", 16)
        config["lora_targets"] = kwargs.get("lora_targets", "attn")

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
