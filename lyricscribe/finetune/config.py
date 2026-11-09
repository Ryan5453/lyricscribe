import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_architecture(model_name: str) -> str:
    """
    Map a model identifier to its architecture name.

    :param model_name: Model identifier (e.g. ``nvidia/parakeet-tdt-0.6b-v3``).
    :return: One of ``"whisper"``, ``"canary"``, ``"parakeet"``, ``"qwen3_asr"``,
        ``"qwen3_omni"``, ``"nemotron"``, ``"cohere"``, ``"granite"``, ``"musicflamingo"``.
    :raises ValueError: If the name doesn't match any known architecture.
    """
    name = model_name.lower().replace("_", "-")
    if "nemotron-3.5-transcribe" in name:
        return "nemotron"
    if "cohere-transcribe" in name:
        return "cohere"
    if "qwen3-asr" in name:
        return "qwen3_asr"
    if "qwen3-omni" in name:
        return "qwen3_omni"
    if "music-flamingo" in name or "musicflamingo" in name:
        return "musicflamingo"
    if "granite-speech" in name:
        if "nar" in name or "ctc" in name:
            raise ValueError(f"{model_name}: only the Granite Speech LLM is trainable here")
        return "granite"
    if "whisper" in name:
        return "whisper"
    if "canary" in name:
        return "canary"
    if "parakeet" in name:
        return "parakeet"
    raise ValueError(
        f"Cannot detect architecture from model name '{model_name}'. "
        "Model name must contain 'whisper', 'canary', 'parakeet', 'qwen3-asr', "
        "'qwen3-omni', 'nemotron-3.5-transcribe', 'cohere-transcribe', 'granite-speech' "
        "or 'music-flamingo'."
    )


_CONFIG_KEYS = {
    "exp_name",
    "batch_size",
    "global_batch",
    "learning_rate",
    "weight_decay",
    "schedule",
    "warmup_steps",
    "cooldown",
    "min_lr",
    "epochs",
    "max_steps",
    "keep_steps",
    "export_steps",
    "export_every_epoch",
    "export_final",
    "resume_every",
    "use_augmentation",
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
    "max_train_windows",
    "train_window_seed",
    "audio_domain_mode",
    "manifests_from",
    "seed",
    "prompt_mode",
    "adaptation",
    "base_dtype",
    "gradient_checkpointing",
    "num_workers",
}

# Per-device batch (gradient accumulation makes up the global batch) and default LR.
_ARCH_DEFAULTS = {
    "whisper": {"batch_size": 16, "learning_rate": 1e-5},
    "parakeet": {"batch_size": 4, "learning_rate": 3.16e-5},
    "canary": {"batch_size": 16, "learning_rate": 3.16e-5},
    "cohere": {"batch_size": 8, "learning_rate": 3.16e-5},
    "qwen3_asr": {"batch_size": 4, "learning_rate": 3.16e-5},
    "granite": {"batch_size": 1, "learning_rate": 3.16e-5},
    "nemotron": {"batch_size": 1, "learning_rate": 1e-4},
    "qwen3_omni": {"batch_size": 1, "learning_rate": 1e-4},
    "musicflamingo": {"batch_size": 1, "learning_rate": 1e-4},
}

ADAPTATIONS = ("full", "nvidia_adapter")
# Models that only train with the NVIDIA adapter recipe; Qwen3-ASR can train either way.
ADAPTER_ARCHITECTURES = ("nemotron", "qwen3_omni", "musicflamingo")
ADAPTER_LR = 1e-4


def create_finetune_config(
    base_model: str,
    train_dir: Path,
    output_dir: Path,
    filenames: list[str],
    val_dir: Path | None = None,
    **kwargs,
) -> dict:
    """
    Create a finetuning job configuration dictionary. The run length is resolved by
    :func:`resolve_steps` once the training manifest exists.

    :param base_model: Model identifier (e.g., "nvidia/parakeet-tdt-0.6b-v3"); the architecture
        is detected from it.
    :param train_dir: Directory with training songs.
    :param output_dir: Directory for experiment outputs.
    :param filenames: Audio filenames to train on (multiple = random mix).
    :param val_dir: Directory with validation songs (optional).
    :param kwargs: Settings from :data:`_CONFIG_KEYS`.
    :return: Configuration dictionary.
    :raises ValueError: On unknown keys or an adaptation the model can't train with.
    """
    if not filenames:
        raise ValueError("At least one --filename is required")
    unknown = set(kwargs) - _CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown finetune config keys: {sorted(unknown)}")

    architecture = detect_architecture(base_model)
    adaptation = kwargs.get("adaptation") or ("nvidia_adapter" if architecture in ADAPTER_ARCHITECTURES else "full")
    if adaptation not in ADAPTATIONS:
        raise ValueError(f"adaptation must be one of {ADAPTATIONS}, got {adaptation!r}")
    allowed = ("nvidia_adapter",) if architecture in ADAPTER_ARCHITECTURES else (
        ADAPTATIONS if architecture == "qwen3_asr" else ("full",))
    if adaptation not in allowed:
        raise ValueError(f"{architecture} trains with {allowed}, not {adaptation!r}")
    defaults = dict(_ARCH_DEFAULTS.get(architecture, {}))
    if adaptation == "nvidia_adapter":
        defaults["learning_rate"] = ADAPTER_LR
    kwargs = {**defaults, **{k: v for k, v in kwargs.items() if v is not None}}
    schedule = kwargs.get("schedule", "wsd")
    if schedule not in ("wsd", "cosine", "linear"):
        raise ValueError(f"schedule must be wsd, cosine or linear, got {schedule!r}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    arch_short = {"qwen3_asr": "qasr", "qwen3_omni": "qomni", "nemotron": "ntron"}.get(
        architecture, architecture[:3].lower()
    )
    filenames_short = "_".join(Path(f).stem for f in filenames)
    exp_name = kwargs.get("exp_name") or f"{arch_short}_{filenames_short}_{timestamp}"

    config = {
        "architecture": architecture,
        "base_model": base_model,
        "adaptation": adaptation,
        "train_dir": str(train_dir),
        "val_dir": str(val_dir) if val_dir else None,
        "output_dir": str(output_dir),
        "exp_name": exp_name,
        "filenames": filenames,
        "batch_size": kwargs.get("batch_size", 16),
        "global_batch": kwargs.get("global_batch", 64),
        "learning_rate": kwargs.get("learning_rate"),
        "weight_decay": kwargs.get("weight_decay", 0.01),
        "schedule": schedule,
        "warmup_steps": kwargs.get("warmup_steps", 200),
        # WSD: the last fraction of the run that decays (0 = a constant-LR trunk to branch from).
        "cooldown": kwargs.get("cooldown", 0.2),
        "min_lr": kwargs.get("min_lr", 0.0),
        "epochs": kwargs.get("epochs", 3),
        "max_steps": kwargs.get("max_steps"),
        "keep_steps": sorted(set(kwargs.get("keep_steps") or [])),
        "export_steps": sorted(set(kwargs.get("export_steps") or [])),
        "export_every_epoch": kwargs.get("export_every_epoch", False),
        "export_final": kwargs.get("export_final", True),
        "resume_every": kwargs.get("resume_every", 500),
        "seed": kwargs.get("seed", 0),
        "use_augmentation": kwargs.get("use_augmentation", True),
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
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    for key in ("prompt_mode", "base_dtype", "gradient_checkpointing", "num_workers"):
        if key in kwargs:
            config[key] = kwargs[key]
    return config


def resolve_steps(config: dict, n_windows: int) -> dict:
    """
    Fix the run's step plan from the training manifest's size: steps per epoch, the final
    step, the WSD decay start and the export steps.

    :param config: Config from :func:`create_finetune_config`.
    :param n_windows: Training windows.
    :return: *config*, updated.
    :raises ValueError: If the manifest is smaller than one global batch.
    """
    from lyricscribe.finetune.engine import cooldown_start, epochs_to_steps, steps_per_epoch

    spe = steps_per_epoch(n_windows, int(config["global_batch"]))
    if spe < 1:
        raise ValueError(f"{n_windows} windows is less than one global batch of {config['global_batch']}")
    max_steps = int(config["max_steps"] or epochs_to_steps(config["epochs"], spe))
    exports = set(config["export_steps"])
    if config["export_every_epoch"]:
        exports |= {e * spe for e in range(1, max_steps // spe + 1)}
    config.update({
        "steps_per_epoch": spe,
        "max_steps": max_steps,
        "decay_start": (
            cooldown_start(max_steps, config["cooldown"]) if config["schedule"] == "wsd" and config["cooldown"] else None
        ),
        "export_steps": sorted(x for x in exports if x < max_steps),
    })
    return config


def load_finetune_config(job_dir: Path) -> dict:
    """
    Load finetune config from job directory.

    :param job_dir: Path to job directory
    :return: Configuration dictionary
    """
    with open(job_dir / "config.json") as f:
        return json.load(f)


def list_exports(job_dir: Path) -> list[tuple[int, Path]]:
    """
    The job's inference exports, by step.

    :param job_dir: Job directory.
    :return: ``(step, path)`` pairs, oldest first.
    """
    root = job_dir / "exports"
    if not root.is_dir():
        return []
    return sorted((int(p.name.split("_")[1]), p) for p in root.glob("step_*") if p.is_dir())
