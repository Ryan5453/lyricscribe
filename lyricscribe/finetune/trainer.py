import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import jiwer
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import torch
import torchaudio
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from lyricscribe.finetune.config import get_latest_checkpoint

logger = logging.getLogger(__name__)


def _count_manifest_lines(manifest_path: Path) -> int:
    with open(manifest_path) as f:
        return sum(1 for _ in f)


class NeMoTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.current_epoch = config.get("current_epoch", 0)
        self.model = None
        self.trainer = None
        self.train_manifest = None
        self.val_manifest = None

    def setup(self, train_manifest: Path, val_manifest: Path | None) -> None:
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest

        if self.model is None:
            logger.info(f"Loading NeMo model: {self.config['base_model']}")
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.config["base_model"]
            )

        # Disable CUDA graphs for RNNT/TDT models (e.g. Parakeet).
        # CUDA graphs cause illegal memory access on H200 GPUs.
        decoding = getattr(self.model, "decoding", None)
        inner = getattr(decoding, "decoding", None)
        computer = getattr(inner, "decoding_computer", None)
        if computer is not None and hasattr(computer, "cuda_graphs_mode"):
            computer.cuda_graphs_mode = None

        cfg = self.model.cfg
        OmegaConf.set_struct(cfg, False)

        is_canary = self.config["architecture"] == "canary"

        if is_canary:
            # Canary hard-requires Lhotse. Keep it enabled and fix
            # broken None defaults (NeMo #14816).
            # Audio files MUST be mono — Canary's prompt formatter
            # rejects MultiCut (stereo). Manifests must point to mono files.
            cfg.train_ds.num_buckets = cfg.train_ds.get("num_buckets") or 30
            cfg.train_ds.min_tps = cfg.train_ds.get("min_tps") if cfg.train_ds.get("min_tps") is not None else -1
            cfg.train_ds.max_tps = cfg.train_ds.get("max_tps") if cfg.train_ds.get("max_tps") is not None else float("inf")
        else:
            # Parakeet: disable Lhotse and use the standard NeMo dataloader
            # which handles stereo→mono via channel_selector.
            cfg.train_ds.use_lhotse = False
            cfg.train_ds.channel_selector = "average"

        cfg.train_ds.manifest_filepath = str(train_manifest)
        cfg.train_ds.batch_size = self.config["batch_size"]
        cfg.train_ds.shuffle = True
        cfg.train_ds.num_workers = 4
        # Override pretrained max_duration (default 10s filters everything)
        cfg.train_ds.max_duration = self.config.get("max_duration", 300.0)

        if val_manifest:
            if is_canary:
                cfg.validation_ds.num_buckets = cfg.validation_ds.get("num_buckets") or 30
                cfg.validation_ds.min_tps = cfg.validation_ds.get("min_tps") if cfg.validation_ds.get("min_tps") is not None else -1
                cfg.validation_ds.max_tps = cfg.validation_ds.get("max_tps") if cfg.validation_ds.get("max_tps") is not None else float("inf")
            else:
                cfg.validation_ds.use_lhotse = False
                cfg.validation_ds.channel_selector = "average"
            cfg.validation_ds.manifest_filepath = str(val_manifest)
            cfg.validation_ds.batch_size = self.config["batch_size"]
            cfg.validation_ds.num_workers = 4
            cfg.validation_ds.max_duration = self.config.get("max_duration", 300.0)

        if self.config.get("use_augmentation", True):
            cfg.spec_augment.freq_masks = 2
            cfg.spec_augment.freq_width = 27
            cfg.spec_augment.time_masks = 10
            cfg.spec_augment.time_width = 0.05
        else:
            cfg.spec_augment.freq_masks = 0
            cfg.spec_augment.time_masks = 0

        self.model.setup_training_data(cfg.train_ds)
        if val_manifest:
            self.model.setup_validation_data(cfg.validation_ds)

        # Re-disable struct mode — setup_training_data/setup_validation_data
        # can re-enable it on sub-configs.
        OmegaConf.set_struct(cfg, False)

        # Configure optimizer and scheduler once across all chunks
        cfg.optim.name = "adamw"
        cfg.optim.lr = self.config["learning_rate"]
        cfg.optim.weight_decay = 0.01
        cfg.optim.sched.name = "CosineAnnealing"

        num_train_samples = _count_manifest_lines(self.train_manifest)
        steps_per_epoch = math.ceil(num_train_samples / self.config["batch_size"])
        total_steps = self.config["max_epochs"] * steps_per_epoch

        warmup_epochs = self.config.get("warmup_epochs", 5)
        cfg.optim.sched.warmup_steps = warmup_epochs * steps_per_epoch
        cfg.optim.sched.max_steps = total_steps

    def train_one_epoch(self, target_epoch: int) -> dict:
        # Reuse the same PL Trainer to preserve optimizer state across epochs
        if self.trainer is None:
            wandb_logger = WandbLogger(
                project="lyricscribe-finetune",
                name=self.config["exp_name"],
                tags=[self.config["architecture"], self.config["base_model"]],
            )

            trainer_kwargs = dict(
                max_epochs=self.config["max_epochs"],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                precision="16-mixed" if torch.cuda.is_available() else 32,
                gradient_clip_val=1.0,
                enable_progress_bar=True,
                logger=wandb_logger,
                enable_checkpointing=False,
            )

            # Canary uses Lhotse (iterable dataset, no __len__).
            if self.model.cfg.train_ds.get("use_lhotse", False):
                num_train_samples = _count_manifest_lines(self.train_manifest)
                steps_per_epoch = max(1, num_train_samples // self.config["batch_size"])
                trainer_kwargs["use_distributed_sampler"] = False
                trainer_kwargs["limit_train_batches"] = steps_per_epoch

            self.trainer = pl.Trainer(**trainer_kwargs)

        self.trainer.fit_loop.max_epochs = target_epoch

        logger.info(f"Training epoch {target_epoch}...")
        self.trainer.fit(self.model)

        metrics = {}
        if hasattr(self.trainer, "callback_metrics"):
            for k, v in self.trainer.callback_metrics.items():
                metrics[k] = float(v) if isinstance(v, torch.Tensor) else v

        return metrics

    def save_checkpoint(self, epoch: int, checkpoint_dir: Path) -> Path:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.nemo"
        self.model.save_to(str(checkpoint_path))
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        self.model = nemo_asr.models.ASRModel.restore_from(str(checkpoint_path))

        try:
            epoch = int(checkpoint_path.stem.split("_")[1])
        except (IndexError, ValueError):
            epoch = 0

        logger.info(f"Loaded checkpoint from epoch {epoch}")
        return epoch


class WhisperFinetuneDataset(torch.utils.data.Dataset):
    """Dataset for Whisper finetuning from NeMo-format manifest."""

    def __init__(self, manifest_path: Path, processor: WhisperProcessor):
        self.processor = processor
        self.entries = []
        with open(manifest_path) as f:
            for line in f:
                self.entries.append(json.loads(line))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]

        # Load audio slice if offset/duration are present (chunked manifest)
        offset = entry.get("offset", 0)
        duration = entry.get("duration")

        if offset > 0 or duration is not None:
            import soundfile as sf
            info = sf.info(entry["audio_filepath"])
            frame_offset = int(offset * info.samplerate)
            # Clamp to file bounds to avoid empty reads
            max_frames = info.frames - frame_offset
            if max_frames <= 0:
                audio = torch.zeros(1, 16000)
                sr = 16000
            else:
                num_frames = int(duration * info.samplerate) if duration else -1
                if num_frames > 0:
                    num_frames = min(num_frames, max_frames)
                audio, sr = torchaudio.load(
                    entry["audio_filepath"],
                    frame_offset=frame_offset,
                    num_frames=num_frames,
                )
        else:
            audio, sr = torchaudio.load(entry["audio_filepath"])

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0)
        else:
            audio = audio.squeeze(0)

        # Guard against empty audio (e.g. chunk past end of file)
        if audio.numel() == 0:
            audio = torch.zeros(16000)
            sr = 16000

        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        input_features = self.processor.feature_extractor(
            audio.numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features[0]

        labels = self.processor.tokenizer(entry["text"]).input_ids
        # Whisper's max generation length is 448 tokens
        if len(labels) > 448:
            labels = labels[:448]

        return {"input_features": input_features, "labels": labels}


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        return batch


class WhisperTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.current_epoch = config.get("current_epoch", 0)
        self.model = None
        self.processor = None
        self.hf_trainer = None
        self.train_manifest = None
        self.val_manifest = None

    def _compute_metrics(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 padding with pad token for decoding
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = jiwer.wer(label_str, pred_str)
        return {"wer": wer}

    def setup(self, train_manifest: Path, val_manifest: Path | None) -> None:
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest

        import os
        os.environ.setdefault("WANDB_PROJECT", "lyricscribe-finetune")

        if self.model is None:
            logger.info(f"Loading Whisper model: {self.config['base_model']}")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.config["base_model"]
            )
            self.processor = WhisperProcessor.from_pretrained(self.config["base_model"])

            if torch.cuda.is_available():
                self.model = self.model.cuda()

    def train_one_epoch(self, target_epoch: int) -> dict:
        train_dataset = WhisperFinetuneDataset(self.train_manifest, self.processor)
        eval_dataset = (
            WhisperFinetuneDataset(self.val_manifest, self.processor)
            if self.val_manifest
            else None
        )

        job_dir = Path(self.config["output_dir"]) / self.config["exp_name"]

        warmup_ratio = self.config.get("warmup_epochs", 5) / max(
            self.config["max_epochs"], 1
        )

        # Create trainer once and reuse to preserve optimizer state across epochs.
        # num_train_epochs is set to the total so the LR schedule spans the full run.
        if self.hf_trainer is None:
            logger.info(
                f"Whisper training: {len(train_dataset)} samples. "
                "Note: audio is truncated to 30 seconds by Whisper's feature extractor."
            )

            training_args = Seq2SeqTrainingArguments(
                output_dir=str(job_dir / "hf_trainer"),
                per_device_train_batch_size=self.config["batch_size"],
                learning_rate=self.config["learning_rate"],
                warmup_ratio=warmup_ratio,
                num_train_epochs=self.config["max_epochs"],
                fp16=torch.cuda.is_available(),
                save_strategy="no",
                logging_steps=100,
                remove_unused_columns=False,
                label_names=["labels"],
                weight_decay=0.01,
                max_grad_norm=1.0,
                dataloader_num_workers=4,
                predict_with_generate=True,
                report_to="wandb",
                run_name=self.config["exp_name"],
            )

            self.hf_trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=self._compute_metrics,
                data_collator=WhisperDataCollator(processor=self.processor),
                tokenizer=self.processor,
            )
        else:
            self.hf_trainer.train_dataset = train_dataset
            if eval_dataset is not None:
                self.hf_trainer.eval_dataset = eval_dataset

        self.hf_trainer.args.num_train_epochs = target_epoch

        logger.info(f"Training epoch {target_epoch}...")
        self.hf_trainer.train(resume_from_checkpoint=False)

        metrics = {}
        if eval_dataset is not None:
            eval_results = self.hf_trainer.evaluate()
            metrics = {k: float(v) for k, v in eval_results.items()}
        if self.hf_trainer.state.log_history:
            last_log = self.hf_trainer.state.log_history[-1]
            if "loss" in last_log:
                metrics["train_loss"] = last_log["loss"]

        return metrics

    def save_checkpoint(self, epoch: int, checkpoint_dir: Path) -> Path:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}"
        self.model.save_pretrained(checkpoint_path)
        self.processor.save_pretrained(checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        self.model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
        self.processor = WhisperProcessor.from_pretrained(checkpoint_path)

        try:
            epoch = int(checkpoint_path.name.split("_")[1])
        except (IndexError, ValueError):
            epoch = 0

        if torch.cuda.is_available():
            self.model = self.model.cuda()

        return epoch


def create_trainer(config: dict):
    architecture = config.get("architecture", "parakeet")

    if architecture in ["canary", "parakeet"]:
        return NeMoTrainer(config)
    elif architecture == "whisper":
        return WhisperTrainer(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def run_training_job(
    config: dict,
    train_manifest: Path,
    val_manifest: Path | None,
    chunk_end_epoch: int | None = None,
) -> dict:
    """
    Run one training job (one chunk of epochs).

    :param config: Job configuration dictionary
    :param train_manifest: Path to training manifest
    :param val_manifest: Path to validation manifest (optional)
    :param chunk_end_epoch: Hard cap from the chunk definition (if provided)
    :return: Dict with job results
    """
    trainer = create_trainer(config)

    job_dir = Path(config["output_dir"]) / config["exp_name"]
    start_epoch = config.get("current_epoch", 0)
    latest_checkpoint = get_latest_checkpoint(job_dir, config)

    if latest_checkpoint and start_epoch > 0:
        logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
        try:
            loaded_epoch = trainer.load_checkpoint(latest_checkpoint)
            start_epoch = loaded_epoch + 1
        except Exception as e:
            logger.warning(
                f"Failed to load checkpoint {latest_checkpoint}: {e}. "
                "Starting from last known good epoch."
            )

    trainer.setup(train_manifest, val_manifest)

    max_epoch = config["max_epochs"]
    if chunk_end_epoch is not None:
        max_epoch = min(max_epoch, chunk_end_epoch)

    epochs_to_train = min(
        config.get("epochs_per_job", 5), max_epoch - start_epoch
    )

    if epochs_to_train <= 0:
        logger.info("Training already complete!")
        return {"status": "complete", "epochs_trained": 0}

    checkpoint_dir = job_dir / "checkpoints"
    last_checkpoint_path = None

    logger.info(f"Training {epochs_to_train} epochs (from epoch {start_epoch}), saving after each")

    for epoch in range(start_epoch + 1, start_epoch + epochs_to_train + 1):
        metrics = trainer.train_one_epoch(epoch)

        last_checkpoint_path = trainer.save_checkpoint(epoch, checkpoint_dir)

        config["current_epoch"] = epoch
        config["completed_epochs"] = epoch
        with open(job_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        metrics_entry = {"epoch": epoch}
        metrics_entry.update(metrics)
        with open(job_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics_entry) + "\n")

        logger.info(f"Epoch {epoch} complete, checkpoint saved")

    return {
        "status": "success",
        "start_epoch": start_epoch,
        "end_epoch": start_epoch + epochs_to_train,
        "checkpoint_path": str(last_checkpoint_path),
    }
