import json
import logging
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import jiwer
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import soundfile as sf
import torch
import torchaudio
import wandb
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from lyricscribe.finetune.config import get_latest_checkpoint

logger = logging.getLogger(__name__)


def _is_main_process() -> bool:
    """Return True if this is rank 0 (or single-GPU). Used to guard
    file writes (checkpoints, metrics, config) in multi-GPU DDP."""
    import os
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def _count_manifest_lines(manifest_path: Path) -> int:
    """
    Count the number of lines in a manifest JSONL file.

    :param manifest_path: Path to the manifest file.
    :return: Number of lines.
    """
    with open(manifest_path) as f:
        return sum(1 for _ in f)


class NeMoTrainer:
    """
    Trainer for NeMo ASR models (Parakeet, Canary) using PyTorch Lightning.

    Handles model loading, data configuration, per-epoch training, and
    checkpoint management. Designed to be called once per chunk of epochs
    by the orchestration system.

    :param config: Job configuration dictionary.
    """

    def __init__(self, config: dict):
        self.config = config
        self.current_epoch = config.get("current_epoch", 0)
        self.model = None
        self.trainer = None
        self.train_manifest = None
        self.val_manifest = None

    def setup(self, train_manifest: Path, val_manifest: Path | None) -> None:
        """
        Configure the model, data loaders, optimizer, and scheduler.

        For Parakeet, disables Lhotse and uses the standard NeMo dataloader
        with ``channel_selector="average"`` for stereo-to-mono conversion.

        For Canary, keeps Lhotse enabled (hard requirement) with duration-based
        batching and fixes for broken None defaults in pretrained configs.

        :param train_manifest: Path to the training manifest JSONL file.
        :param val_manifest: Path to the validation manifest JSONL file,
            or ``None`` to skip validation.
        """
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
            # Canary hard-requires Lhotse (asserts use_lhotse=True).
            # Fix broken None defaults in pretrained configs (NeMo #14816).
            cfg.train_ds.num_buckets = cfg.train_ds.get("num_buckets") or 30
            cfg.train_ds.min_tps = cfg.train_ds.get("min_tps") if cfg.train_ds.get("min_tps") is not None else -1
            cfg.train_ds.max_tps = cfg.train_ds.get("max_tps") if cfg.train_ds.get("max_tps") is not None else float("inf")
            cfg.train_ds.batch_duration = 600
            cfg.train_ds.batch_size = None
        else:
            # Parakeet: use the standard NeMo dataloader.
            # channel_selector="average" mixes stereo to mono at load time.
            cfg.train_ds.use_lhotse = False
            cfg.train_ds.channel_selector = "average"
            cfg.train_ds.batch_size = self.config["batch_size"]

        cfg.train_ds.manifest_filepath = str(train_manifest)
        cfg.train_ds.min_duration = 0.1
        cfg.train_ds.max_duration = 240.0
        cfg.train_ds.shuffle = True
        cfg.train_ds.num_workers = 4

        if val_manifest:
            if is_canary:
                cfg.validation_ds.num_buckets = cfg.validation_ds.get("num_buckets") or 30
                cfg.validation_ds.min_tps = cfg.validation_ds.get("min_tps") if cfg.validation_ds.get("min_tps") is not None else -1
                cfg.validation_ds.max_tps = cfg.validation_ds.get("max_tps") if cfg.validation_ds.get("max_tps") is not None else float("inf")
                cfg.validation_ds.batch_duration = 600
                cfg.validation_ds.batch_size = None
            else:
                cfg.validation_ds.use_lhotse = False
                cfg.validation_ds.channel_selector = "average"
                cfg.validation_ds.batch_size = self.config["batch_size"]
            cfg.validation_ds.manifest_filepath = str(val_manifest)
            cfg.validation_ds.min_duration = 0.1
            cfg.validation_ds.max_duration = 240.0
            cfg.validation_ds.num_workers = 4

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
        batch_size = self.config.get("batch_size") or max(1, 600 // 180)
        steps_per_epoch = math.ceil(num_train_samples / max(batch_size, 1))
        total_steps = self.config["max_epochs"] * steps_per_epoch

        warmup_epochs = self.config.get("warmup_epochs", 5)
        cfg.optim.sched.warmup_steps = warmup_epochs * steps_per_epoch
        cfg.optim.sched.max_steps = total_steps

    def train_one_epoch(self, target_epoch: int) -> dict:
        """
        Train the model up to *target_epoch* using PyTorch Lightning.

        On the first call, initializes a wandb run (best-effort) and creates
        the PL Trainer. Subsequent calls reuse the same Trainer to preserve
        optimizer state across epochs.

        :param target_epoch: The epoch number to train up to (1-indexed).
        :return: Dict of metrics from ``trainer.callback_metrics``.
        """
        if self.trainer is None:
            try:
                wandb.init(
                    project="lyricscribe-finetune",
                    name=self.config["exp_name"],
                    id=self.config["exp_name"],
                    resume="allow",
                    tags=[self.config["architecture"], self.config["base_model"]],
                    settings=wandb.Settings(init_timeout=300),
                )
                wandb_logger = WandbLogger(experiment=wandb.run)
            except Exception as e:
                logger.warning(f"wandb init failed ({e}), training without wandb")
                wandb_logger = None

            uses_lhotse = self.model.cfg.train_ds.get("use_lhotse", False)

            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            if num_gpus > 1:
                logger.info(f"Multi-GPU detected: {num_gpus} devices, using DDP")

            # Mid-epoch step checkpoints so SLURM timeouts don't waste
            # an entire epoch of training. PL restores full state (model,
            # optimizer, scheduler, step counter) on resume.
            self._step_ckpt_dir = (
                Path(self.config["output_dir"])
                / self.config["exp_name"]
                / "pl_step_checkpoints"
            )
            step_checkpoint = ModelCheckpoint(
                dirpath=str(self._step_ckpt_dir),
                every_n_train_steps=2000,
                save_top_k=-1,
                filename="step-{step:08d}",
            )

            trainer_kwargs = dict(
                max_epochs=self.config["max_epochs"],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=num_gpus,
                strategy="ddp" if num_gpus > 1 else "auto",
                precision="bf16-mixed" if torch.cuda.is_available() else 32,
                gradient_clip_val=1.0,
                enable_progress_bar=True,
                logger=wandb_logger,
                enable_checkpointing=True,
                callbacks=[step_checkpoint],
                log_every_n_steps=1,
            )

            # Cap validation to a fixed number of batches so we get a quick
            # sanity check each epoch instead of running over the entire
            # ~88k-chunk validation manifest. The full eval can be done
            # separately with the inference harness on a real test set.
            if self.val_manifest is not None:
                trainer_kwargs["limit_val_batches"] = self.config.get(
                    "eval_subset_size", 200
                )

            if uses_lhotse:
                # Lhotse iterable datasets don't have __len__.
                num_train_samples = _count_manifest_lines(self.train_manifest)
                batch_size_est = self.config.get("batch_size") or max(1, 600 // 180)
                steps_per_epoch = max(1, num_train_samples // max(batch_size_est, 1))
                trainer_kwargs["use_distributed_sampler"] = False
                trainer_kwargs["limit_train_batches"] = steps_per_epoch

            self.trainer = pl.Trainer(**trainer_kwargs)

        self.trainer.fit_loop.max_epochs = target_epoch

        # Resume from the latest step checkpoint if one exists (e.g. after
        # a SLURM timeout mid-epoch). PL restores model weights, optimizer,
        # scheduler, and dataloader position so we lose at most ~2000 steps.
        ckpt_path = None
        step_ckpt_dir = getattr(self, "_step_ckpt_dir", None)
        if step_ckpt_dir and step_ckpt_dir.exists():
            step_ckpts = sorted(
                step_ckpt_dir.glob("step-*.ckpt"),
                key=lambda p: p.stat().st_mtime,
            )
            if step_ckpts:
                ckpt_path = str(step_ckpts[-1])
                logger.info(f"Resuming from step checkpoint: {ckpt_path}")

        logger.info(f"Training epoch {target_epoch}...")
        self.trainer.fit(self.model, ckpt_path=ckpt_path)

        metrics = {}
        if hasattr(self.trainer, "callback_metrics"):
            for k, v in self.trainer.callback_metrics.items():
                metrics[k] = float(v) if isinstance(v, torch.Tensor) else v

        # NeMo logs with on_epoch=True which PL may not forward to wandb
        # in our setup. Manually log so we get charts.
        if metrics:
            if wandb.run is not None:
                wandb.log(metrics, step=target_epoch)
            logger.info(f"Epoch {target_epoch} metrics: {metrics}")

        return metrics

    def save_checkpoint(self, epoch: int, checkpoint_dir: Path) -> Path:
        """
        Save a NeMo ``.nemo`` checkpoint.

        :param epoch: Epoch number for the filename.
        :param checkpoint_dir: Directory to save into (created if missing).
        :return: Path to the saved checkpoint file.
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}.nemo"
        self.model.save_to(str(checkpoint_path))
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """
        Restore a model from a ``.nemo`` checkpoint.

        :param checkpoint_path: Path to the checkpoint file.
        :return: The epoch number extracted from the filename.
        """
        self.model = nemo_asr.models.ASRModel.restore_from(str(checkpoint_path))

        try:
            epoch = int(checkpoint_path.stem.split("_")[1])
        except (IndexError, ValueError):
            epoch = 0

        logger.info(f"Loaded checkpoint from epoch {epoch}")
        return epoch


class WhisperFinetuneDataset(torch.utils.data.Dataset):
    """
    Map-style dataset for Whisper finetuning from a manifest JSONL file.

    Each sample loads an audio chunk (using ``offset``/``duration`` from the
    manifest if present), converts to mono 16 kHz, extracts log-mel features,
    and tokenizes the transcript. Chunks are pre-sized during manifest
    creation to fit within Whisper's 448-token decoder limit.

    :param manifest_path: Path to the JSONL manifest file.
    :param processor: A ``WhisperProcessor`` instance for feature extraction
        and tokenization.
    """

    def __init__(self, manifest_path: Path, processor: WhisperProcessor):
        self.processor = processor
        self.entries = []
        with open(manifest_path) as f:
            for line in f:
                self.entries.append(json.loads(line))

    def __len__(self) -> int:
        """
        Return the number of manifest entries.

        :return: Entry count.
        """
        return len(self.entries)

    def _load_audio(self, entry: dict) -> tuple[torch.Tensor, int]:
        """
        Load audio from a manifest entry, handling offset/duration slicing
        and stereo-to-mono conversion.

        :param entry: A single manifest entry dict.
        :return: Tuple of (mono audio tensor, sample rate).
        :raises RuntimeError: If the audio file cannot be decoded.
        """
        offset = entry.get("offset", 0)
        duration = entry.get("duration")

        if offset > 0 or duration is not None:
            info = sf.info(entry["audio_filepath"])
            frame_offset = int(offset * info.samplerate)
            max_frames = info.frames - frame_offset
            if max_frames <= 0:
                return torch.zeros(16000), 16000
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

        if audio.numel() == 0:
            return torch.zeros(16000), 16000

        return audio, sr

    def __getitem__(self, idx: int) -> dict:
        """
        Load and preprocess a single training sample.

        :param idx: Index into the manifest.
        :return: Dict with ``input_features`` (mel spectrogram tensor) and
            ``labels`` (token ID list).
        """
        entry = self.entries[idx]

        try:
            audio, sr = self._load_audio(entry)
        except Exception as e:
            logger.warning(f"Failed to load audio for {entry.get('audio_filepath', '?')}: {e}")
            audio = torch.zeros(16000)
            sr = 16000

        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)

        input_features = self.processor.feature_extractor(
            audio.numpy(), sampling_rate=16000, return_tensors="pt"
        ).input_features[0]

        # Set per-sample decoder prefix tokens so each label sequence starts
        # with the correct <|lang|><|transcribe|> prefix Whisper expects.
        # Each DataLoader worker has its own dataset/processor copy and
        # __getitem__ runs sequentially within a worker, so mutating the
        # tokenizer state here is safe.
        self.processor.tokenizer.set_prefix_tokens(
            language=entry["language"], task="transcribe"
        )
        labels = self.processor.tokenizer(entry["text"]).input_ids

        return {"input_features": input_features, "labels": labels}


@dataclass
class WhisperDataCollator:
    """
    Collate function for Whisper finetuning batches.

    Pads input features and label sequences, replacing label padding
    positions with ``-100`` so they are ignored by the loss.

    :param processor: A ``WhisperProcessor`` instance.
    """

    processor: WhisperProcessor

    def __call__(self, features: list[dict]) -> dict:
        """
        Collate a list of samples into a padded batch.

        :param features: List of sample dicts from ``WhisperFinetuneDataset``.
        :return: Batched dict with ``input_features`` and ``labels``.
        """

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
    """
    Trainer for Whisper models using HuggingFace's ``Seq2SeqTrainer``.

    Handles model loading, dataset creation from manifests, per-epoch
    training with WER evaluation, and checkpoint management.

    :param config: Job configuration dictionary.
    """

    def __init__(self, config: dict):
        self.config = config
        self.current_epoch = config.get("current_epoch", 0)
        self.model = None
        self.processor = None
        self.hf_trainer = None
        self.train_manifest = None
        self.val_manifest = None

    def _compute_metrics(self, pred: "transformers.EvalPrediction") -> dict:
        """
        Compute word error rate (WER) from model predictions.

        :param pred: ``EvalPrediction`` from the HuggingFace trainer.
        :return: Dict with a ``"wer"`` key.
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 padding with pad token for decoding
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id

        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = jiwer.wer(label_str, pred_str)
        return {"wer": wer}

    def setup(self, train_manifest: Path, val_manifest: Path | None) -> None:
        """
        Load the Whisper model and processor, and configure wandb.

        :param train_manifest: Path to the training manifest JSONL file.
        :param val_manifest: Path to the validation manifest JSONL file,
            or ``None`` to skip validation.
        """
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest

        import os
        os.environ.setdefault("WANDB_PROJECT", "lyricscribe-finetune")
        # Pin the wandb run ID to the experiment name and allow resume so
        # all SLURM-restarted chunks log into a single continuous wandb run
        # instead of creating a new one each time.
        os.environ["WANDB_RUN_ID"] = self.config["exp_name"]
        os.environ["WANDB_RESUME"] = "allow"

        if self.model is None:
            logger.info(f"Loading Whisper model: {self.config['base_model']}")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.config["base_model"]
            )
            self.processor = WhisperProcessor.from_pretrained(self.config["base_model"])

            # In-loop eval cannot pass per-sample language to generate(), so
            # let Whisper auto-detect language per sample. The training-time
            # labels still get the correct per-sample prefix via the dataset.
            self.model.generation_config.language = None
            self.model.generation_config.task = "transcribe"
            self.model.generation_config.forced_decoder_ids = None

            # Don't manually move to CUDA here — Seq2SeqTrainer handles
            # device placement, including multi-GPU DDP via torchrun.

    def train_one_epoch(self, target_epoch: int) -> dict:
        """
        Train the model up to *target_epoch* using HuggingFace's
        ``Seq2SeqTrainer``.

        On the first call, creates the trainer with the full LR schedule
        spanning all epochs. Subsequent calls update ``num_train_epochs``
        and resume training, preserving optimizer state.

        :param target_epoch: The epoch number to train up to (1-indexed).
        :return: Dict of metrics (train loss, eval WER if validation set
            is available).
        """
        train_dataset = WhisperFinetuneDataset(self.train_manifest, self.processor)
        eval_dataset = None
        if self.val_manifest:
            full_eval = WhisperFinetuneDataset(self.val_manifest, self.processor)
            # Eval is autoregressive generation which is much slower than
            # training (~3 hours on the full validation set vs ~3 hours per
            # training epoch). Sample a fixed subset for during-training
            # sanity checks; full eval can be run separately if needed.
            eval_subset_size = self.config.get("eval_subset_size", 200)
            if len(full_eval) > eval_subset_size:
                rng = torch.Generator().manual_seed(42)
                indices = torch.randperm(len(full_eval), generator=rng)[:eval_subset_size].tolist()
                eval_dataset = torch.utils.data.Subset(full_eval, indices)
            else:
                eval_dataset = full_eval

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
                save_strategy="steps",
                save_steps=2000,
                save_total_limit=2,
                logging_steps=100,
                remove_unused_columns=False,
                label_names=["labels"],
                weight_decay=0.01,
                max_grad_norm=1.0,
                dataloader_num_workers=16,
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

        # Resume from the latest *complete* HF step checkpoint if one
        # exists. Incomplete checkpoints (from SLURM-killed mid-save) are
        # detected by missing trainer_state.json and removed.
        hf_trainer_dir = Path(self.config["output_dir"]) / self.config["exp_name"] / "hf_trainer"
        resume_ckpt = None
        if hf_trainer_dir.exists():
            step_checkpoints = sorted(
                hf_trainer_dir.glob("checkpoint-*"),
                key=lambda p: int(p.name.split("-")[1]),
            )
            for ckpt in reversed(step_checkpoints):
                if (ckpt / "trainer_state.json").exists():
                    resume_ckpt = str(ckpt)
                    break
                else:
                    logger.warning(f"Removing incomplete checkpoint: {ckpt}")
                    shutil.rmtree(ckpt, ignore_errors=True)

        if resume_ckpt:
            logger.info(f"Resuming from HF step checkpoint: {resume_ckpt}")

        logger.info(f"Training epoch {target_epoch}...")
        self.hf_trainer.train(resume_from_checkpoint=resume_ckpt)

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
        """
        Save model and processor weights to a directory.

        Only writes on rank 0 in multi-GPU DDP.

        :param epoch: Epoch number for the directory name.
        :param checkpoint_dir: Parent directory for checkpoints.
        :return: Path to the saved checkpoint directory.
        """
        checkpoint_path = checkpoint_dir / f"epoch_{epoch}"
        if _is_main_process():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(checkpoint_path)
            self.processor.save_pretrained(checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """
        Restore model and processor from a checkpoint directory.

        :param checkpoint_path: Path to the checkpoint directory.
        :return: The epoch number extracted from the directory name.
        """
        self.model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
        self.processor = WhisperProcessor.from_pretrained(checkpoint_path)

        try:
            epoch = int(checkpoint_path.name.split("_")[1])
        except (IndexError, ValueError):
            epoch = 0

        # Seq2SeqTrainer handles device placement (including multi-GPU).
        return epoch


def create_trainer(config: dict):
    """
    Factory function that returns the appropriate trainer for the given
    architecture.

    :param config: Job configuration dictionary with an ``"architecture"`` key.
    :return: A ``NeMoTrainer`` or ``WhisperTrainer`` instance.
    :raises ValueError: If the architecture is not recognized.
    """
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

    Loads the model (or resumes from the latest checkpoint), trains for
    ``epochs_per_job`` epochs, saves a checkpoint after each epoch, and
    writes metrics to ``metrics.jsonl``.

    On ``torch.cuda.OutOfMemoryError`` the batch size is halved, the
    discovered value is persisted to ``config.json``, the trainer is
    recreated from the latest checkpoint, and the chunk is retried. This
    repeats until either training succeeds or batch size hits 1.

    :param config: Job configuration dictionary.
    :param train_manifest: Path to the training manifest JSONL file.
    :param val_manifest: Path to the validation manifest JSONL file,
        or ``None`` to skip validation.
    :param chunk_end_epoch: Hard cap on the final epoch from the chunk
        definition. Overrides ``max_epochs`` if lower.
    :return: Dict with ``"status"``, ``"start_epoch"``, ``"end_epoch"``,
        and ``"checkpoint_path"`` keys.
    """
    job_dir = Path(config["output_dir"]) / config["exp_name"]

    while True:
        try:
            return _run_training_job_inner(
                config, train_manifest, val_manifest, chunk_end_epoch, job_dir
            )
        except torch.cuda.OutOfMemoryError as e:
            current_bs = config.get("batch_size", 1)
            if current_bs <= 1:
                logger.error("OOM at batch_size=1, cannot reduce further")
                raise
            new_bs = max(1, current_bs // 2)
            logger.warning(
                f"OOM at batch_size={current_bs} ({e}). "
                f"Halving to batch_size={new_bs} and retrying."
            )
            config["batch_size"] = new_bs
            with open(job_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)
            # Force garbage collection of the old trainer/model before
            # clearing CUDA cache. Without this, the old model's GPU
            # tensors are still referenced and empty_cache() can't free
            # them, causing each retry to OOM on top of the previous.
            import gc
            gc.collect()
            torch.cuda.empty_cache()


def _run_training_job_inner(
    config: dict,
    train_manifest: Path,
    val_manifest: Path | None,
    chunk_end_epoch: int | None,
    job_dir: Path,
) -> dict:
    """
    Inner training implementation. Builds a fresh trainer, resumes from
    the latest checkpoint if any, and trains the requested epochs. Any
    OOM raised here is caught and handled by ``run_training_job``.
    """
    trainer = create_trainer(config)

    start_epoch = config.get("current_epoch", 0)
    latest_checkpoint = get_latest_checkpoint(job_dir, config)

    if latest_checkpoint and start_epoch > 0:
        logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
        try:
            loaded_epoch = trainer.load_checkpoint(latest_checkpoint)
            start_epoch = loaded_epoch
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

    logger.info(
        f"Training {epochs_to_train} epochs "
        f"(from epoch {start_epoch}, batch_size={config.get('batch_size')}), "
        "saving after each"
    )

    for epoch in range(start_epoch + 1, start_epoch + epochs_to_train + 1):
        metrics = trainer.train_one_epoch(epoch)

        last_checkpoint_path = trainer.save_checkpoint(epoch, checkpoint_dir)

        # Only rank 0 writes config/metrics to avoid races in multi-GPU DDP.
        if _is_main_process():
            config["current_epoch"] = epoch
            config["completed_epochs"] = epoch
            with open(job_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

            metrics_entry = {"epoch": epoch}
            metrics_entry.update(metrics)
            with open(job_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps(metrics_entry) + "\n")

            # Clean up mid-epoch step checkpoints after a successful epoch.
            # The .nemo epoch checkpoint is the durable artifact; step
            # checkpoints are only needed for mid-epoch SLURM timeout recovery.
            step_ckpt_dir = job_dir / "pl_step_checkpoints"
            if step_ckpt_dir.exists():
                for ckpt in step_ckpt_dir.glob("*.ckpt"):
                    ckpt.unlink()
                logger.info("Cleaned up mid-epoch step checkpoints")

        logger.info(f"Epoch {epoch} complete, checkpoint saved")

    return {
        "status": "success",
        "start_epoch": start_epoch,
        "end_epoch": start_epoch + epochs_to_train,
        "checkpoint_path": str(last_checkpoint_path),
    }
