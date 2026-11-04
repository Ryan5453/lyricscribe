import json
import logging
import gc
import math
import os
import random
import shutil
import subprocess
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path


def _configure_cuda_toolkit_for_numba() -> None:
    """
    Numba (used by NeMo's RNNT/TDT losses and CUDA-graph decoding) needs
    libnvvm.so and libcudart.so at import time. On systems without a
    system CUDA Toolkit (e.g. dev boxes running only the driver), those
    ship with the pip-installed ``nvidia-cuda-nvcc-cu12`` /
    ``nvidia-cuda-runtime-cu12`` packages — but numba only picks them up
    if ``CUDA_HOME`` and ``LD_LIBRARY_PATH`` point at their locations.

    This must run *before* ``import nemo`` — NeMo lazily imports numba
    on first RNNT forward, and we set these at module-import time so the
    CLI entry point configures them before NeMo loads.

    On SLURM/production where ``module load cuda/…`` already set up a
    system toolkit, env vars are left alone (the toolkit wins over pip
    shims via existing LD_LIBRARY_PATH).
    """
    try:
        import nvidia  # noqa: F401
    except ImportError:
        return

    site = Path(__import__("nvidia").__path__[0])
    nvcc_root = site / "cuda_nvcc"
    cudart_lib = site / "cuda_runtime" / "lib"

    if not nvcc_root.exists():
        return

    os.environ.setdefault("CUDA_HOME", str(nvcc_root))

    nvvm_lib64 = nvcc_root / "nvvm" / "lib64"
    nvvm_unversioned = nvvm_lib64 / "libnvvm.so"
    if nvvm_unversioned.exists():
        try:
            soname = subprocess.check_output(
                ["objdump", "-p", str(nvvm_unversioned)], text=True
            )
            for line in soname.splitlines():
                if "SONAME" in line:
                    _, name = line.strip().rsplit(None, 1)
                    versioned = nvvm_lib64 / name
                    if not versioned.exists():
                        versioned.symlink_to(nvvm_unversioned.name)
                    break
        except (OSError, subprocess.CalledProcessError):
            pass

    if cudart_lib.exists():
        versioned = cudart_lib / "libcudart.so.12"
        if versioned.exists():
            target_lib64 = nvcc_root / "lib64"
            target_lib64.mkdir(exist_ok=True)
            for name in ("libcudart.so", "libcudart.so.12"):
                link = target_lib64 / name
                if not link.exists():
                    try:
                        link.symlink_to(versioned)
                    except OSError:
                        pass


_configure_cuda_toolkit_for_numba()

import jiwer
import nemo.collections.asr as nemo_asr
import numpy as np
from lhotse import CutSet, MonoCut
from lhotse.cut import Cut, MixedCut
from lhotse.dataset.collation import collate_vectors
from nemo.collections.asr.data.audio_to_text_lhotse_prompted import (
    PromptedAudioToTextLhotseDataset,
    PromptedAudioToTextMiniBatch,
    _drop_in_memory_data,
)
from nemo.collections.common.data import apply_prompt_format_fn
from nemo.collections.common.data import prompt_fn as _prompt_fn
from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config
from omegaconf import OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
import soundfile as sf
import torch
import torchaudio
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.models.whisper.modeling_whisper import shift_tokens_right

from lyricscribe.finetune.config import get_latest_checkpoint
from lyricscribe.finetune.qwen import Qwen3ASRTrainer, Qwen3OmniLoraTrainer

logger = logging.getLogger(__name__)


def _is_main_process() -> bool:
    """
    Return True if this is rank 0 (or single-GPU). Used to guard
    file writes (checkpoints, metrics, config) in multi-GPU DDP.

    :return: ``True`` on rank 0.
    """
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def _build_mono_averaging_canary_dataset_cls():
    """
    Build a subclass of NeMo's PromptedAudioToTextLhotseDataset that:

    1. Downmixes multi-channel audio to mono by *averaging* channels
       (rather than picking one or summing-with-clip). Canary rejects
       MultiCut at prompt-format time, and its dataloader has no built-in
       option to average channels on the fly, so we do it here.
    2. Delegates prompt formatting to the (patched) registry, so the
       stereo→mono audio averaging stays orthogonal from the prompt
       logic. See ``_patch_prompt_format_for_multi_channel`` for the
       MultiCut-tolerating prompt-function wrapper.

    Returns the class so importing NeMo stays lazy.

    :return: The dataset class.
    """
    class MonoAveragingPromptedDataset(PromptedAudioToTextLhotseDataset):
        """
        Load audio, average stereo channels, and format prompts as usual.

        Keeps the original cuts (which may already carry pre-formatted
        prompt attributes from NeMo's ``pretokenize`` step) — the prompt
        format function is separately patched at the registry level to
        accept MultiCut, so we don't need to rebuild the cuts here.
        """

        def __getitem__(self, cuts: CutSet) -> PromptedAudioToTextMiniBatch:
            """
            Load and channel-average each cut's audio and collate the prompts.

            :param cuts: Cuts in the batch.
            :return: The collated mini-batch.
            """
            audios: list[torch.Tensor] = []
            audio_lens_list: list[int] = []
            for cut in cuts:
                arr = cut.load_audio()
                if arr.ndim == 2:
                    if arr.shape[0] > 1:
                        arr = arr.mean(axis=0)
                    else:
                        arr = arr.squeeze(0)
                audios.append(torch.from_numpy(np.ascontiguousarray(arr)).float())
                audio_lens_list.append(arr.shape[-1])

            audio_lens = torch.tensor(audio_lens_list, dtype=torch.int32)
            audio = collate_vectors(audios, padding_value=0.0)

            attrs = ("input_ids", "context_ids", "answer_ids")
            pre_formatted = all(hasattr(c, a) for c in cuts for a in attrs)
            if pre_formatted:
                prompts_with_answers, prompts, answers = zip(
                    *((c.input_ids, c.context_ids, c.answer_ids) for c in cuts)
                )
            else:
                formatted = [apply_prompt_format_fn(cut, self.prompt) for cut in cuts]
                prompts_with_answers = [ex["input_ids"] for ex in formatted]
                prompts = [ex["context_ids"] for ex in formatted]
                answers = [ex["answer_ids"] for ex in formatted]

            transcript, transcript_lens = self._collate_tokens(answers)
            prompts_with_answers, prompts_with_answers_lens = self._collate_tokens(
                prompts_with_answers
            )
            prompts, prompt_lens = self._collate_tokens(prompts)

            return PromptedAudioToTextMiniBatch(
                audio=audio,
                audio_lens=audio_lens,
                transcript=transcript,
                transcript_lens=transcript_lens,
                prompt=prompts,
                prompt_lens=prompt_lens,
                prompted_transcript=prompts_with_answers,
                prompted_transcript_lens=prompts_with_answers_lens,
                cuts=_drop_in_memory_data(cuts),
            )

    return MonoAveragingPromptedDataset


def _patch_prompt_format_for_multi_channel() -> None:
    """
    Wrap every registered prompt-format function so it tolerates MultiCut
    by first converting to a channel-0 MonoCut view (audio gets averaged
    in the dataset's ``__getitem__`` — this conversion only changes *type*
    so the prompt formatter accepts the cut).

    NeMo's Lhotse dataloader tokenizes cuts via
    ``CutSet.map(tokenize_with_prompt)`` *before* our dataset runs (when
    ``pretokenize=True``), so patching the prompt format functions at the
    registry level is the earliest hook point. Idempotent: attaches a
    ``_lyricscribe_patched`` marker on each wrapper.
    """
    def _to_mono_view(cut):
        """
        View a multi-channel cut as a MonoCut so prompt formatters accept it.

        :param cut: Lhotse cut.
        :return: The cut itself if mono or mixed, else a MonoCut of its first channel.
        """
        if isinstance(cut, (MonoCut, MixedCut)):
            return cut
        if not hasattr(cut, "num_channels") or cut.num_channels <= 1:
            return cut
        channel = cut.channel[0] if isinstance(cut.channel, (list, tuple)) else cut.channel
        if not isinstance(channel, int):
            channel = 0
        return MonoCut(
            id=cut.id,
            start=cut.start,
            duration=cut.duration,
            channel=channel,
            recording=cut.recording,
            supervisions=cut.supervisions,
            custom=getattr(cut, "custom", None),
        )

    for key, existing in list(_prompt_fn.PROMPT_FORMAT_FNS.items()):
        if isinstance(key, tuple):
            example_type = key[0]
        else:
            example_type = key
        if not (isinstance(example_type, type) and issubclass(example_type, Cut)):
            continue
        if getattr(existing, "_lyricscribe_patched", False):
            continue

        def _make_wrapper(fn):
            """
            Wrap one prompt-format function to accept multi-channel cuts.

            :param fn: Registered prompt-format function.
            :return: The marked wrapper.
            """
            def wrapped(example, prompt):
                """
                Format *example* through its mono view.

                :param example: Cut to format.
                :param prompt: Prompt formatter.
                :return: Whatever *fn* returns.
                """
                return fn(_to_mono_view(example), prompt)

            wrapped._lyricscribe_patched = True  # type: ignore[attr-defined]
            wrapped.__name__ = getattr(fn, "__name__", "wrapped_prompt_fn")
            return wrapped

        _prompt_fn.PROMPT_FORMAT_FNS[key] = _make_wrapper(existing)


def _install_mono_averaging_dataloader(model) -> None:
    """
    Override ``model._setup_dataloader_from_config`` so the Lhotse dataloader
    uses the mono-averaging dataset above. Must be called before
    ``setup_training_data`` / ``setup_validation_data``.

    :param model: Canary model to patch.
    """
    _patch_prompt_format_for_multi_channel()

    dataset_cls = _build_mono_averaging_canary_dataset_cls()

    def _setup(self, config):
        """
        Build the Lhotse dataloader with the mono-averaging dataset.

        :param config: Dataset config (``train_ds`` / ``validation_ds``).
        :return: The dataloader.
        """
        assert config.get("use_lhotse", False), (
            "Canary requires Lhotse dataloading (use_lhotse=True)."
        )
        return get_lhotse_dataloader_from_config(
            config,
            global_rank=config.get("global_rank", self.global_rank),
            world_size=config.get("world_size", self.world_size),
            dataset=dataset_cls(
                tokenizer=self.tokenizer,
                prompt=self.prompt,
            ),
            tokenizer=self.tokenizer,
        )

    model._setup_dataloader_from_config = types.MethodType(_setup, model)


class _SkipNaNOrInfLossCallback(Callback):
    """
    Replace ``training_step``'s loss with a harmless zero when it is
    non-finite (NaN/Inf). Without this, a single bad batch poisons the
    optimizer and every subsequent step diverges.

    Uses the same DDP-safe pattern as ``_SkipBadBatchCallback``: we
    all-reduce the "skip" decision so every rank agrees, and return a
    zero that still touches every trainable parameter so gradient hooks
    fire uniformly across ranks.
    """

    def setup(self, trainer, pl_module, stage=None):
        """
        Wrap ``pl_module.training_step`` with the non-finite-loss guard.

        :param trainer: Lightning trainer.
        :param pl_module: Module being trained.
        :param stage: Lightning stage name (unused).
        """
        original_step = pl_module.training_step

        def safe_step(batch, batch_idx):
            """
            Run the original step, swapping a non-finite loss on any rank for a zero loss.

            :param batch: Training batch.
            :param batch_idx: Batch index.
            :return: The original output, or a zero loss over all trainable parameters.
            """
            skip_flag = torch.zeros(1, device=pl_module.device)
            out = original_step(batch, batch_idx)

            if isinstance(out, dict):
                loss = out.get("loss")
            else:
                loss = out

            if loss is None or not torch.isfinite(loss).all():
                skip_flag[0] = 1.0

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(skip_flag, op=torch.distributed.ReduceOp.MAX)

            if skip_flag.item() > 0:
                logger.warning(
                    f"Skipping batch {batch_idx}: non-finite loss "
                    f"(loss={loss.item() if loss is not None else 'None'})"
                )
                return sum(p.sum() for p in pl_module.parameters() if p.requires_grad) * 0.0

            return out

        pl_module.training_step = safe_step


class _SkipBadBatchCallback(Callback):
    """
    Skip training batches that crash the RNNT/TDT loss.

    NeMo's RNNT loss raises ``RuntimeError: Invalid parameter`` when a
    sample has zero encoder frames after subsampling (e.g. audio at the
    stated manifest offset is too short or corrupted). Raw signal_len
    may be non-zero, so we can't filter pre-forward. Instead we wrap
    the model's training_step to catch the error and return a zero loss.

    Under DDP, a naive catch causes an NCCL ALLREDUCE deadlock: if one
    rank fails and returns a detached zero tensor while other ranks
    succeed and fire their gradient hooks normally, the failing rank
    never participates in the gradient sync and other ranks time out.
    We fix this by (1) all-reducing the skip decision so every rank
    agrees, and (2) returning a zero loss that still touches every
    trainable parameter so DDP gradient hooks fire uniformly.
    """

    def setup(self, trainer, pl_module, stage=None):
        """
        Wrap ``pl_module.training_step`` with the bad-batch guard.

        :param trainer: Lightning trainer.
        :param pl_module: Module being trained.
        :param stage: Lightning stage name (unused).
        """
        original_step = pl_module.training_step

        def safe_training_step(batch, batch_idx):
            """
            Run the original step, swapping an RNNT invalid-input error on any rank
            for a zero loss.

            :param batch: Training batch.
            :param batch_idx: Batch index.
            :return: The original loss, or a zero loss over all trainable parameters.
            :raises RuntimeError: For any other error from the original step.
            """
            skip_flag = torch.zeros(1, device=pl_module.device)
            loss = None
            local_err: str | None = None
            try:
                loss = original_step(batch, batch_idx)
            except RuntimeError as e:
                if "Invalid parameter" in str(e) or "working space memory" in str(e):
                    local_err = str(e)
                    skip_flag[0] = 1.0
                else:
                    raise

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(skip_flag, op=torch.distributed.ReduceOp.MAX)

            if skip_flag.item() > 0:
                if local_err is not None:
                    logger.warning(
                        f"Skipping batch {batch_idx}: RNNT loss got invalid input "
                        f"(likely zero encoder frames). Error: {local_err}"
                    )
                return sum(p.sum() for p in pl_module.parameters() if p.requires_grad) * 0.0

            return loss

        pl_module.training_step = safe_training_step


def _count_manifest_lines(manifest_path: Path) -> int:
    """
    Count the number of lines in a manifest JSONL file.

    :param manifest_path: Path to the manifest file.
    :return: Number of lines.
    """
    with open(manifest_path) as f:
        return sum(1 for _ in f)


def _sum_manifest_durations(manifest_path: Path) -> float:
    """
    Sum the ``duration`` field over a manifest JSONL file, in seconds.

    :param manifest_path: Path to the manifest file.
    :return: Total audio duration in seconds.
    """
    total = 0.0
    with open(manifest_path) as f:
        for line in f:
            try:
                total += float(json.loads(line).get("duration") or 0.0)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return total


class NeMoTrainer:
    """
    Trainer for NeMo ASR models (Parakeet, Canary) using PyTorch Lightning.

    Handles model loading, data configuration, per-epoch training, and
    checkpoint management. Designed to be called once per chunk of epochs
    by the orchestration system.

    :param config: Job configuration dictionary.
    """

    def __init__(self, config: dict):
        """
        Store the config; the model loads in :meth:`setup`.

        :param config: Job configuration dictionary.
        """
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
            # cpu: NeMo's default puts every DDP rank's copy on cuda:0 first.
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.config["base_model"],
                map_location="cpu",
            )
            self.model = self.model.cpu()

        # CUDA graphs cause illegal memory accesses on H200.
        decoding = getattr(self.model, "decoding", None)
        inner = getattr(decoding, "decoding", None)
        computer = getattr(inner, "decoding_computer", None)
        if computer is not None and hasattr(computer, "cuda_graphs_mode"):
            computer.cuda_graphs_mode = None

        cfg = self.model.cfg
        OmegaConf.set_struct(cfg, False)

        is_canary = self.config["architecture"] == "canary"

        if is_canary:
            # Pretrained configs ship None for these (NeMo #14816).
            cfg.train_ds.num_buckets = cfg.train_ds.get("num_buckets") or 30
            cfg.train_ds.min_tps = cfg.train_ds.get("min_tps") if cfg.train_ds.get("min_tps") is not None else -1
            cfg.train_ds.max_tps = cfg.train_ds.get("max_tps") if cfg.train_ds.get("max_tps") is not None else float("inf")
            cfg.train_ds.batch_duration = self.config.get("batch_duration", 600)
            cfg.train_ds.batch_size = None

            _install_mono_averaging_dataloader(self.model)

        else:
            cfg.train_ds.use_lhotse = False
            cfg.train_ds.channel_selector = "average"
            cfg.train_ds.batch_size = self.config["batch_size"]

        cfg.train_ds.manifest_filepath = str(train_manifest)
        cfg.train_ds.min_duration = 0.1
        cfg.train_ds.max_duration = self.config.get("max_duration", 240.0)

        if self.config.get("freeze_encoder", False):
            frozen = 0
            for p in self.model.encoder.parameters():
                p.requires_grad = False
                frozen += 1
            logger.info(f"Froze {frozen} encoder parameter tensors")

        if hasattr(cfg, "loss") and cfg.loss is not None:
            cfg.loss.reduction = "mean_batch"
        cfg.train_ds.shuffle = True
        cfg.train_ds.num_workers = self.config.get("num_workers", 4)
        cfg.train_ds.pin_memory = True

        if val_manifest:
            if is_canary:
                cfg.validation_ds.num_buckets = cfg.validation_ds.get("num_buckets") or 30
                cfg.validation_ds.min_tps = cfg.validation_ds.get("min_tps") if cfg.validation_ds.get("min_tps") is not None else -1
                cfg.validation_ds.max_tps = cfg.validation_ds.get("max_tps") if cfg.validation_ds.get("max_tps") is not None else float("inf")
                cfg.validation_ds.batch_duration = self.config.get("batch_duration", 600)
                cfg.validation_ds.batch_size = None
            else:
                cfg.validation_ds.use_lhotse = False
                cfg.validation_ds.channel_selector = "average"
                cfg.validation_ds.batch_size = self.config["batch_size"]
            cfg.validation_ds.manifest_filepath = str(val_manifest)
            cfg.validation_ds.min_duration = 0.1
            cfg.validation_ds.max_duration = self.config.get("max_duration", 240.0)
            cfg.validation_ds.num_workers = self.config.get("num_workers", 4)
            cfg.validation_ds.pin_memory = True

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

        # setup_*_data can re-enable struct mode on sub-configs.
        OmegaConf.set_struct(cfg, False)

        cfg.optim.name = "adamw"
        cfg.optim.lr = self.config["learning_rate"]
        cfg.optim.weight_decay = 0.01
        cfg.optim.sched.name = "CosineAnnealing"

        steps_per_epoch = self._estimate_steps_per_epoch()
        total_steps = self.config["max_epochs"] * steps_per_epoch

        warmup_epochs = self.config.get("warmup_epochs", 5)
        cfg.optim.sched.warmup_steps = warmup_epochs * steps_per_epoch
        cfg.optim.sched.max_steps = total_steps

    def _estimate_steps_per_epoch(self) -> int:
        """
        Estimate optimizer steps per epoch from the training manifest:
        total duration / ``batch_duration`` for Canary's duration-based
        Lhotse batching, sample count / ``batch_size`` otherwise.

        :return: Steps per epoch (at least 1).
        """
        if self.config["architecture"] == "canary":
            total_duration = _sum_manifest_durations(self.train_manifest)
            batch_duration = float(self.config.get("batch_duration") or 600.0)
            return max(1, math.ceil(total_duration / batch_duration))
        num_train_samples = _count_manifest_lines(self.train_manifest)
        batch_size = int(self.config.get("batch_size") or 1)
        return max(1, math.ceil(num_train_samples / max(batch_size, 1)))

    def train_one_epoch(self, target_epoch: int) -> dict:
        """
        Train the model up to *target_epoch* using PyTorch Lightning.

        On the first call, creates the PL Trainer. Subsequent calls reuse
        the same Trainer to preserve optimizer state across epochs.

        :param target_epoch: The epoch number to train up to (1-indexed).
        :return: Dict of metrics from ``trainer.callback_metrics``.
        """
        if self.trainer is None:
            uses_lhotse = self.model.cfg.train_ds.get("use_lhotse", False)

            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            if num_gpus > 1:
                logger.info(f"Multi-GPU detected: {num_gpus} devices, using DDP")

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

            # save_on_train_epoch_end is a no-op on a checkpoint with every_n_train_steps.
            epoch_checkpoint = ModelCheckpoint(
                dirpath=str(self._step_ckpt_dir),
                every_n_epochs=1,
                save_top_k=-1,
                filename="step-{step:08d}",
                save_on_train_epoch_end=True,
            )

            callbacks = [
                step_checkpoint,
                epoch_checkpoint,
                _SkipBadBatchCallback(),
                _SkipNaNOrInfLossCallback(),
            ]

            trainer_kwargs = dict(
                max_epochs=self.config["max_epochs"],
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=num_gpus,
                strategy="ddp" if num_gpus > 1 else "auto",
                precision="bf16-mixed" if torch.cuda.is_available() else 32,
                gradient_clip_val=1.0,
                enable_progress_bar=True,
                logger=False,
                enable_checkpointing=True,
                callbacks=callbacks,
                log_every_n_steps=1,
            )

            if uses_lhotse:
                # Lhotse iterable datasets don't have __len__.
                trainer_kwargs["use_distributed_sampler"] = False
                trainer_kwargs["limit_train_batches"] = self._estimate_steps_per_epoch()

            self.trainer = pl.Trainer(**trainer_kwargs)

        self.trainer.fit_loop.max_epochs = target_epoch

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

        if metrics:
            logger.info(f"Epoch {target_epoch} metrics: {metrics}")

        return metrics

    def train_n_steps(self, n: int) -> None:
        """
        Run exactly *n* training steps against the already-configured
        training dataloader. Used by the VRAM probe so the memory profile
        matches production 1:1 — same PL strategy, DDP rank count,
        precision, gradient clipping, bad-batch skipping callbacks, and
        Lhotse vs. standard-dataloader dispatch — minus the
        epoch-level bookkeeping, ModelCheckpoint, and resume logic.

        :param n: Number of training steps to run.
        """
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        uses_lhotse = self.model.cfg.train_ds.get("use_lhotse", False)

        callbacks = [
            _SkipBadBatchCallback(),
            _SkipNaNOrInfLossCallback(),
        ]

        trainer_kwargs = dict(
            max_epochs=1,
            max_steps=n,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=num_gpus,
            strategy="ddp" if num_gpus > 1 else "auto",
            precision="bf16-mixed" if torch.cuda.is_available() else 32,
            gradient_clip_val=1.0,
            enable_progress_bar=False,
            logger=False,
            enable_checkpointing=False,
            callbacks=callbacks,
            log_every_n_steps=1,
        )
        if uses_lhotse:
            trainer_kwargs["use_distributed_sampler"] = False
            trainer_kwargs["limit_train_batches"] = n

        probe_trainer = pl.Trainer(**trainer_kwargs)
        probe_trainer.fit(self.model)

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

    Labels follow Whisper's dual-mode format: with probability
    ``timestamp_prob`` (when the entry carries ``segments``) the label is
    built in timestamp mode, otherwise in ``<|notimestamps|>`` mode.
    Training only one mode unlearns the other; long-form (>30s) decoding
    requires timestamps.

    :param manifest_path: Path to the JSONL manifest file.
    :param processor: A ``WhisperProcessor`` instance for feature extraction
        and tokenization.
    :param timestamp_prob: Probability of building a timestamp-mode label
        for entries that have ``segments``. Use 0.0 for eval datasets.
    """

    MAX_LABEL_TOKENS = 448
    TIME_PRECISION = 0.02  # seconds per timestamp token step

    def __init__(
        self,
        manifest_path: Path,
        processor: WhisperProcessor,
        timestamp_prob: float = 0.0,
    ):
        """
        Read the manifest and resolve the timestamp token range.

        :param manifest_path: Path to the JSONL manifest file.
        :param processor: ``WhisperProcessor`` for features and tokenization.
        :param timestamp_prob: Probability of a timestamp-mode label for
            entries that have ``segments``.
        """
        self.processor = processor
        self.timestamp_prob = timestamp_prob
        self.entries = []
        with open(manifest_path) as f:
            for line in f:
                self.entries.append(json.loads(line))
        tokenizer = processor.tokenizer
        self._notimestamps_id = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
        self._timestamp_begin = self._notimestamps_id + 1
        # <|0.00|> .. <|30.00|> — 1501 tokens at 0.02s resolution.
        self._max_timestamp_index = 1500

    def _timestamp_token(self, seconds: float) -> int:
        """
        Token id for the timestamp nearest *seconds*, clamped to 0–30 s.

        :param seconds: Window-relative time.
        :return: Timestamp token id.
        """
        idx = round(seconds / self.TIME_PRECISION)
        return self._timestamp_begin + max(0, min(self._max_timestamp_index, idx))

    def _build_timestamp_labels(self, entry: dict) -> list[int] | None:
        """
        Build timestamp-mode label ids from the entry's ``segments``.

        :param entry: Manifest entry with ``segments``.
        :return: Label ids, or ``None`` if they exceed the decoder cap.
        """
        tokenizer = self.processor.tokenizer
        ids = list(tokenizer.prefix_tokens)
        prev_end = 0.0
        for start_s, end_s, seg_text in entry["segments"]:
            start_s = max(float(start_s), prev_end)
            end_s = max(float(end_s), start_s)
            ids.append(self._timestamp_token(start_s))
            # Leading space matches Whisper's native segment tokenization.
            ids.extend(tokenizer(" " + seg_text, add_special_tokens=False).input_ids)
            ids.append(self._timestamp_token(end_s))
            prev_end = end_s
        ids.append(tokenizer.eos_token_id)
        if len(ids) > self.MAX_LABEL_TOKENS:
            return None
        return ids

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

        # predict_timestamps persists on the tokenizer, so set it on every call.
        labels = None
        use_timestamps = (
            self.timestamp_prob > 0
            and entry.get("segments")
            and random.random() < self.timestamp_prob
        )
        if use_timestamps:
            self.processor.tokenizer.set_prefix_tokens(
                language=entry["language"], task="transcribe",
                predict_timestamps=True,
            )
            labels = self._build_timestamp_labels(entry)
        if labels is None:
            self.processor.tokenizer.set_prefix_tokens(
                language=entry["language"], task="transcribe",
                predict_timestamps=False,
            )
            text = " " + entry["text"] if entry["text"] else entry["text"]
            labels = self.processor.tokenizer(text).input_ids

        return {"input_features": input_features, "labels": labels}


@dataclass
class WhisperDataCollator:
    """
    Collate function for Whisper finetuning batches.

    Pads input features and label sequences, replacing label padding
    positions with ``-100`` so they are ignored by the loss.

    :param processor: A ``WhisperProcessor`` instance.
    :param decoder_start_token_id: Whisper SOT token id used to seed
        teacher-forcing inputs.
    :param pad_token_id: Token id used to fill shifted-out positions.
    """

    processor: WhisperProcessor
    decoder_start_token_id: int
    pad_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        """
        Collate a list of samples into a padded batch.

        :param features: List of sample dicts from ``WhisperFinetuneDataset``.
        :return: Batched dict with ``input_features``, ``labels``, and
            ``decoder_input_ids``.
        """

        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        label_ids = labels_batch["input_ids"]
        attention_mask = labels_batch.attention_mask

        # shift_tokens_right prepends SOT again.
        if (label_ids[:, 0] == self.decoder_start_token_id).all():
            label_ids = label_ids[:, 1:]
            attention_mask = attention_mask[:, 1:]

        # Label smoothing makes the Trainer pop labels, so the model can't shift them itself.
        decoder_input_ids = shift_tokens_right(
            label_ids,
            self.pad_token_id,
            self.decoder_start_token_id,
        )

        labels = label_ids.masked_fill(attention_mask.ne(1), -100)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids
        return batch


class _StopAndSaveAfterEpochCallback(TrainerCallback):
    """
    Stop training (and save a checkpoint) once ``target_epoch`` epochs
    are complete. Keeping ``num_train_epochs`` at ``max_epochs`` lets
    the LR schedule span the whole run; the save gives the next
    ``train()`` call an epoch-boundary checkpoint to resume
    optimizer/scheduler state from.
    """

    def __init__(self):
        """
        Start with no target epoch; :meth:`WhisperTrainer.train_one_epoch` sets it.
        """
        self.target_epoch: int | None = None

    def on_epoch_end(self, args, state, control, **kwargs):
        """
        Request a save and stop once ``target_epoch`` is reached.

        :param args: Training arguments.
        :param state: Trainer state.
        :param control: Trainer control flags.
        :param kwargs: Unused callback extras.
        :return: The updated *control*.
        """
        if (
            self.target_epoch is not None
            and state.epoch is not None
            and state.epoch >= self.target_epoch - 1e-3
        ):
            control.should_save = True
            control.should_training_stop = True
        return control


class WhisperTrainer:
    """
    Trainer for Whisper models using HuggingFace's ``Seq2SeqTrainer``.

    Handles model loading, dataset creation from manifests, per-epoch
    training with WER evaluation, and checkpoint management.

    :param config: Job configuration dictionary.
    """

    def __init__(self, config: dict):
        """
        Store the config; the model loads in :meth:`setup`.

        :param config: Job configuration dictionary.
        """
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

        pad_id = self.processor.tokenizer.pad_token_id
        vocab_size = self.processor.tokenizer.vocab_size

        label_ids[label_ids == -100] = pad_id
        pred_ids[pred_ids == -100] = pad_id
        pred_ids[(pred_ids < 0) | (pred_ids >= vocab_size)] = pad_id

        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # jiwer rejects empty references (instrumental windows).
        paired = [(r, h) for r, h in zip(label_str, pred_str) if r.strip()]
        if not paired:
            return {"wer": 0.0}
        refs, hyps = zip(*paired)
        wer = jiwer.wer(list(refs), list(hyps))
        return {"wer": wer}

    def setup(self, train_manifest: Path, val_manifest: Path | None) -> None:
        """
        Load the Whisper model and processor.

        :param train_manifest: Path to the training manifest JSONL file.
        :param val_manifest: Path to the validation manifest JSONL file,
            or ``None`` to skip validation.
        """
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest

        if self.model is None:
            logger.info(f"Loading Whisper model: {self.config['base_model']}")
            # fp16 AMP needs fp32 master weights; some checkpoints are stored in fp16.
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.config["base_model"], dtype=torch.float32
            )
            self.processor = WhisperProcessor.from_pretrained(self.config["base_model"])

            # In-loop eval can't pass per-sample language to generate().
            self.model.generation_config.language = None
            self.model.generation_config.task = "transcribe"
            self.model.generation_config.forced_decoder_ids = None

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
        train_dataset = WhisperFinetuneDataset(
            self.train_manifest,
            self.processor,
            timestamp_prob=self.config.get("whisper_timestamp_prob", 0.5),
        )
        eval_dataset = None
        if self.val_manifest:
            eval_dataset = WhisperFinetuneDataset(
                self.val_manifest, self.processor, timestamp_prob=0.0
            )

        job_dir = Path(self.config["output_dir"]) / self.config["exp_name"]

        # ``warmup_steps`` reads values < 1 as a fraction of all steps and
        # >= 1 as an absolute step count, so clamp below 1.
        warmup_ratio = min(
            0.999,
            self.config.get("warmup_epochs", 5) / max(self.config["max_epochs"], 1),
        )

        if self.hf_trainer is None:
            logger.info(
                f"Whisper training: {len(train_dataset)} samples. "
                "Note: audio is truncated to 30 seconds by Whisper's feature extractor."
            )

            training_args = Seq2SeqTrainingArguments(
                output_dir=str(job_dir / "hf_trainer"),
                per_device_train_batch_size=self.config["batch_size"],
                gradient_accumulation_steps=self.config.get("grad_accum", 1),
                learning_rate=self.config["learning_rate"],
                warmup_steps=warmup_ratio,
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
                label_smoothing_factor=0.1,
                dataloader_num_workers=16,
                dataloader_persistent_workers=True,
                dataloader_prefetch_factor=4,
                dataloader_pin_memory=True,
                predict_with_generate=True,
                report_to="none",
                run_name=self.config["exp_name"],
                torch_compile=self.config.get("torch_compile", False),
            )

            self._stop_callback = _StopAndSaveAfterEpochCallback()
            self.hf_trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=self._compute_metrics,
                data_collator=WhisperDataCollator(
                    processor=self.processor,
                    decoder_start_token_id=self.model.config.decoder_start_token_id,
                    pad_token_id=self.model.config.pad_token_id,
                ),
                processing_class=self.processor,
                callbacks=[self._stop_callback],
            )

        else:
            self.hf_trainer.train_dataset = train_dataset
            if eval_dataset is not None:
                self.hf_trainer.eval_dataset = eval_dataset

        self._stop_callback.target_epoch = target_epoch

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

    def train_n_steps(self, n: int) -> None:
        """
        Run exactly *n* training steps against the training manifest.
        Used by the VRAM probe so the memory profile matches production
        1:1 — same HF ``Seq2SeqTrainer``, DDP rank count (picked up from
        torchrun env vars), precision, gradient clipping, collator —
        minus checkpointing, evaluation, logging-to-disk.

        :param n: Number of training steps to run.
        """
        train_dataset = WhisperFinetuneDataset(
            self.train_manifest,
            self.processor,
            timestamp_prob=self.config.get("whisper_timestamp_prob", 0.5),
        )

        tmp_dir = tempfile.mkdtemp(prefix="lyricscribe_probe_whisper_")
        try:
            training_args = Seq2SeqTrainingArguments(
                output_dir=tmp_dir,
                per_device_train_batch_size=self.config["batch_size"],
                gradient_accumulation_steps=self.config.get("grad_accum", 1),
                learning_rate=self.config["learning_rate"],
                warmup_steps=0,
                max_steps=n,
                num_train_epochs=1,
                fp16=torch.cuda.is_available(),
                save_strategy="no",
                logging_steps=1,
                remove_unused_columns=False,
                label_names=["labels"],
                weight_decay=0.01,
                max_grad_norm=1.0,
                dataloader_num_workers=2,
                dataloader_pin_memory=True,
                predict_with_generate=False,
                report_to="none",
                run_name="lyricscribe_probe",
            )

            hf_trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=WhisperDataCollator(
                    processor=self.processor,
                    decoder_start_token_id=self.model.config.decoder_start_token_id,
                    pad_token_id=self.model.config.pad_token_id,
                ),
                processing_class=self.processor,
            )
            hf_trainer.train()
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

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

            # save_pretrained() can write eos_token_id as a one-element list;
            # SuppressTokensAtBeginLogitsProcessor needs an int.
            gen_config_path = checkpoint_path / "generation_config.json"
            if gen_config_path.exists():
                with open(gen_config_path) as f:
                    gc = json.load(f)
                eos = gc.get("eos_token_id")
                if isinstance(eos, list) and len(eos) == 1:
                    gc["eos_token_id"] = eos[0]
                    with open(gen_config_path, "w") as f:
                        json.dump(gc, f, indent=2)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """
        Restore model and processor from a checkpoint directory.

        :param checkpoint_path: Path to the checkpoint directory.
        :return: The epoch number extracted from the directory name.
        """
        self.model = WhisperForConditionalGeneration.from_pretrained(
            checkpoint_path, dtype=torch.float32
        )
        self.processor = WhisperProcessor.from_pretrained(checkpoint_path)

        try:
            epoch = int(checkpoint_path.name.split("_")[1])
        except (IndexError, ValueError):
            epoch = 0

        return epoch


def create_trainer(config: dict):
    """
    Factory function that returns the appropriate trainer for the given
    architecture.

    :param config: Job configuration dictionary with an ``"architecture"`` key.
    :return: A ``NeMoTrainer``, ``WhisperTrainer``, ``Qwen3ASRTrainer`` or
        ``Qwen3OmniLoraTrainer`` instance.
    :raises ValueError: If the architecture is not recognized.
    """
    architecture = config.get("architecture", "parakeet")

    if architecture in ["canary", "parakeet"]:
        return NeMoTrainer(config)
    elif architecture == "whisper":
        return WhisperTrainer(config)
    elif architecture == "qwen3_asr":
        return Qwen3ASRTrainer(config)
    elif architecture == "qwen3_omni":
        return Qwen3OmniLoraTrainer(config)
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

    On ``torch.cuda.OutOfMemoryError`` the per-step workload is halved
    (``batch_size`` for Whisper/Parakeet, ``batch_duration`` for Canary),
    the discovered value is persisted to ``config.json``, the trainer is
    recreated from the latest checkpoint, and the chunk is retried. This
    repeats until either training succeeds or the workload can no longer
    be reduced. Qwen architectures re-raise the OOM instead.

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
                config, train_manifest, val_manifest, chunk_end_epoch, job_dir,
            )
        except torch.cuda.OutOfMemoryError as e:
            architecture = str(config.get("architecture", "")).lower()
            if architecture in ("qwen3_asr", "qwen3_omni"):
                # Halving per rank desyncs DDP step counts and deadlocks NCCL.
                logger.error(f"OOM at batch_size={config.get('batch_size')}: {e}")
                raise
            if architecture == "canary":
                current_duration = float(config.get("batch_duration") or 600.0)
                if current_duration <= 1.0:
                    logger.error("OOM at batch_duration=1s, cannot reduce further")
                    raise
                new_duration = max(1.0, current_duration / 2.0)
                logger.warning(
                    f"OOM at batch_duration={current_duration:g}s ({e}). "
                    f"Halving to batch_duration={new_duration:g}s and retrying."
                )
                config["batch_duration"] = new_duration
            else:
                current_bs = int(config.get("batch_size") or 1)
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

    :param config: Job configuration dictionary.
    :param train_manifest: Path to the training manifest JSONL file.
    :param val_manifest: Path to the validation manifest JSONL file,
        or ``None`` to skip validation.
    :param chunk_end_epoch: Hard cap on the final epoch, or ``None``.
    :param job_dir: Experiment directory.
    :return: Status dict, as from :func:`run_training_job`.
    """
    # cuDNN SDPA rebuilds its graph for every padded batch shape.
    torch.backends.cuda.enable_cudnn_sdp(False)
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

        if _is_main_process():
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
