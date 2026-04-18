import json
import logging
import math
import os
import shutil
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

    # Numba's find_lib requires versioned sonames (``libnvvm.so.4``) but
    # the pip-installed cuda-nvcc ships only the unversioned ``libnvvm.so``.
    # Read the SONAME and create a matching symlink so find_lib picks it up.
    nvvm_lib64 = nvcc_root / "nvvm" / "lib64"
    nvvm_unversioned = nvvm_lib64 / "libnvvm.so"
    if nvvm_unversioned.exists():
        try:
            import subprocess
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

    # Numba finds ``libcudart`` via ``CUDA_HOME/lib64`` (not via
    # LD_LIBRARY_PATH — setting LD_LIBRARY_PATH at Python runtime is
    # ignored by the dynamic linker). Create ``cuda_nvcc/lib64/`` and
    # symlink libcudart into it so numba's ``find_lib`` picks it up.
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
from omegaconf import OmegaConf
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
import soundfile as sf
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


def _is_main_process() -> bool:
    """Return True if this is rank 0 (or single-GPU). Used to guard
    file writes (checkpoints, metrics, config) in multi-GPU DDP."""
    import os
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
    """
    import numpy as np
    from lhotse import CutSet
    from lhotse.dataset.collation import collate_vectors
    from nemo.collections.asr.data.audio_to_text_lhotse_prompted import (
        PromptedAudioToTextLhotseDataset,
        PromptedAudioToTextMiniBatch,
        _drop_in_memory_data,
    )
    from nemo.collections.common.data import apply_prompt_format_fn

    class MonoAveragingPromptedDataset(PromptedAudioToTextLhotseDataset):
        """Load audio, average stereo channels, and format prompts as usual.

        Keeps the original cuts (which may already carry pre-formatted
        prompt attributes from NeMo's ``pretokenize`` step) — the prompt
        format function is separately patched at the registry level to
        accept MultiCut, so we don't need to rebuild the cuts here.
        """

        def __getitem__(self, cuts: CutSet) -> PromptedAudioToTextMiniBatch:
            # Load per-cut and average channels so we never pass
            # (B, C, T) to the model (the preprocessor only accepts (B, T)).
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

            # Mirror the fast/slow prompt paths from the upstream dataset.
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
    from lhotse import MonoCut
    from lhotse.cut import Cut, MixedCut
    from nemo.collections.common.data import prompt_fn as _prompt_fn

    def _to_mono_view(cut):
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
        # The registry is keyed by either ``example_type`` or
        # ``(example_type, formatter_type)``. Only patch entries whose
        # example type is a Cut (that's where MultiCut can show up).
        if isinstance(key, tuple):
            example_type = key[0]
        else:
            example_type = key
        if not (isinstance(example_type, type) and issubclass(example_type, Cut)):
            continue
        if getattr(existing, "_lyricscribe_patched", False):
            continue

        def _make_wrapper(fn):
            def wrapped(example, prompt):
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
    """
    from nemo.collections.common.data.lhotse import get_lhotse_dataloader_from_config

    # Must patch the prompt format registry before the dataloader tokenizes
    # cuts (with pretokenize=True, CutSet.map runs the formatter at manifest
    # iteration time, before our dataset ever sees the cut).
    _patch_prompt_format_for_multi_channel()

    dataset_cls = _build_mono_averaging_canary_dataset_cls()

    def _setup(self, config):
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
        original_step = pl_module.training_step

        def safe_step(batch, batch_idx):
            skip_flag = torch.zeros(1, device=pl_module.device)
            out = original_step(batch, batch_idx)

            # training_step may return a tensor or a dict with 'loss'.
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
    """Skip training batches that crash the RNNT/TDT loss.

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
        original_step = pl_module.training_step

        def safe_training_step(batch, batch_idx):
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
            # ``map_location='cpu'`` is critical for multi-GPU DDP: NeMo's
            # default placement sends weights to ``cuda:0``, so without this
            # every rank's subprocess briefly holds a full model copy on
            # GPU 0 before Lightning moves it to the rank-specific device.
            # On tight GPUs (e.g. 16 GB A4000 as cuda:0) that OOMs before
            # training ever starts.
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                model_name=self.config["base_model"],
                map_location="cpu",
            )
            self.model = self.model.cpu()

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
            cfg.train_ds.batch_duration = self.config.get("batch_duration", 600)
            cfg.train_ds.batch_size = None

            # Canary rejects multi-channel cuts in its prompt formatter and
            # has no on-the-fly stereo→mono averaging. Install a dataset
            # subclass that averages channels per-sample before collation.
            _install_mono_averaging_dataloader(self.model)

        else:
            # Parakeet: use the standard NeMo dataloader.
            # channel_selector="average" mixes stereo to mono at load time.
            cfg.train_ds.use_lhotse = False
            cfg.train_ds.channel_selector = "average"
            cfg.train_ds.batch_size = self.config["batch_size"]

        cfg.train_ds.manifest_filepath = str(train_manifest)
        cfg.train_ds.min_duration = 0.1
        cfg.train_ds.max_duration = self.config.get("max_duration", 240.0)

        # Optional encoder freeze for both Canary and Parakeet: cuts
        # AdamW state by the ratio of encoder-to-total params (~5x for
        # Canary, ~2x for Parakeet) so the full-Adam finetune fits on
        # a 12–16 GB GPU. No effect on Whisper. Enable via
        # ``"freeze_encoder": true`` in the job config.
        if self.config.get("freeze_encoder", False):
            frozen = 0
            for p in self.model.encoder.parameters():
                p.requires_grad = False
                frozen += 1
            logger.info(f"Froze {frozen} encoder parameter tensors")

        # Ensure the RNNT/TDT loss returns a scalar (mean over batch).
        # Newer NeMo/PL versions error on logging per-sample loss vectors.
        if hasattr(cfg, "loss") and cfg.loss is not None:
            cfg.loss.reduction = "mean_batch"
        cfg.train_ds.shuffle = True
        # num_workers overrideable via config — on tight-CPU dev boxes
        # 4 is optimal, on H200 nodes with 64+ cores raise to 16–32.
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

            # Both NeMo architectures can produce non-finite losses on rare
            # bad samples (Canary: bf16 attention overflow on long prompts;
            # Parakeet: bf16 RNNT logit overflow). Skip those batches
            # instead of poisoning the optimizer and driving every
            # subsequent step to NaN.
            callbacks = [
                step_checkpoint,
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

            # No limit_val_batches: the val manifest passed in is already
            # the fixed-size subset written at setup time, so we evaluate
            # all of it. This keeps Whisper and NeMo aligned on the same
            # samples. Full-corpus eval goes through the inference harness.

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

        if metrics:
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

        # Sanitize both pred and label IDs before decoding. Generated
        # predictions can contain -100 padding or out-of-vocab values
        # when the model hallucinates, which trips batch_decode with
        # OverflowError. Clip to the valid tokenizer vocab range.
        pad_id = self.processor.tokenizer.pad_token_id
        vocab_size = self.processor.tokenizer.vocab_size

        label_ids[label_ids == -100] = pad_id
        pred_ids[pred_ids == -100] = pad_id
        pred_ids[(pred_ids < 0) | (pred_ids >= vocab_size)] = pad_id

        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # jiwer rejects batches with any empty reference (division by zero
        # on 0-word references). Our manifests include empty-text windows
        # that land on purely instrumental regions — drop those pairs
        # before computing WER instead of crashing the whole chunk.
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
            # The val manifest passed in here is the fixed-size subset
            # materialized at setup time (``val_subset_manifest.jsonl``),
            # so we don't re-subset in Python. That keeps this eval pool
            # identical to what NeMo sees. Full-corpus eval runs
            # separately via the inference harness.
            eval_dataset = WhisperFinetuneDataset(self.val_manifest, self.processor)

        job_dir = Path(self.config["output_dir"]) / self.config["exp_name"]

        # Clamp so short smoke runs (e.g. max_epochs < warmup_epochs) still
        # satisfy HF's warmup_ratio ∈ [0, 1] constraint — 1.0 means the whole
        # run is warmup, which is the correct degenerate behaviour.
        warmup_ratio = min(
            1.0,
            self.config.get("warmup_epochs", 5) / max(self.config["max_epochs"], 1),
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
                gradient_accumulation_steps=self.config.get("grad_accum", 1),
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
                # Keep workers alive across epoch boundaries instead of
                # re-spawning — eliminates the ~2 s per-epoch cold start
                # where every worker re-imports transformers + torch.
                dataloader_persistent_workers=True,
                # Each worker keeps N batches pre-fetched ahead of the
                # GPU, so even a slow individual batch decode gets hidden
                # behind the prior step's compute. 4 is conservative —
                # memory cost = 4 × batch of mel spectrograms per worker,
                # typically well under 1 GB of host RAM.
                dataloader_prefetch_factor=4,
                dataloader_pin_memory=True,
                predict_with_generate=True,
                report_to="none",
                run_name=self.config["exp_name"],
                # ``torch_compile`` can fuse Whisper's small ops and
                # eliminate dtype-cast traffic, but the compile cost
                # hits the first ~10 steps and only amortizes in long
                # training runs. Default off — enable explicitly with
                # ``"torch_compile": true`` in the job config for
                # production runs where it pays back.
                torch_compile=self.config.get("torch_compile", False),
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

            # Fix a transformers 4.57.x bug: save_pretrained() writes
            # eos_token_id as a single-element list in generation_config.json,
            # which crashes Whisper's SuppressTokensAtBeginLogitsProcessor
            # at inference time (it uses eos_token_id as a slice index).
            # Unwrap it back to an int.
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
                config, train_manifest, val_manifest, chunk_end_epoch, job_dir,
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
