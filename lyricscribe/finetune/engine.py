import copy
import functools
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, Sampler
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed

from lyricscribe.transcribe.adapter import TRAIN_RECORD

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
GLOBAL_BATCH = 64
WARMUP_STEPS = 200
COOLDOWN = 0.2
RESUME_EVERY = 500
MIN_WINDOW_SECONDS = 0.5
DITHER = 1e-5
SPEC_AUGMENT = {"freq_masks": 2, "freq_width": 27, "time_masks": 10, "time_width": 0.05}
OPTIMIZER = {
    "optim": "adamw_torch",
    "adam_beta1": 0.9,
    "adam_beta2": 0.98,
    "adam_epsilon": 1e-8,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
}
SCHEDULES = ("wsd", "cosine", "linear")
STATE_FILE = "training_state.json"


def world_size() -> int:
    """
    Number of DDP ranks (1 outside ``torchrun``).

    :return: World size.
    """
    return int(os.environ.get("WORLD_SIZE", "1"))


def local_rank() -> int:
    """
    This process's local rank (0 outside ``torchrun``).

    :return: Local rank.
    """
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main_process() -> bool:
    """
    Whether this is global rank 0.

    :return: ``True`` on rank 0.
    """
    return int(os.environ.get("RANK", "0")) == 0


def device_map() -> dict:
    """
    ``device_map`` placing a whole model on this rank's GPU.

    :return: ``{"": "cuda:<local rank>"}``, or CPU without CUDA.
    """
    return {"": f"cuda:{local_rank()}" if torch.cuda.is_available() else "cpu"}


def steps_per_epoch(n_windows: int, global_batch: int = GLOBAL_BATCH) -> int:
    """
    Optimizer steps per epoch: the sampler drops the permutation's last ``n % global_batch`` windows.

    :param n_windows: Training windows.
    :param global_batch: Windows per optimizer step.
    :return: Steps per epoch.
    """
    return n_windows // global_batch


def epochs_to_steps(epochs: float, spe: int) -> int:
    """
    End step of a run of *epochs* epochs: ⌈f · S⌉.

    :param epochs: Epochs, possibly fractional.
    :param spe: Steps per epoch S.
    :return: The step.
    """
    f = Fraction(str(epochs))
    return -(-f.numerator * spe // f.denominator)


def cooldown_start(end_step: int, cooldown: float = COOLDOWN) -> int:
    """
    First decay step of a run ending at *end_step* whose last ⌈c · N⌉ steps cool down.

    :param end_step: The run's final step N.
    :param cooldown: Cooldown fraction c.
    :return: N − ⌈c · N⌉.
    """
    f = Fraction(str(cooldown))
    return end_step - (-(-f.numerator * end_step // f.denominator))


def wsd_multiplier(step: int, warmup: int, branch: int | None, end: int) -> float:
    """
    Warmup-stable-decay: min(1, (s + 1) / W) until *branch*, then 1 − √((s − branch) / (end − branch)).

    :param step: 0-based optimizer step.
    :param warmup: Warmup steps W.
    :param branch: First decay step, or ``None`` for a constant-LR run.
    :param end: Final step N.
    :return: The LR multiplier.
    """
    if branch is None or step < branch:
        return min(1.0, (step + 1) / warmup)
    return max(0.0, 1.0 - math.sqrt((step - branch) / (end - branch)))


def cosine_multiplier(step: int, warmup: int, end: int, min_ratio: float) -> float:
    """
    Linear warmup (s + 1) / W, then a cosine reaching *min_ratio* at the last update (step ``end − 1``).

    :param step: 0-based optimizer step.
    :param warmup: Warmup steps W.
    :param end: Final step N.
    :param min_ratio: Final LR over peak LR.
    :return: The LR multiplier.
    """
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, end - 1 - warmup))
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def linear_multiplier(step: int, warmup: int, end: int, min_ratio: float) -> float:
    """
    Linear warmup (s + 1) / W, then a linear decay reaching *min_ratio* at the last update.

    :param step: 0-based optimizer step.
    :param warmup: Warmup steps W.
    :param end: Final step N.
    :param min_ratio: Final LR over peak LR.
    :return: The LR multiplier.
    """
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, end - 1 - warmup))
    return 1.0 - (1.0 - min_ratio) * progress


def schedule(config: dict):
    """
    The run's LR multiplier as a ``functools.partial``, whose empty ``__dict__`` keeps a resumed
    ``LambdaLR`` from overwriting a branch's schedule with the trunk's.

    :param config: Job config (``schedule``, ``warmup_steps``, ``decay_start``, ``max_steps``,
        ``min_lr``, ``learning_rate``).
    :return: ``multiplier(step)``.
    """
    kind = config.get("schedule", "wsd")
    warmup, end = int(config.get("warmup_steps", WARMUP_STEPS)), int(config["max_steps"])
    if kind == "wsd":
        return functools.partial(wsd_multiplier, warmup=warmup, branch=config.get("decay_start"), end=end)
    min_ratio = float(config.get("min_lr") or 0.0) / float(config["learning_rate"])
    fn = {"cosine": cosine_multiplier, "linear": linear_multiplier}[kind]
    return functools.partial(fn, warmup=warmup, end=end, min_ratio=min_ratio)


def batch_split(per_device: int, global_batch: int = GLOBAL_BATCH) -> tuple[int, int]:
    """
    Gradient accumulation that makes ``per_device × accumulation × world`` the global batch.

    :param per_device: Windows per device per micro-batch.
    :param global_batch: Windows per optimizer step.
    :return: ``(per_device, accumulation)``.
    :raises ValueError: If the global batch isn't divisible by ``per_device × world``.
    """
    ranks = world_size()
    if global_batch % (per_device * ranks):
        raise ValueError(f"global batch {global_batch} is not divisible by batch_size {per_device} × {ranks} ranks")
    return per_device, global_batch // (per_device * ranks)


def sample_rng(seed: int, epoch: int, index: int, salt: str) -> np.random.Generator:
    """
    Generator for one window's random draws, independent of workers and resumes.

    :param seed: Run seed.
    :param epoch: Epoch of this visit.
    :param index: Window index.
    :param salt: Name of the draw (e.g. ``"dither"``).
    :return: A seeded generator.
    """
    digest = hashlib.sha256(f"{seed}:{epoch}:{index}:{salt}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def load_window(entry: dict) -> torch.Tensor | None:
    """
    Load a manifest window as 16 kHz mono float32 with torchaudio's resampler.

    :param entry: Manifest row with ``audio_filepath`` and optional ``offset`` / ``duration`` in seconds.
    :return: 1-D waveform, or ``None`` for an out-of-range window or one shorter than :data:`MIN_WINDOW_SECONDS`.
    """
    offset = entry.get("offset", 0) or 0
    duration = entry.get("duration")
    info = sf.info(entry["audio_filepath"])
    start = int(offset * info.samplerate)
    if start >= info.frames:
        return None
    frames = int(duration * info.samplerate) if duration else -1
    data, sr = sf.read(entry["audio_filepath"], start=start, frames=frames, dtype="float32", always_2d=True)
    if data.shape[0] < MIN_WINDOW_SECONDS * sr:
        return None
    audio = torch.from_numpy(data.mean(axis=1))
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio


def language_code(row: dict) -> str:
    """
    Base language code of a manifest row, e.g. ``"en"`` from ``"en-US"``.

    :param row: Manifest row; a missing ``language`` counts as English.
    :return: Lowercase base code.
    """
    return (row.get("language") or "en").split("-")[0].split("_")[0].lower()


class ManifestDataset(Dataset):
    """
    Windows of a JSONL manifest, loaded with :func:`load_window`.

    Indices may encode the epoch (``index + epoch * len``, see :class:`EpochSampler`),
    so per-window randomness (dither, label modes) differs by epoch yet is
    reproducible across workers and resumes.

    :param manifest: Manifest path.
    :param seed: Run seed.
    :param dither: Gaussian noise standard deviation added to each waveform (``0`` = none).
    """

    def __init__(self, manifest: Path, seed: int = 0, dither: float = 0.0) -> None:
        """
        Read every row of the manifest.

        :param manifest: Manifest path.
        :param seed: Run seed.
        :param dither: Waveform dither.
        """
        with Path(manifest).open() as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        self.seed = seed
        self.dither = dither
        logger.info(f"Loaded {len(self.rows)} windows from {manifest}")

    def __len__(self) -> int:
        """
        Number of windows.

        :return: Row count.
        """
        return len(self.rows)

    def locate(self, index: int) -> tuple[int, int]:
        """
        Split an epoch-encoded index.

        :param index: ``window + epoch * len``.
        :return: ``(window, epoch)``.
        """
        return index % len(self.rows), index // len(self.rows)

    def __getitem__(self, index: int) -> dict:
        """
        Load one window.

        :param index: Window index, optionally epoch-encoded.
        :return: ``audio`` (float32 numpy, or ``None`` when unreadable or too short), ``text``,
            ``language``, ``row``, ``index`` and ``epoch``.
        """
        idx, epoch = self.locate(index)
        row = self.rows[idx]
        try:
            audio = load_window(row)
            if audio is None:
                logger.warning(f"BAD_WINDOW {row.get('audio_filepath', '?')} @{row.get('offset', 0)}: short or out of range")
        except (OSError, RuntimeError, ValueError) as e:
            logger.warning(f"BAD_WINDOW {row.get('audio_filepath', '?')} @{row.get('offset', 0)}: unreadable: {e}")
            audio = None
        if audio is not None and self.dither > 0:
            noise = sample_rng(self.seed, epoch, idx, "dither").standard_normal(audio.shape[0])
            audio = audio + self.dither * torch.from_numpy(noise.astype(np.float32))
        return {
            "audio": None if audio is None else audio.numpy(),
            "text": row.get("text") or "",
            "language": language_code(row),
            "row": row,
            "index": idx,
            "epoch": epoch,
        }


def is_bad(item: dict) -> bool:
    """
    Whether a dataset item is an unreadable or rejected window.

    :param item: A :class:`ManifestDataset` or Whisper dataset item.
    :return: ``True`` for a bad window.
    """
    return bool(item.get("bad")) or ("audio" in item and item["audio"] is None)


class CountingCollator:
    """
    Wrap a collator to count the bad windows it receives into the batch's ``_bad_windows``.

    :param collator: The architecture's collator.
    """

    def __init__(self, collator) -> None:
        """
        Keep the wrapped collator.

        :param collator: The architecture's collator.
        """
        self.collator = collator

    def __call__(self, items: list[dict]) -> dict:
        """
        Collate and count.

        :param items: Dataset items.
        :return: The wrapped collator's batch plus ``_bad_windows``.
        """
        batch = self.collator(items)
        batch["_bad_windows"] = sum(is_bad(it) for it in items)
        return batch


def training_processor(processor):
    """
    A copy of *processor* whose feature extractor adds no dither, so the only dither in
    training is :class:`ManifestDataset`'s (σ :data:`DITHER`); the original, which is
    exported with the model, keeps its vendor setting for inference.

    :param processor: The model's processor.
    :return: The training copy (*processor* itself when no feature extractor dithers).
    """
    for attr in ("feature_extractor", "audio_processor"):
        fe = getattr(processor, attr, None)
        if fe is not None and getattr(fe, "dither", 0.0):
            train = copy.copy(processor)
            fe = copy.deepcopy(fe)
            logger.info(f"{type(fe).__name__}: built-in dither {fe.dither} off for training")
            fe.dither = 0.0
            setattr(train, attr, fe)
            return train
    return processor


def configured_prompt_mode(config: dict, modes: tuple[str, ...], default: str | None = None) -> str:
    """
    The prompt a trainer trains with: the config's ``prompt_mode``, named as inference names it.

    :param config: Trainer config.
    :param modes: The inference backend's ``prompt_modes``, default first.
    :param default: Default when the config has none (else ``modes[0]``).
    :return: The prompt mode.
    :raises ValueError: If the config names a mode the backend doesn't have.
    """
    mode = config.get("prompt_mode") or default or modes[0]
    if mode not in modes:
        raise ValueError(f"prompt_mode {mode!r} is not one of this model's prompts {modes}")
    return mode


def drop_bad(items: list[dict]) -> list[dict]:
    """
    Drop unreadable windows from a batch, keeping one silent empty-label item if none survive.

    :param items: Dataset items.
    :return: The readable items, or one placeholder flagged ``"placeholder": True``.
    """
    good = [it for it in items if it.get("audio") is not None]
    if good:
        return good
    return [{**items[0], "audio": np.zeros(SAMPLE_RATE, dtype=np.float32), "text": "", "placeholder": True}]


class EpochSampler(Sampler):
    """
    One seeded permutation per epoch, cut to a multiple of the global batch, yielding
    epoch-encoded indices. Every rank draws the same permutation, so any
    ``per_device × accumulation × world`` split sees the same windows at each optimizer step.

    :param n: Dataset length.
    :param seed: Run seed.
    :param global_batch: Windows per optimizer step.
    """

    def __init__(self, n: int, seed: int, global_batch: int = GLOBAL_BATCH) -> None:
        """
        Keep the largest multiple of the global batch.

        :param n: Dataset length.
        :param seed: Run seed.
        :param global_batch: Windows per optimizer step.
        """
        self.n = n
        self.seed = seed
        self.global_batch = global_batch
        self.keep = (n // global_batch) * global_batch
        self.epoch = 0
        self.pinned = False

    def set_epoch(self, epoch: int) -> None:
        """
        HF / accelerate's per-epoch call; ignored once :meth:`pin_epoch` has set the epoch, since
        under DDP their call can't reach this sampler through the resume wrappers.

        :param epoch: Epoch number.
        """
        if self.pinned:
            if epoch != self.epoch:
                logger.warning(f"sampler epoch {self.epoch} kept over set_epoch({epoch})")
            return
        self.epoch = epoch

    def pin_epoch(self, epoch: int) -> None:
        """
        Set the epoch from the trainer's step count (see :class:`SamplerEpochCallback`).

        :param epoch: Epoch number.
        """
        self.epoch = epoch
        self.pinned = True

    def order(self, epoch: int) -> list[int]:
        """
        The window order of one epoch.

        :param epoch: Epoch number.
        :return: Window indices (not epoch-encoded).
        """
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)
        return torch.randperm(self.n, generator=g)[: self.keep].tolist()

    def __iter__(self):
        """
        Yield this epoch's epoch-encoded indices.

        :return: Index iterator.
        """
        base = self.epoch * self.n
        return iter([i + base for i in self.order(self.epoch)])

    def __len__(self) -> int:
        """
        Windows per epoch.

        :return: ``n`` rounded down to a multiple of the global batch.
        """
        return self.keep


@dataclass
class FeatureLayout:
    """
    Where a model's log-mel features live in its batch, for SpecAugment.

    :param key: Batch key of the features.
    :param time_dim: Time axis of the feature tensor.
    :param freq_dim: Frequency axis of the feature tensor.
    :param stack: Frames stacked along the frequency axis (Granite stacks 2 × 80 bins).
    :param frames_per_length: Feature frames per unit of ``_feature_lengths``.
    """

    key: str = "input_features"
    time_dim: int = 1
    freq_dim: int = 2
    stack: int = 1
    frames_per_length: int = 1


def spec_augment(
    features: torch.Tensor,
    lengths: torch.Tensor,
    layout: FeatureLayout,
    generator: torch.Generator,
    freq_masks: int = SPEC_AUGMENT["freq_masks"],
    freq_width: int = SPEC_AUGMENT["freq_width"],
    time_masks: int = SPEC_AUGMENT["time_masks"],
    time_width: float = SPEC_AUGMENT["time_width"],
) -> torch.Tensor:
    """
    SpecAugment applied the same way to every model: per sample, ``freq_masks`` bands of up to
    ``freq_width`` mel bins and ``time_masks`` spans of up to ``time_width`` of its valid length,
    each replaced by the sample's per-bin mean over its valid frames (NeMo's zero fill on
    per-feature-normalised mels).

    :param features: Feature tensor with a leading batch axis.
    :param lengths: Valid frames per sample.
    :param layout: Axes of *features*.
    :param generator: CPU generator for the mask draws.
    :param freq_masks: Frequency masks per sample.
    :param freq_width: Widest frequency mask, in mel bins.
    :param time_masks: Time masks per sample.
    :param time_width: Widest time mask, as a fraction of the valid length.
    :return: The masked features (modified in place).
    """
    # A (batch, time, freq) view of the same storage.
    freq_after_move = layout.freq_dim if layout.freq_dim > layout.time_dim else layout.freq_dim + 1
    view = features.movedim(layout.time_dim, 1).movedim(freq_after_move, 2)
    num_frames, num_bins = view.shape[1], view.shape[2]
    bins = num_bins // layout.stack
    draws = torch.rand((len(lengths), freq_masks + time_masks, 2), generator=generator)
    for b, length in enumerate(lengths.tolist()):
        length = max(1, min(num_frames, int(length) * layout.frames_per_length))
        fill = view[b, :length].float().mean(dim=0).to(view.dtype)
        for m in range(freq_masks):
            width = int(draws[b, m, 0] * (freq_width + 1))
            start = int(draws[b, m, 1] * max(1, bins - width))
            for s in range(layout.stack):
                lo = s * bins + start
                view[b, :, lo:lo + width] = fill[lo:lo + width]
        max_width = max(1, int(time_width * length))
        for m in range(freq_masks, freq_masks + time_masks):
            width = int(draws[b, m, 0] * (max_width + 1))
            start = int(draws[b, m, 1] * max(1, length - width))
            view[b, start:start + width, :] = fill
    return features


def skip_if_non_finite(model, loss: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """
    Swap a non-finite training loss for a zero touching every trainable parameter, deciding
    across all ranks so DDP's gradient sync stays in step.

    :param model: The (possibly DDP-wrapped) model.
    :param loss: This rank's loss.
    :return: ``(loss, skipped)``.
    """
    bad = torch.tensor([0.0 if torch.isfinite(loss).all() else 1.0], device=loss.device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(bad, op=torch.distributed.ReduceOp.MAX)
    if bad.item() == 0:
        return loss, False
    logger.warning(f"NONFINITE_LOSS skipped (this rank: {loss.item()})")
    return sum(p.sum() for p in model.parameters() if p.requires_grad) * 0.0, True


class FinetuneTrainer(Trainer):
    """
    HF ``Trainer`` shared by every model: the epoch-seeded window order, SpecAugment
    on the GPU batch (seeded by step, so resumes are exact), a token-level loss normalised over
    the whole global batch for every model, and a DDP-safe skip of non-finite losses.

    ``loss_mode`` is ``"model"`` (the model's own loss, which honours ``num_items_in_batch``),
    ``"seq2seq"`` (cross-entropy on aligned logits/labels) or ``"causal"`` (shifted), the last two
    for models whose built-in loss is a per-batch mean.
    """

    loss_is_scaled_for_ga = True

    def configure(
        self,
        seed: int,
        loss_mode: str = "model",
        layout: FeatureLayout | None = None,
        augment: bool = True,
        multiplier=None,
        global_batch: int = GLOBAL_BATCH,
    ) -> "FinetuneTrainer":
        """
        Set the run options.

        :param seed: Run seed (window order, mask draws).
        :param loss_mode: ``"model"``, ``"seq2seq"`` or ``"causal"``.
        :param layout: Feature layout for SpecAugment, or ``None`` for no features.
        :param augment: Apply SpecAugment in training.
        :param multiplier: The LR multiplier (:func:`schedule`), or ``None`` for the
            ``TrainingArguments`` scheduler.
        :param global_batch: Windows per optimizer step, for the window order.
        :return: ``self``.
        :raises ValueError: If ``loss_mode`` is ``"model"`` but the model ignores ``num_items_in_batch``
            (its loss would be a per-micro-batch mean, wrong under accumulation).
        """
        if loss_mode == "model" and not self.model_accepts_loss_kwargs:
            raise ValueError(f"{type(self.model).__name__} ignores num_items_in_batch; use a token loss mode")
        self.run_seed = seed
        self.global_batch = global_batch
        self.loss_mode = loss_mode
        self.layout = layout
        self.augment = augment and layout is not None
        self.multiplier = multiplier
        self.counters = {"nonfinite_loss": 0, "placeholder_batches": 0, "bad_windows": 0}
        self._micro = (-1, 0)
        self.window_sampler = None
        self.add_callback(SamplerEpochCallback(self))
        self.add_callback(RnnDropoutSeedCallback())
        return self

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        """
        A ``LambdaLR`` over :attr:`multiplier` (HF's own schedulers otherwise).

        :param num_training_steps: Total steps (unused by :attr:`multiplier`).
        :param optimizer: Optimizer (default: the trainer's).
        :return: The scheduler.
        """
        if getattr(self, "multiplier", None) is None:
            return super().create_scheduler(num_training_steps, optimizer)
        if self.lr_scheduler is None:
            self.lr_scheduler = LambdaLR(optimizer or self.optimizer, self.multiplier)
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _get_train_sampler(self, train_dataset=None):
        """
        The :class:`EpochSampler`.

        :param train_dataset: Training dataset (defaults to ``self.train_dataset``).
        :return: The sampler.
        """
        dataset = train_dataset if train_dataset is not None else self.train_dataset
        self.window_sampler = EpochSampler(len(dataset), self.run_seed, self.global_batch)
        return self.window_sampler

    def _mask_generator(self) -> torch.Generator:
        """
        Generator for this micro-batch's SpecAugment draws, from (seed, step, micro-batch, rank).

        :return: Seeded CPU generator.
        """
        step = self.state.global_step
        k = self._micro[1] + 1 if self._micro[0] == step else 0
        self._micro = (step, k)
        rank = int(os.environ.get("RANK", "0"))
        digest = hashlib.sha256(f"{self.run_seed}:{step}:{k}:{rank}:specaug".encode()).digest()
        return torch.Generator().manual_seed(int.from_bytes(digest[:8], "little"))

    def _token_loss(self, model, inputs, num_items_in_batch):
        """
        Cross-entropy summed over label tokens, divided by the global batch's label tokens.

        :param model: Possibly wrapped model.
        :param inputs: Batch with ``labels``.
        :param num_items_in_batch: Label tokens across the accumulated global batch.
        :return: ``(loss, outputs)``.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self.loss_mode == "causal":
            logits, labels = logits[:, :-1], labels[:, 1:]
        total = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1).to(logits.device),
            ignore_index=-100, reduction="sum",
        )
        return self._normalise(total, labels, num_items_in_batch), outputs

    def _normalise(self, total: torch.Tensor, labels: torch.Tensor, num_items_in_batch) -> torch.Tensor:
        """
        Divide a summed token loss by the global token count, mirroring HF's cross-device scaling.

        :param total: Summed loss over this micro-batch's label tokens.
        :param labels: This micro-batch's labels (``-100`` = ignored), used outside training.
        :param num_items_in_batch: Label tokens across the global batch, or ``None``.
        :return: The normalised loss.
        """
        if num_items_in_batch is None:
            return total / (labels != -100).sum().clamp(min=1)
        loss = total / num_items_in_batch.to(total.device) if torch.is_tensor(num_items_in_batch) else total / num_items_in_batch
        if self.args.average_tokens_across_devices:
            loss = loss * self.accelerator.num_processes
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        SpecAugment (training only), the loss of ``loss_mode``, and the non-finite skip.

        :param model: Possibly wrapped model.
        :param inputs: Collated batch.
        :param return_outputs: Also return the model outputs.
        :param num_items_in_batch: Label tokens across the global batch.
        :return: The loss, or ``(loss, outputs)``.
        :raises RuntimeError: If training without ``num_items_in_batch`` (the loss would not be global).
        """
        lengths = inputs.pop("_feature_lengths", None)
        placeholder = bool(inputs.pop("_placeholder", False))
        bad = int(inputs.pop("_bad_windows", 0))
        if model.training:
            self.counters["bad_windows"] += bad
        if model.training and num_items_in_batch is None:
            raise RuntimeError("training needs num_items_in_batch for a global-batch loss")
        if model.training and self.augment and lengths is not None:
            spec_augment(inputs[self.layout.key], lengths, self.layout, self._mask_generator())
        if self.loss_mode == "model":
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        else:
            loss, outputs = self._token_loss(model, inputs, num_items_in_batch)
        if model.training:
            if placeholder:
                self.counters["placeholder_batches"] += 1
                loss = loss * 0.0
            loss, skipped = skip_if_non_finite(model, loss)
            self.counters["nonfinite_loss"] += int(skipped)
        return (loss, outputs) if return_outputs else loss


class SamplerEpochCallback(TrainerCallback):
    """
    Set the window sampler's epoch from the step count at each epoch start, through a direct
    reference: under DDP, accelerate's ``set_epoch`` can't reach it through
    ``SkipBatchSampler → BatchSamplerShard`` on a mid-epoch resume, and the epoch-0 order,
    dither and timestamp draws would replay.

    :param trainer: The :class:`FinetuneTrainer` that owns the sampler.
    """

    def __init__(self, trainer) -> None:
        """
        Keep the trainer.

        :param trainer: The :class:`FinetuneTrainer`.
        """
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, **kwargs):
        """
        Pin the sampler to epoch ``global_step // steps_per_epoch`` before the epoch's iterator is built.

        :param args: Training arguments.
        :param state: Trainer state.
        :param control: Trainer control flags.
        :param kwargs: Unused.
        :return: *control*.
        """
        sampler = self.trainer.window_sampler
        loader = kwargs.get("train_dataloader")
        if sampler is not None:
            # HF's own count: dataloader batches per rank over accumulation.
            spe = len(loader) // args.gradient_accumulation_steps if loader is not None else sampler.keep // sampler.global_batch
            sampler.pin_epoch(state.global_step // max(1, spe))
        return control


class RnnDropoutSeedCallback(TrainerCallback):
    """
    Reseed cuDNN's RNN dropout state (Parakeet's prediction-network LSTM) from the CUDA
    generator at every step. cuDNN draws that state from the generator once and then advances
    it internally, so the generator states a checkpoint saves can't restore it; writing the
    generator state back unchanged makes cuDNN redraw it at the next RNN forward, in a resumed
    run exactly as in an uninterrupted one.
    """

    def on_step_begin(self, args, state, control, **kwargs):
        """
        Write this device's CUDA generator state back (after HF has restored it on a resume).

        :param args: Training arguments.
        :param state: Trainer state.
        :param control: Trainer control flags.
        :param kwargs: Unused.
        :return: *control*.
        """
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
        return control


class ZeroNonFiniteGradsCallback(TrainerCallback):
    """
    Zero the gradients before an optimizer step whose all-reduced, clipped gradients are
    non-finite; every rank sees the same gradients and agrees.
    """

    def __init__(self) -> None:
        """
        Start the counter at zero.
        """
        self.count = 0

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        """
        Zero all gradients if any is non-finite.

        :param args: Training arguments.
        :param state: Trainer state.
        :param control: Trainer control flags.
        :param model: The model.
        :param kwargs: Unused.
        :return: *control*.
        """
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        if grads and not torch.isfinite(torch.stack(torch._foreach_norm(grads))).all():
            logger.warning(f"NONFINITE_GRAD step {state.global_step}: zeroing gradients")
            torch._foreach_zero_(grads)
            self.count += 1
        return control


class GradientCheckCallback(TrainerCallback):
    """
    On a fresh run, require a non-zero gradient on every trainable parameter: all but LoRA ``A``
    matrices at the first optimizer step, and the ``A`` matrices at the third (they stay zero
    until the first update has moved ``B``, so they are checked at the second step).
    """

    def __init__(self) -> None:
        """
        Armed for both checks.
        """
        self.pending = {0: "first", 1: "lora_A"}

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        """
        Check the gradients at global steps 0 and 1.

        :param args: Training arguments.
        :param state: Trainer state.
        :param control: Trainer control flags.
        :param model: The model.
        :param kwargs: Unused.
        :return: *control*.
        :raises RuntimeError: Listing parameters with a missing or all-zero gradient.
        """
        kind = self.pending.pop(state.global_step, None)
        if kind is None:
            return control
        missing, zero = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad or ((".lora_A." in name) != (kind == "lora_A")):
                continue
            if p.grad is None:
                missing.append(name)
            elif not p.grad.any():
                zero.append(name)
        if missing or zero:
            raise RuntimeError(
                f"GRADCHECK ({kind}) failed: {len(missing)} trainable params without grad (e.g. {missing[:3]}), "
                f"{len(zero)} with all-zero grad (e.g. {zero[:3]})"
            )
        logger.info(f"GRADCHECK ({kind}) ok at step {state.global_step}")
        return control


class ExactSaveCallback(TrainerCallback):
    """
    Save resumable checkpoints at exact steps (kept) plus every ``resume_every`` steps
    (only the newest kept), replacing HF's rotation.

    :param keep: Steps whose checkpoints are never deleted.
    :param resume_every: Interval of the rolling resume checkpoint (``0`` = none).
    :param save_final: Also save at the segment's last step.
    """

    def __init__(self, keep: list[int], resume_every: int = RESUME_EVERY, save_final: bool = True) -> None:
        """
        Record the save plan.

        :param keep: Kept steps.
        :param resume_every: Rolling checkpoint interval.
        :param save_final: Save at the last step.
        """
        self.keep = set(keep)
        self.resume_every = resume_every
        self.save_final = save_final

    def on_step_end(self, args, state, control, **kwargs):
        """
        Request a save at kept steps, rolling-checkpoint steps and the last step.

        :param args: Training arguments.
        :param state: Trainer state.
        :param control: Trainer control flags.
        :param kwargs: Unused.
        :return: *control*.
        """
        step = state.global_step
        final = self.save_final and step == state.max_steps
        if step in self.keep or final or (self.resume_every and step % self.resume_every == 0):
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        """
        Delete older rolling checkpoints (never kept ones or linked branch sources).

        :param args: Training arguments.
        :param state: Trainer state.
        :param control: Trainer control flags.
        :param kwargs: Unused.
        :return: *control*.
        """
        if not is_main_process():
            return control
        for ckpt in Path(args.output_dir).glob("checkpoint-*"):
            step = int(ckpt.name.split("-")[1])
            if step < state.global_step and step not in self.keep and not ckpt.is_symlink():
                shutil.rmtree(ckpt, ignore_errors=True)
        return control


def slurm_seconds_left() -> float | None:
    """
    Remaining walltime of the current Slurm job.

    :return: Seconds left, or ``None`` outside Slurm or when ``squeue`` can't tell.
    """
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        left = subprocess.run(["squeue", "-h", "-j", job, "-o", "%L"], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not left or left in ("UNLIMITED", "NOT_SET", "INVALID"):
        return None
    days, _, rest = left.rpartition("-")
    parts = [int(p) for p in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return (int(days) if days else 0) * 86400 + h * 3600 + m * 60 + sec


class ThroughputCallback(TrainerCallback):
    """
    Log seconds per optimizer step over a window after warm-up, and once the projected time to
    the last step against the Slurm walltime left.

    :param skip: Steps excluded from timing.
    """

    def __init__(self, skip: int = 5) -> None:
        """
        Start untimed.

        :param skip: Warm-up steps not timed.
        """
        self.skip = skip
        self.t0 = None
        self.s0 = None
        self.seconds_per_step = None

    def on_step_end(self, args, state, control, **kwargs):
        """
        Start the clock after ``skip`` steps and update the running rate.

        :param args: Training arguments.
        :param state: Trainer state.
        :param control: Trainer control flags.
        :param kwargs: Unused.
        :return: *control*.
        """
        if self.t0 is None:
            if state.global_step >= self.skip:
                self.t0, self.s0 = time.time(), state.global_step
            return control
        done = state.global_step - self.s0
        if done > 0:
            self.seconds_per_step = (time.time() - self.t0) / done
            if done % 20 == 0 and is_main_process():
                logger.info(f"THROUGHPUT {self.seconds_per_step:.3f} s/step at step {state.global_step}")
            if done == 20 and is_main_process():
                remaining = (state.max_steps - state.global_step) * self.seconds_per_step
                left = slurm_seconds_left()
                if left is not None and remaining > left:
                    logger.warning(f"WALLTIME: {remaining / 3600:.2f} h to step {state.max_steps}, "
                                   f"{left / 3600:.2f} h left; rerun to resume from the last checkpoint")
        return control


def training_arguments(config: dict, output_dir: Path, max_steps: int, per_device: int, **overrides) -> TrainingArguments:
    """
    ``TrainingArguments``: AdamW (β 0.9/0.98, ε 1e-8, the config's weight decay off biases and
    norms), clip 1.0, bf16 autocast. The LR schedule is :func:`schedule`, installed by
    :meth:`FinetuneTrainer.create_scheduler`.

    :param config: Job config (``learning_rate``, ``weight_decay``, ``global_batch``, ``seed``, workers).
    :param output_dir: HF trainer output dir.
    :param max_steps: Final optimizer step of this segment.
    :param per_device: Windows per device per micro-batch.
    :param overrides: Model-specific fields (e.g. gradient checkpointing).
    :return: Training arguments.
    """
    per_device, accum = batch_split(per_device, int(config.get("global_batch", GLOBAL_BATCH)))
    seed = int(config.get("seed", 0))
    workers = int(config.get("num_workers", 8))
    kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": per_device,
        "per_device_eval_batch_size": per_device,
        "prediction_loss_only": True,
        "gradient_accumulation_steps": accum,
        "learning_rate": float(config["learning_rate"]),
        "max_steps": max_steps,
        "lr_scheduler_type": "constant",
        **OPTIMIZER,
        "weight_decay": float(config.get("weight_decay", OPTIMIZER["weight_decay"])),
        "bf16": torch.cuda.is_available(),
        "seed": seed,
        "data_seed": seed,
        "save_strategy": "no",
        "eval_strategy": "no",
        "logging_steps": int(config.get("logging_steps", 10)),
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "dataloader_num_workers": workers,
        "dataloader_persistent_workers": workers > 0,
        "dataloader_pin_memory": True,
        "ddp_find_unused_parameters": False,
        "average_tokens_across_devices": True,
        "report_to": "none",
        "run_name": config["exp_name"],
    }
    kwargs.update(overrides)
    return TrainingArguments(**kwargs)


@dataclass
class TrainingParts:
    """
    What a model family contributes to a run.

    :param model: The model to train (trainable parameters already selected and in fp32).
    :param dataset: Training dataset.
    :param collator: Training collator.
    :param trainer_cls: :class:`FinetuneTrainer` subclass.
    :param loss_mode: See :class:`FinetuneTrainer`.
    :param layout: Feature layout for SpecAugment.
    :param per_device: Windows per device per micro-batch.
    :param args: Extra ``TrainingArguments`` fields.
    :param export: ``export(path)`` writes an inference-loadable checkpoint.
    :param notes: Recipe facts recorded in the training state.
    :param prompt_mode: The prompt the collator trains with, as inference names it; recorded
        in each export's :data:`TRAIN_RECORD` for the inference loaders to check.
    :param eval_dataset: Held-out windows scored at every export, or ``None``.
    """

    model: object
    dataset: Dataset
    collator: object
    trainer_cls: type = FinetuneTrainer
    loss_mode: str = "model"
    layout: FeatureLayout | None = None
    per_device: int = 4
    args: dict = field(default_factory=dict)
    export: object = None
    notes: dict = field(default_factory=dict)
    prompt_mode: str = "native"
    eval_dataset: Dataset | None = None


def latest_checkpoint(hf_dir: Path) -> str | None:
    """
    Newest complete HF checkpoint in *hf_dir*.

    :param hf_dir: HF trainer output dir.
    :return: Its path, or ``None``.
    """
    if not hf_dir.exists():
        return None
    for ckpt in sorted(hf_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]), reverse=True):
        if (ckpt / "trainer_state.json").exists():
            return str(ckpt)
    return None


def parameter_summary(model) -> dict:
    """
    Trainable and total parameter counts, and the dtypes of trainable parameters.

    :param model: The model.
    :return: ``trainable``, ``total`` and ``trainable_dtypes``.
    """
    trainable = total = 0
    dtypes: dict[str, int] = {}
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
    return {"trainable": trainable, "total": total, "trainable_dtypes": dtypes}


def _effective(trainer: Trainer, parts: TrainingParts) -> dict:
    """
    What the run actually uses.

    :param trainer: The built trainer, after optimizer and scheduler creation.
    :param parts: The architecture's parts.
    :return: Optimizer, schedule samples, batch split, precision and augmentation.
    """
    opt = trainer.optimizer
    groups = [
        {"params": sum(p.numel() for p in g["params"]), "weight_decay": g.get("weight_decay"), "lr": g.get("lr")}
        for g in opt.param_groups
    ]
    sched = trainer.lr_scheduler
    fn = sched.lr_lambdas[0]
    keys = getattr(fn, "keywords", {})
    points = {0, 1, keys.get("warmup", 1) - 1, keys.get("warmup", 1), trainer.args.max_steps - 1}
    if keys.get("branch") is not None:
        points |= {keys["branch"] - 1, keys["branch"], keys["branch"] + 1}
    probe = {str(step): float(fn(step)) for step in sorted(x for x in points if x >= 0)}
    args = trainer.args
    return {
        "optimizer": type(opt).__name__,
        "optimizer_defaults": {k: v for k, v in opt.defaults.items() if isinstance(v, (int, float, tuple, bool))},
        "param_groups": groups,
        "max_grad_norm": args.max_grad_norm,
        "lr_lambda": probe,
        "schedule": {"fn": getattr(getattr(fn, "func", fn), "__name__", str(fn)), **keys},
        "learning_rate": args.learning_rate,
        "per_device": args.per_device_train_batch_size,
        "grad_accum": args.gradient_accumulation_steps,
        "world_size": world_size(),
        "global_batch": args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size(),
        "bf16_autocast": args.bf16,
        "loss_mode": parts.loss_mode,
        "spec_augment": SPEC_AUGMENT if trainer.augment else None,
        "layout": None if parts.layout is None else vars(parts.layout),
        "gradient_checkpointing": args.gradient_checkpointing,
    }


class ExportCallback(TrainerCallback):
    """
    Export an inference checkpoint (with its :data:`TRAIN_RECORD`) at listed steps.

    :param steps: Steps to export at.
    :param export: ``export(step)`` callable.
    """

    def __init__(self, steps: list[int], export) -> None:
        """
        Keep the plan.

        :param steps: Steps to export at.
        :param export: ``export(step)`` callable.
        """
        self.steps = set(steps)
        self.export = export

    def on_step_end(self, args, state, control, **kwargs):
        """
        Export at a listed step (every rank calls *export*, which reduces the counters
        collectively and writes on rank 0).

        :param args: Training arguments.
        :param state: Trainer state.
        :param control: Trainer control flags.
        :param kwargs: Unused.
        :return: *control*.
        """
        if state.global_step in self.steps:
            self.export(state.global_step)
        return control


COUNTERS_FILE = "counters.json"
# Counted per rank (summed across ranks); the others are decided collectively and equal on every rank.
SUMMED_COUNTERS = ("bad_windows", "placeholder_batches")


class Counters:
    """
    Run counters cumulative from step 0: the totals saved in the checkpoint a segment resumes
    from (a branch's trunk checkpoint included) plus this segment's, summed across ranks.

    :param trainer: The :class:`FinetuneTrainer`.
    :param zero_grads: The :class:`ZeroNonFiniteGradsCallback`.
    :param resume: Checkpoint the segment resumes from, or ``None``.
    """

    def __init__(self, trainer, zero_grads, resume: str | None) -> None:
        """
        Load the resumed totals.

        :param trainer: The trainer.
        :param zero_grads: The non-finite-gradient callback.
        :param resume: Resumed checkpoint, or ``None``.
        """
        self.trainer = trainer
        self.zero_grads = zero_grads
        path = Path(resume) / COUNTERS_FILE if resume else None
        self.base = json.loads(path.read_text()) if path is not None and path.exists() else {}

    def totals(self) -> dict:
        """
        The cumulative counters; a collective call under DDP (every rank must make it).

        :return: ``bad_windows``, ``placeholder_batches``, ``nonfinite_loss``, ``nonfinite_grad``.
        """
        local = self.trainer.counters
        summed = _reduce_counts({k: local[k] for k in SUMMED_COUNTERS})
        now = {**summed, "nonfinite_loss": local["nonfinite_loss"], "nonfinite_grad": self.zero_grads.count}
        return {k: int(self.base.get(k, 0)) + int(v) for k, v in now.items()}


class CountersSaveCallback(TrainerCallback):
    """
    Write the cumulative counters into each saved checkpoint, so resumed and branched
    segments continue the totals.

    :param counters: The run's :class:`Counters`.
    """

    def __init__(self, counters: Counters) -> None:
        """
        Keep the counters.

        :param counters: The run's counters.
        """
        self.counters = counters

    def on_save(self, args, state, control, **kwargs):
        """
        Reduce on every rank, write on rank 0.

        :param args: Training arguments.
        :param state: Trainer state.
        :param control: Trainer control flags.
        :param kwargs: Unused.
        :return: *control*.
        """
        totals = self.counters.totals()
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if is_main_process() and ckpt.is_dir():
            (ckpt / COUNTERS_FILE).write_text(json.dumps(totals))
        return control


def _reduce_counts(counts: dict) -> dict:
    """
    Sum per-rank counters across DDP ranks.

    :param counts: This rank's counters.
    :return: The summed counters (unchanged outside DDP).
    """
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return counts
    keys = sorted(counts)
    t = torch.tensor([float(counts[k]) for k in keys], device="cuda" if torch.cuda.is_available() else "cpu")
    torch.distributed.all_reduce(t)
    return {k: int(v) for k, v in zip(keys, t.tolist())}


def run(arch_trainer, config: dict, job_dir: Path, save: bool = True) -> dict:
    """
    Run (or resume) a job to its ``max_steps``: a whole run, a constant-LR trunk or a cooldown
    branch. Saves the kept checkpoints and writes the exports the config lists.

    :param arch_trainer: Model-family trainer with ``parts(manifest, seed, eval_manifest)``.
    :param config: Job config.
    :param job_dir: Job directory.
    :param save: Write checkpoints, exports and the training state (``False`` for probes).
    :return: Run status (``global_step``, ``complete``, counters, ``seconds_per_step``).
    """
    from lyricscribe.transcribe.job import environment_record

    env = environment_record()
    torch.backends.cuda.enable_cudnn_sdp(False)
    seed = int(config.get("seed", 0))
    # Before model init and get_peft_model, so the seed sets the LoRA init too.
    set_seed(seed)
    manifest = Path(config.get("train_manifest") or job_dir / "train_manifest.jsonl")
    eval_manifest = config.get("eval_manifest") or job_dir / "val_subset_manifest.jsonl"
    eval_manifest = Path(eval_manifest) if save and Path(eval_manifest).exists() else None
    parts = arch_trainer.parts(manifest, seed, eval_manifest)
    max_steps = int(config["max_steps"])
    args = training_arguments(config, job_dir / "hf_trainer", max_steps, parts.per_device, **parts.args)
    zero_grads = ZeroNonFiniteGradsCallback()
    speed = ThroughputCallback()
    callbacks = [zero_grads, speed]
    if save:
        callbacks.append(ExactSaveCallback(
            list(config.get("keep_steps", [])), int(config.get("resume_every", RESUME_EVERY)),
            bool(config.get("save_final", True)),
        ))
    state_path = job_dir / STATE_FILE
    record = json.loads(state_path.read_text()) if state_path.exists() else {"segments": []}

    def export(step: int) -> None:
        """
        Write an inference checkpoint with its training record, score the held-out windows and
        append to ``metrics.jsonl``. Every rank calls it (the counters and evaluation are
        collective); rank 0 writes.

        :param step: Step the weights are from.
        """
        totals = counters.totals()
        metrics = {}
        if parts.eval_dataset is not None:
            metrics = {k: float(v) for k, v in trainer.evaluate(parts.eval_dataset).items()}
        if not is_main_process():
            return
        path = job_dir / "exports" / f"step_{step}"
        parts.export(path)
        summary = {k: entry.get(k) for k in ("windows", "steps_per_epoch", "parameters", "notes")}
        (path / TRAIN_RECORD).write_text(json.dumps({
            "family": config["architecture"],
            "base_model": config["base_model"],
            "adaptation": config.get("adaptation", "full"),
            "schedule": config.get("schedule", "wsd"),
            "prompt_mode": parts.prompt_mode,
            "learning_rate": float(config["learning_rate"]),
            "seed": seed,
            "steps": {"step": step, "max_steps": max_steps, "decay_start": config.get("decay_start"),
                      "parent": config.get("parent")},
            "training_state": {**summary, "counters": totals, "counters_scope": "cumulative from step 0, all ranks"},
            "metrics": metrics,
            "environment": env,
        }, indent=2, default=str))
        losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
        with open(job_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps({"step": step, "epoch": step / max(1, entry["steps_per_epoch"]),
                                **({"train_loss": losses[-1]} if losses else {}), **metrics}) + "\n")

    export_steps = sorted({int(x) for x in config.get("export_steps", [])} | (
        {max_steps} if config.get("export_final", True) else set()))
    if save and export_steps:
        callbacks.append(ExportCallback(export_steps, export))
    resume = latest_checkpoint(job_dir / "hf_trainer") if save else None
    if resume is None:
        callbacks.append(GradientCheckCallback())
    global_batch = int(config.get("global_batch", GLOBAL_BATCH))
    trainer = parts.trainer_cls(
        model=parts.model, args=args, train_dataset=parts.dataset, data_collator=CountingCollator(parts.collator),
        callbacks=callbacks,
    ).configure(
        seed, parts.loss_mode, parts.layout, bool(config.get("use_augmentation", True)),
        multiplier=schedule(config), global_batch=global_batch,
    )
    counters = Counters(trainer, zero_grads, resume)
    if save:
        trainer.add_callback(CountersSaveCallback(counters))
    trainer.create_optimizer_and_scheduler(num_training_steps=args.max_steps)
    entry = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "resume_from": resume,
        "manifest": str(manifest),
        "windows": len(parts.dataset),
        "steps_per_epoch": steps_per_epoch(len(parts.dataset), global_batch),
        "first_step_windows": EpochSampler(len(parts.dataset), seed, global_batch).order(0)[:global_batch],
        "parameters": parameter_summary(parts.model),
        "effective": _effective(trainer, parts),
        "notes": parts.notes,
        "prompt_mode": parts.prompt_mode,
        "environment": env,
    }
    if save and is_main_process():
        record["segments"].append(entry)
        state_path.write_text(json.dumps(record, indent=2, default=str))
    logger.info(f"Training to step {max_steps} ({config.get('schedule', 'wsd')}, decay from "
                f"{config.get('decay_start')}), resume={resume}")
    trainer.train(resume_from_checkpoint=resume)
    entry.update({
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "global_step": trainer.state.global_step,
        "complete": trainer.state.global_step >= max_steps,
        "counters": counters.totals(),
        "counters_scope": "cumulative from step 0, all ranks",
        "seconds_per_step": speed.seconds_per_step,
        "last_logs": [h for h in trainer.state.log_history if "loss" in h][-3:],
    })
    if save and is_main_process():
        record["segments"][-1] = entry
        state_path.write_text(json.dumps(record, indent=2, default=str))
    return entry


def extend_trunk(job_dir: Path, max_steps: int, keep_steps: list[int] | None = None) -> dict:
    """
    Continue a constant-LR trunk further, keeping more checkpoints along the way.

    :param job_dir: Trunk job dir.
    :param max_steps: New final step.
    :param keep_steps: Extra kept checkpoint steps.
    :return: The updated config.
    :raises ValueError: For a run that decays, or a final step not past the current one.
    """
    cfg_path = job_dir / "config.json"
    config = json.loads(cfg_path.read_text())
    if config.get("schedule", "wsd") != "wsd" or config.get("decay_start") is not None:
        raise ValueError(f"{job_dir} decays; only constant-LR trunks can be extended")
    if max_steps <= config["max_steps"]:
        raise ValueError(f"new final step {max_steps} is not past {config['max_steps']}")
    config.setdefault("history", []).append({
        "max_steps": config["max_steps"], "keep_steps": config.get("keep_steps", []),
        "extended": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    config["max_steps"] = max_steps
    config["keep_steps"] = sorted(set(config.get("keep_steps", [])) | set(keep_steps or []))
    cfg_path.write_text(json.dumps(config, indent=2))
    return config


def create_branch(job_dir: Path, at_step: int, end_step: int, name: str) -> Path:
    """
    Branch a WSD cooldown off a kept trunk checkpoint: a new job dir whose HF dir links the
    trunk's ``checkpoint-<at_step>``, decaying from *at_step* to *end_step* and exporting at the end.

    :param job_dir: Trunk job dir.
    :param at_step: Kept trunk checkpoint to branch from.
    :param end_step: Final step of the branch.
    :param name: Branch name (dir under ``<job_dir>/branches``).
    :return: The branch job dir.
    :raises FileNotFoundError: If the trunk checkpoint is missing or incomplete.
    """
    config = json.loads((job_dir / "config.json").read_text())
    if config.get("schedule", "wsd") != "wsd" or config.get("decay_start") is not None:
        raise ValueError(f"{job_dir} decays; branch from a constant-LR trunk")
    src = job_dir / "hf_trainer" / f"checkpoint-{at_step}"
    if not (src / "trainer_state.json").exists():
        raise FileNotFoundError(f"no complete trunk checkpoint {src}")
    if not at_step < end_step:
        raise ValueError(f"branch start {at_step} is not before its end {end_step}")
    branch = job_dir / "branches" / name
    if (branch / "config.json").exists():
        raise FileExistsError(f"{branch} already exists")
    (branch / "hf_trainer").mkdir(parents=True)
    (branch / "hf_trainer" / src.name).symlink_to(src.resolve())
    config.update({
        "output_dir": str(branch.parent),
        "exp_name": name,
        "parent": str(job_dir),
        "train_manifest": str(job_dir / "train_manifest.jsonl"),
        "eval_manifest": str(job_dir / "val_subset_manifest.jsonl"),
        "decay_start": at_step,
        "max_steps": end_step,
        "keep_steps": [],
        "export_steps": [],
        "export_final": True,
        "history": [],
    })
    (branch / "config.json").write_text(json.dumps(config, indent=2))
    from lyricscribe.finetune.jobs import write_status

    write_status(branch, config, "pending")
    return branch
