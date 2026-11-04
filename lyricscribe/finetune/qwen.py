import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, TrainingArguments

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

LANGUAGE_NAMES = {
    "en": "English", "es": "Spanish", "de": "German", "fr": "French",
    "it": "Italian", "pt": "Portuguese", "ru": "Russian",
}

OMNI_PROMPT = (
    "Transcribe the sung lyrics in this audio. Output only the lyrics text, "
    "with no commentary. The language is {language}."
)

# LoRA target sets for the Omni thinker. "wide" adds the MoE expert FFNs,
# which PEFT wraps as ParamWrapper modules that reject nonzero dropout.
OMNI_LORA_TARGETS = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "wide": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def _is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", 0)) == 0


def _language_code(row: dict) -> str:
    return (row.get("language") or "en").split("-")[0].split("_")[0].lower()


class WindowDataset(Dataset):
    """
    Audio windows from a finetune manifest, decoded at 16 kHz mono.

    :param manifest: Manifest JSONL path.
    """

    def __init__(self, manifest: Path):
        with Path(manifest).open() as f:
            self.rows = [json.loads(line) for line in f if line.strip()]
        logger.info(f"Loaded {len(self.rows)} windows from {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        audio, _ = librosa.load(
            row["audio_filepath"], sr=SAMPLE_RATE, mono=True,
            offset=float(row.get("offset", 0.0)),
            duration=float(row["duration"]) if row.get("duration") else None,
        )
        return {
            "audio": np.asarray(audio, dtype=np.float32),
            "text": row["text"],
            "language": _language_code(row),
        }


def _pad_stack(tensors: list[torch.Tensor], key: str) -> torch.Tensor:
    """
    Stack per-item tensors, right-padding the last dim to the batch max.

    Widths differ when a window's audio ends before its manifest
    ``duration``.

    :param tensors: Per-item tensors (no batch dim).
    :param key: Feature name, for the one-time ragged-batch notice.
    :return: Batched tensor.
    """
    width = max(t.shape[-1] for t in tensors)
    widths = {t.shape[-1] for t in tensors}
    if len(widths) > 1 and not _pad_stack.ragged_seen:
        _pad_stack.ragged_seen = True
        logger.info(f"RAGGED_BATCH: {key} widths {sorted(widths)} -> padded to {width}")
    return torch.stack([
        t if t.shape[-1] == width else torch.nn.functional.pad(t, (0, width - t.shape[-1]))
        for t in tensors
    ])


_pad_stack.ragged_seen = False


def _concat_prompt_target(prompt_ids, target_ids, pad_id):
    """
    Right-pad prompt+target sequences; the prompt is masked out of the loss.

    :return: ``(input_ids, attention_mask, labels)`` batch tensors.
    """
    seqs = [torch.cat([p, t]) for p, t in zip(prompt_ids, target_ids)]
    width = max(s.shape[0] for s in seqs)
    input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    labels = torch.full((len(seqs), width), -100, dtype=torch.long)
    attention = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, (p, t, s) in enumerate(zip(prompt_ids, target_ids, seqs)):
        input_ids[i, : s.shape[0]] = s
        labels[i, p.shape[0] : s.shape[0]] = t
        attention[i, : s.shape[0]] = 1
    return input_ids, attention, labels


class Qwen3ASRCollator:
    """
    Prompt (processed with its audio, so the audio token is expanded to its
    true length) followed by the transcript + EOS. The prompt is exactly
    what native inference sends: ``apply_transcription_request`` with the
    language forced via the ``language <Name><asr_text>`` prefill.

    :param processor: ``Qwen3ASRProcessor``.
    """

    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def __call__(self, features: list[dict]) -> dict:
        prompts, targets, feats, masks = [], [], [], []
        for item in features:
            enc = self.processor.apply_transcription_request(
                audio=[item["audio"]], language=LANGUAGE_NAMES.get(item["language"], item["language"]),
            )
            prompts.append(enc["input_ids"][0])
            targets.append(self.tokenizer(
                item["text"] + self.tokenizer.eos_token,
                add_special_tokens=False, return_tensors="pt",
            )["input_ids"][0])
            feats.append(enc["input_features"][0])
            masks.append(enc["input_features_mask"][0])
        input_ids, attention, labels = _concat_prompt_target(prompts, targets, self.pad_id)
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "input_features": _pad_stack(feats, "input_features"),
            "input_features_mask": _pad_stack(masks, "input_features_mask"),
        }


class Qwen3OmniCollator:
    """
    Plain-instruction prompt + transcript for the Omni thinker.
    :data:`OMNI_PROMPT` must match the prompt used at inference.

    :param processor: ``Qwen3OmniMoeProcessor``.
    """

    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def _prompt(self, language: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio": ""},
            {"type": "text", "text": OMNI_PROMPT.format(language=language)},
        ]}]
        return self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)

    def __call__(self, features: list[dict]) -> dict:
        prompts, targets, extras = [], [], []
        for item in features:
            enc = self.processor(
                text=[self._prompt(LANGUAGE_NAMES.get(item["language"], item["language"]))],
                audio=[item["audio"]], sampling_rate=SAMPLE_RATE, return_tensors="pt",
            )
            prompts.append(enc["input_ids"][0])
            targets.append(self.tokenizer(
                item["text"] + self.tokenizer.eos_token,
                add_special_tokens=False, return_tensors="pt",
            )["input_ids"][0])
            # The processor's attention_mask covers the prompt only and must
            # not overwrite the prompt+target mask built below.
            extras.append({k: v for k, v in enc.items() if k not in ("input_ids", "attention_mask")})
        input_ids, attention, labels = _concat_prompt_target(prompts, targets, self.pad_id)
        batch = {"input_ids": input_ids, "attention_mask": attention, "labels": labels}
        for key in extras[0]:
            tensors = [e[key][0] if e[key].dim() > 1 else e[key] for e in extras]
            if torch.is_tensor(tensors[0]):
                batch[key] = _pad_stack(tensors, key)
        assert batch["attention_mask"].shape == batch["input_ids"].shape, (
            f"attention_mask {tuple(batch['attention_mask'].shape)} != input_ids "
            f"{tuple(batch['input_ids'].shape)} -- a processor key overwrote it")
        return batch


@torch.no_grad()
def mean_token_ce(model, collator, dataset, batch_size: int = 2) -> float:
    """
    Mean per-token cross-entropy over the transcript positions of *dataset*.

    Computed here (eval mode, causal shift, sum over masked tokens) rather
    than read from the Trainer, whose logged loss is accumulation-scaled.

    :param model: Model (or PEFT-wrapped thinker) on its device.
    :param collator: Matching collator.
    :param dataset: :class:`WindowDataset`.
    :param batch_size: Windows per forward pass.
    :return: Mean CE in nats per transcript token.
    """
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    total, tokens = 0.0, 0
    for start in range(0, len(dataset), batch_size):
        batch = collator([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
        batch = {k: v.to(device) for k, v in batch.items()}
        if "input_features" in batch:
            batch["input_features"] = batch["input_features"].to(dtype)
        labels = batch.pop("labels")
        logits = model(**batch).logits.float()
        shift_logits, shift_labels = logits[:, :-1, :], labels[:, 1:]
        mask = shift_labels != -100
        if mask.any():
            total += float(torch.nn.functional.cross_entropy(
                shift_logits[mask], shift_labels[mask], reduction="sum"))
            tokens += int(mask.sum())
    return total / max(tokens, 1)


def _slurm_seconds_left() -> float | None:
    """
    Remaining walltime of the current Slurm job, or ``None`` outside Slurm.
    """
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    try:
        left = subprocess.run(
            ["squeue", "-h", "-j", job, "-o", "%L"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    if not left or left in ("UNLIMITED", "NOT_SET", "INVALID"):
        return None
    days, _, rest = left.rpartition("-")
    parts = [int(p) for p in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts
    return (int(days) if days else 0) * 86400 + h * 3600 + m * 60 + s


class _StepBudgetCallback(TrainerCallback):
    """
    Log the step budget up front and, after a few steps, project the time to
    the end of the target epoch against the remaining Slurm walltime.

    :param probe_steps: Steps to time before projecting.
    """

    def __init__(self, probe_steps: int = 20):
        self.probe_steps = probe_steps
        self.target_epoch = None
        self._t0 = None
        self._step0 = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._t0, self._step0 = time.time(), state.global_step
        per_epoch = state.max_steps / max(args.num_train_epochs, 1e-9)
        logger.info(
            f"STEP_BUDGET: {state.max_steps} optimizer steps total "
            f"({per_epoch:.0f}/epoch), resuming at step {state.global_step}"
        )

    def on_step_end(self, args, state, control, **kwargs):
        done = state.global_step - (self._step0 or 0)
        if done != self.probe_steps or not _is_main_process():
            return
        per_step = (time.time() - self._t0) / done
        per_epoch = state.max_steps / max(args.num_train_epochs, 1e-9)
        target = self.target_epoch or args.num_train_epochs
        remaining = max(0.0, target * per_epoch - state.global_step) * per_step
        left = _slurm_seconds_left()
        msg = (f"STEP_BUDGET: {per_step:.2f} s/step -> {remaining / 3600:.2f} h to "
               f"epoch {target}")
        if left is not None:
            msg += f"; walltime left {left / 3600:.2f} h"
            if remaining > left:
                msg += (" -- WALLTIME_WARNING: epoch will not finish in this job; "
                        f"progress survives via step checkpoints every {args.save_steps} steps")
        logger.info(msg)


class _StopAfterEpochCallback(TrainerCallback):
    """
    Save and stop once ``target_epoch`` is reached (LR schedule spans the run).
    """

    def __init__(self):
        self.target_epoch = None

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.target_epoch is not None and state.epoch is not None and \
                state.epoch >= self.target_epoch - 1e-3:
            control.should_save = True
            control.should_training_stop = True
        return control


class _QwenTrainerBase:
    """
    Shared HF ``Trainer`` harness implementing the ``create_trainer`` protocol
    (``setup`` / ``train_one_epoch`` / ``save_checkpoint`` / ``load_checkpoint``).

    :param config: Job configuration dictionary.
    """

    collator_cls = None

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.processor = None
        self.hf_trainer = None
        self.train_manifest = None
        self.val_manifest = None
        self.job_dir = Path(config["output_dir"]) / config["exp_name"]

    # Subclasses: build self.model / self.processor from a path or hub id.
    def _load(self, path: str) -> None:
        raise NotImplementedError

    def _training_args(self) -> dict:
        return {}

    def setup(self, train_manifest: Path, val_manifest: Path | None) -> None:
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        if self.model is None:
            self._load(self.config["base_model"])

    def train_one_epoch(self, target_epoch: int) -> dict:
        train_dataset = WindowDataset(self.train_manifest)
        eval_dataset = WindowDataset(self.val_manifest) if self.val_manifest else None
        if self.hf_trainer is None:
            # ``warmup_steps`` reads values < 1 as a fraction of all steps and
            # >= 1 as an absolute step count, so clamp below 1.
            warmup_ratio = min(0.999, self.config["warmup_epochs"] / max(self.config["max_epochs"], 1))
            args = TrainingArguments(
                output_dir=str(self.job_dir / "hf_trainer"),
                num_train_epochs=self.config["max_epochs"],
                per_device_train_batch_size=self.config["batch_size"],
                per_device_eval_batch_size=self.config["batch_size"],
                gradient_accumulation_steps=self.config.get("grad_accum", 1),
                learning_rate=self.config["learning_rate"],
                warmup_steps=warmup_ratio,
                bf16=True,
                logging_steps=self.config.get("logging_steps", 10),
                save_strategy="steps",
                save_steps=self.config.get("save_steps", 500),
                save_total_limit=3,
                eval_strategy="no",
                gradient_checkpointing=True,
                # Reentrant checkpointing breaks DDP ("marked as ready twice").
                gradient_checkpointing_kwargs={"use_reentrant": False},
                dataloader_num_workers=self.config.get("dataloader_num_workers", 8),
                remove_unused_columns=False,
                ddp_find_unused_parameters=False,
                report_to=[],
                run_name=self.config["exp_name"],
                **self._training_args(),
            )
            self._stop = _StopAfterEpochCallback()
            self._budget = _StepBudgetCallback()
            self.hf_trainer = Trainer(
                model=self.model, args=args,
                train_dataset=train_dataset, eval_dataset=eval_dataset,
                data_collator=self.collator_cls(self.processor),
                callbacks=[self._stop, self._budget],
            )
        self._stop.target_epoch = target_epoch
        self._budget.target_epoch = target_epoch

        resume = None
        hf_dir = self.job_dir / "hf_trainer"
        if hf_dir.exists():
            for ckpt in sorted(hf_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]),
                               reverse=True):
                if (ckpt / "trainer_state.json").exists():
                    resume = str(ckpt)
                    break
                if _is_main_process():
                    logger.warning(f"Removing incomplete checkpoint: {ckpt}")
                    shutil.rmtree(ckpt, ignore_errors=True)
        if resume:
            logger.info(f"Resuming from HF step checkpoint: {resume}")
        logger.info(f"Training epoch {target_epoch}...")
        self.hf_trainer.train(resume_from_checkpoint=resume)

        metrics = {}
        if eval_dataset is not None:
            metrics = {k: float(v) for k, v in self.hf_trainer.evaluate().items()}
        for entry in reversed(self.hf_trainer.state.log_history):
            if "loss" in entry:
                metrics["train_loss_logged"] = entry["loss"]
                break
        return metrics


class Qwen3ASRTrainer(_QwenTrainerBase):
    """
    Full finetune of transformers-native Qwen3-ASR (``*-hf`` checkpoints).
    """

    collator_cls = Qwen3ASRCollator

    def _load(self, path: str) -> None:
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        logger.info(f"Loading Qwen3-ASR: {path}")
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(path, dtype=torch.bfloat16)
        self.model.config.use_cache = False
        # Qwen3-ASR ships sampling fields alongside do_sample=False, which
        # save_pretrained() rejects; they are unused under greedy decoding.
        gc = self.model.generation_config
        if gc is not None and not getattr(gc, "do_sample", False):
            for field in ("temperature", "top_p", "top_k", "typical_p", "min_p"):
                if getattr(gc, field, None) is not None:
                    logger.info(f"Clearing generation_config.{field}={getattr(gc, field)}")
                    setattr(gc, field, None)

    def save_checkpoint(self, epoch: int, checkpoint_dir: Path) -> Path:
        path = checkpoint_dir / f"epoch_{epoch}"
        if _is_main_process():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(path)
            self.processor.save_pretrained(path)
            cfg = json.loads((path / "config.json").read_text())
            assert cfg.get("architectures") == ["Qwen3ASRForConditionalGeneration"], cfg.get("architectures")
        return path

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        self._load(str(checkpoint_path))
        try:
            return int(checkpoint_path.name.split("_")[1])
        except (IndexError, ValueError):
            return 0


def load_omni_thinker(base_model: str, device: str = "cuda:0"):
    """
    Load the Qwen3-Omni thinker (the module that actually has a ``forward``).

    ``Qwen3OmniMoeForConditionalGeneration`` is a router over thinker/talker
    that defines only ``generate``; training it directly hits
    ``_forward_unimplemented``.

    :param base_model: Hub id or path of the full Omni checkpoint.
    :param device: Device map target.
    :return: ``(thinker, processor)``.
    """
    from transformers import AutoProcessor, Qwen3OmniMoeForConditionalGeneration

    processor = AutoProcessor.from_pretrained(base_model)
    full = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map=device)
    full.disable_talker()
    thinker = full.thinker
    thinker.config.use_cache = False
    assert type(thinker).forward is not torch.nn.Module.forward, \
        "Omni thinker has no forward() -- check the transformers version"
    return thinker, processor


class Qwen3OmniLoraTrainer(_QwenTrainerBase):
    """
    LoRA on the Qwen3-Omni thinker; checkpoints are PEFT adapters.
    """

    collator_cls = Qwen3OmniCollator

    def _load(self, path: str, adapter: str | None = None) -> None:
        from peft import LoraConfig, PeftModel, get_peft_model

        thinker, self.processor = load_omni_thinker(path)
        if adapter is not None:
            self.model = PeftModel.from_pretrained(thinker, adapter, is_trainable=True)
        else:
            targets = self.config.get("lora_targets", "attn")
            rank = self.config.get("lora_rank", 16)
            peft_config = LoraConfig(
                r=rank, lora_alpha=2 * rank, bias="none", task_type="CAUSAL_LM",
                target_modules=OMNI_LORA_TARGETS[targets],
                lora_dropout=0.0 if targets == "wide" else 0.05,
            )
            logger.info(f"LoRA rank={rank} targets={OMNI_LORA_TARGETS[targets]}")
            self.model = get_peft_model(thinker, peft_config)
        self.model.print_trainable_parameters()

    def save_checkpoint(self, epoch: int, checkpoint_dir: Path) -> Path:
        path = checkpoint_dir / f"epoch_{epoch}"
        if _is_main_process():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(path)
            self.processor.save_pretrained(path)
        return path

    def load_checkpoint(self, checkpoint_path: Path) -> int:
        self._load(self.config["base_model"], adapter=str(checkpoint_path))
        try:
            return int(checkpoint_path.name.split("_")[1])
        except (IndexError, ValueError):
            return 0


def eval_loss(
    architecture: str,
    base_model: str,
    checkpoints: list[str],
    manifest: Path,
    rows: int = 32,
    seed: int = 1234,
    batch_size: int = 2,
) -> dict[str, float]:
    """
    Held-out mean per-token CE of the base model and each checkpoint, on the
    same seeded sample of manifest windows.

    :param architecture: ``"qwen3_asr"`` (checkpoints are full models) or
        ``"qwen3_omni"`` (checkpoints are LoRA adapters on *base_model*).
    :param base_model: Base hub id or path.
    :param checkpoints: Checkpoint dirs to compare against the base.
    :param manifest: Manifest to sample windows from (use one not trained on).
    :param rows: Number of windows.
    :param seed: Sampling seed.
    :param batch_size: Windows per forward pass.
    :return: ``{"base": ce, <checkpoint>: ce, ...}``.
    """
    import random

    torch.backends.cuda.enable_cudnn_sdp(False)  # see _run_training_job_inner
    dataset = WindowDataset(manifest)
    if rows and rows < len(dataset.rows):
        dataset.rows = random.Random(seed).sample(dataset.rows, rows)
    results = {}
    if architecture == "qwen3_asr":
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        collator = Qwen3ASRCollator(AutoProcessor.from_pretrained(base_model))
        for name in [base_model, *checkpoints]:
            model = Qwen3ASRForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16).to("cuda:0")
            results["base" if name == base_model else name] = mean_token_ce(model, collator, dataset, batch_size)
            logger.info(f"{name}: CE {results['base' if name == base_model else name]:.4f}")
            del model
            torch.cuda.empty_cache()
    elif architecture == "qwen3_omni":
        from peft import PeftModel

        thinker, processor = load_omni_thinker(base_model)
        collator = Qwen3OmniCollator(processor)
        results["base"] = mean_token_ce(thinker, collator, dataset, batch_size)
        logger.info(f"base: CE {results['base']:.4f}")
        for path in checkpoints:
            peft = PeftModel.from_pretrained(thinker, path, is_trainable=False)
            results[path] = mean_token_ce(peft, collator, dataset, batch_size)
            logger.info(f"{path}: CE {results[path]:.4f}")
            thinker = peft.unload()
    else:
        raise ValueError(f"eval-loss supports qwen3_asr / qwen3_omni, got {architecture}")
    return results
