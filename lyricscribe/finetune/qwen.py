import json
import logging
import random
import re
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from peft.utils import ModulesToSaveWrapper
from transformers import (
    AutoProcessor,
    MusicFlamingoForConditionalGeneration,
    Qwen3ASRForConditionalGeneration,
    Qwen3OmniMoeForConditionalGeneration,
)

from lyricscribe.finetune import engine
from lyricscribe.finetune.engine import (
    FeatureLayout,
    FinetuneTrainer,
    ManifestDataset,
    TrainingParts,
    configured_prompt_mode,
    drop_bad,
    training_processor,
)
from lyricscribe.transcribe.audio_llm import PROMPT_MODES, audio_llm_inputs, audio_llm_prompt, load_omni_adapter
from lyricscribe.transcribe.chunking import language_name
from lyricscribe.transcribe.base import shared_prompt
from lyricscribe.transcribe.qwen_asr import Qwen3ASRTranscriber, _build_prompt, _encode_audio_per_item

logger = logging.getLogger(__name__)

# NVIDIA's speechlm2 fine-tuning LoRA block (make_ft_config.py).
NVIDIA_LORA = {"r": 32, "lora_alpha": 64, "lora_dropout": 0.0}

# Qwen-family features are ``(batch, mel bins, frames)``.
QWEN_LAYOUT = FeatureLayout(key="input_features", time_dim=2, freq_dim=1)

# NVIDIA recipe mapped onto the Omni thinker: LoRA on the LLM's attention only,
# routers and routed experts frozen; the audio tower (encoder + proj1/proj2) trains at full rank.
OMNI_LORA_MODULES = r"model\.language_model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
OMNI_FULL_RANK = ["audio_tower"]

# NVIDIA's target names applied literally to a dense LLM (Music Flamingo mapping):
# attention and mlp.{up,down}_proj, not mlp.gate_proj.
DENSE_LORA_MODULES = (
    r"model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|mlp\.(?:up_proj|down_proj))"
)
# Qwen3-ASR adapter control: HF 5.22.1 keeps the encoder (Qwen3ASREncoder)
# at model.audio_tower and the projector at model.multi_modal_projector.
QWEN3_ASR_LORA_MODULES = DENSE_LORA_MODULES
QWEN3_ASR_FULL_RANK = ["audio_tower", "multi_modal_projector"]


def _pad_stack(tensors: list[torch.Tensor], key: str) -> torch.Tensor:
    """
    Stack per-item tensors, right-padding the last dim to the batch max.

    Widths differ when a window's audio ends before its manifest ``duration``.

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

    :param prompt_ids: Per-item prompt token ids.
    :param target_ids: Per-item target token ids (transcript + EOS).
    :param pad_id: Padding token id.
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


def _target_ids(tokenizer, text: str) -> torch.Tensor:
    """
    Transcript followed by the end-of-turn token.

    :param tokenizer: Model tokenizer.
    :param text: Transcript.
    :return: 1-D token ids.
    """
    return tokenizer(text + tokenizer.eos_token, add_special_tokens=False, return_tensors="pt")["input_ids"][0]


class Qwen3ASRCollator:
    """
    The exact prompt inference sends (:func:`~lyricscribe.transcribe.qwen_asr._build_prompt`
    with the language forced and an empty context, or the shared prompt as context, processed
    with its audio so the audio token is expanded to its true length) followed by the
    transcript + EOS; the prompt is masked out of the loss.

    :param processor: ``Qwen3ASRProcessor``.
    :param prompt_mode: ``native`` or ``shared``.
    """

    def __init__(self, processor, prompt_mode: str = "native"):
        """
        Keep the processor and resolve the pad id (EOS when the tokenizer has none).

        :param processor: ``Qwen3ASRProcessor``.
        :param prompt_mode: ``native`` or ``shared``.
        """
        self.processor = processor
        self.prompt_mode = prompt_mode
        self.tokenizer = processor.tokenizer
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def __call__(self, features: list[dict]) -> dict:
        """
        Collate dataset items into a training batch.

        :param features: :class:`~lyricscribe.finetune.engine.ManifestDataset` items.
        :return: Token ids, attention mask, labels, the padded audio features and the
            trainer's ``_feature_lengths`` / ``_placeholder``.
        """
        features = drop_bad(features)
        prompts, targets, feats, masks = [], [], [], []
        for item in features:
            context = shared_prompt(item["language"]) if self.prompt_mode == "shared" else ""
            prompt = _build_prompt(context, language_name(item["language"]))
            enc = self.processor(text=[prompt], audio=[item["audio"]], return_tensors="pt")
            prompts.append(enc["input_ids"][0])
            targets.append(_target_ids(self.tokenizer, item["text"]))
            feats.append(enc["input_features"][0])
            masks.append(enc["input_features_mask"][0])
        input_ids, attention, labels = _concat_prompt_target(prompts, targets, self.pad_id)
        mask = _pad_stack(masks, "input_features_mask")
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "input_features": _pad_stack(feats, "input_features"),
            "input_features_mask": mask,
            "_feature_lengths": mask.sum(-1),
            "_placeholder": any(it.get("placeholder") for it in features),
        }


class AudioLLMCollator:
    """
    Qwen3-Omni / Music Flamingo batches: each window's inputs come from
    :func:`~lyricscribe.transcribe.audio_llm.audio_llm_inputs`, exactly as inference
    builds them, followed by the transcript + EOS; the prompt is masked out.

    :param processor: The family's processor.
    :param family: ``"qwen3_omni"`` or ``"musicflamingo"``.
    :param prompt_mode: :data:`~lyricscribe.transcribe.audio_llm.PROMPT_MODES` entry.
    :param mask_key: The processor's per-frame feature mask key.
    """

    def __init__(self, processor, family: str, prompt_mode: str, mask_key: str):
        """
        Keep the processor and resolve the pad id (EOS when the tokenizer has none).

        :param processor: The family's processor.
        :param family: Audio-LLM family.
        :param prompt_mode: Prompt mode.
        :param mask_key: Feature mask key.
        """
        self.processor = processor
        self.family = family
        self.prompt_mode = prompt_mode
        self.mask_key = mask_key
        self.tokenizer = processor.tokenizer
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def __call__(self, features: list[dict]) -> dict:
        """
        Collate dataset items into a training batch.

        :param features: :class:`~lyricscribe.finetune.engine.ManifestDataset` items.
        :return: Token ids, attention mask, labels, the processor's audio tensors and the
            trainer's ``_feature_lengths`` / ``_placeholder``.
        """
        features = drop_bad(features)
        prompts, targets, extras = [], [], []
        for item in features:
            prompt = audio_llm_prompt(self.family, self.prompt_mode, item["language"])
            enc = audio_llm_inputs(self.processor, self.family, prompt, item["audio"])
            prompts.append(enc["input_ids"][0])
            targets.append(_target_ids(self.tokenizer, item["text"]))
            # The processor's attention_mask covers the prompt only.
            extras.append({k: v for k, v in enc.items() if k not in ("input_ids", "attention_mask")})
        input_ids, attention, labels = _concat_prompt_target(prompts, targets, self.pad_id)
        batch = {"input_ids": input_ids, "attention_mask": attention, "labels": labels}
        for key in extras[0]:
            tensors = [e[key] for e in extras]
            if not torch.is_tensor(tensors[0]):
                continue
            if self.family == "musicflamingo" and key in ("input_features", self.mask_key):
                # One row per 30 s audio chunk, not per window.
                batch[key] = torch.cat(tensors, dim=0)
            else:
                batch[key] = _pad_stack([t[0] if t.dim() > 1 else t for t in tensors], key)
        batch["_feature_lengths"] = batch[self.mask_key].sum(-1)
        batch["_placeholder"] = any(it.get("placeholder") for it in features)
        return batch


@torch.no_grad()
def mean_token_ce(model, collator, dataset, batch_size: int = 2) -> float:
    """
    Mean per-token cross-entropy over the transcript positions of *dataset*.

    :param model: Model (or PEFT-wrapped thinker) on its device.
    :param collator: Matching collator.
    :param dataset: :class:`~lyricscribe.finetune.engine.ManifestDataset`.
    :param batch_size: Windows per forward pass.
    :return: Mean CE in nats per transcript token.
    """
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    total, tokens = 0.0, 0
    for start in range(0, len(dataset), batch_size):
        batch = collator([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v) and not k.startswith("_")}
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


def nvidia_adapter(model, modules: str, full_rank: list[str]):
    """
    NVIDIA's speechlm2 fine-tuning recipe on an HF audio-LLM: the audio
    encoder and projection at full rank (``modules_to_save``), LoRA r32/α64 without dropout on
    the LLM modules matched by *modules*, routers, routed experts and everything else frozen;
    trainable weights in fp32 over the frozen base.

    :param model: The base model.
    :param modules: Regex of LLM modules that get LoRA.
    :param full_rank: Module names trained at full rank.
    :return: The PEFT model.
    """
    lora = LoraConfig(**NVIDIA_LORA, bias="none", target_modules=modules, modules_to_save=full_rank)
    peft_model = get_peft_model(model, lora)
    for p in peft_model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    summary = engine.parameter_summary(peft_model)
    logger.info(f"NVIDIA recipe: {summary['trainable'] / 1e6:.1f}M trainable of {summary['total'] / 1e9:.2f}B")
    return peft_model


def adapter_summary(peft_model) -> dict:
    """
    What a PEFT model actually adapts, read from the wrapped modules rather than the config.

    :param peft_model: The PEFT model.
    :return: LoRA settings, wrapped LoRA modules (layer index as ``N``) with counts, and the
        full-rank ``modules_to_save`` copies.
    """
    targets: dict[str, int] = {}
    full_rank = []
    for name, module in peft_model.named_modules():
        name = name.removeprefix("base_model.model.")
        if isinstance(module, ModulesToSaveWrapper):
            full_rank.append(name)
        elif hasattr(module, "lora_A") and not name.endswith((".lora_A", ".lora_B")):
            key = re.sub(r"\.\d+\.", ".N.", name)
            targets[key] = targets.get(key, 0) + 1
    cfg = peft_model.peft_config["default"]
    return {
        "lora": {"r": cfg.r, "lora_alpha": cfg.lora_alpha, "lora_dropout": cfg.lora_dropout,
                 "target_modules": sorted(cfg.target_modules) if isinstance(cfg.target_modules, set) else cfg.target_modules},
        "lora_targets": targets,
        "modules_to_save": full_rank,
        "trainable": engine.parameter_summary(peft_model),
    }


class _QwenTrainerBase:
    """
    Shared trainer for the audio-LLM style models: :meth:`parts` for
    :func:`lyricscribe.finetune.engine.run`, and :meth:`export`.

    :param config: Job configuration dictionary.
    """

    loss_mode = "model"
    layout = QWEN_LAYOUT
    adapter = False

    def __init__(self, config: dict):
        """
        Store the config; the model loads in :meth:`parts`.

        :param config: Job configuration dictionary.
        """
        self.config = config
        self.model = None
        self.processor = None

    def _load(self, path: str) -> None:
        """
        Load ``self.model`` and ``self.processor``.

        :param path: Hub id or local checkpoint path.
        :raises NotImplementedError: On the base class.
        """
        raise NotImplementedError

    def collator(self):
        """
        The training collator.

        :return: A collator.
        :raises NotImplementedError: On the base class.
        """
        raise NotImplementedError

    def _gc_args(self) -> dict:
        """
        Gradient checkpointing (non-reentrant: reentrant breaks DDP with "marked as ready twice").

        :return: ``TrainingArguments`` fields.
        """
        return {"gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False}}

    def parts(self, manifest: Path, seed: int, eval_manifest: Path | None = None) -> TrainingParts:
        """
        The pieces of a run.

        :param manifest: Training manifest.
        :param seed: Run seed.
        :param eval_manifest: Held-out windows scored at each export, or ``None``.
        :return: Model, datasets, collator, trainer class, loss mode and feature layout.
        """
        if self.model is None:
            self._load(self.config["base_model"])
        return TrainingParts(
            model=self.model,
            dataset=ManifestDataset(manifest, seed=seed, dither=engine.DITHER),
            eval_dataset=ManifestDataset(eval_manifest, seed=seed) if eval_manifest else None,
            collator=self.collator(),
            trainer_cls=FinetuneTrainer,
            loss_mode=self.loss_mode,
            layout=self.layout,
            per_device=int(self.config["batch_size"]),
            args=self._gc_args(),
            export=self.export,
            notes=self.notes(),
            prompt_mode=self.prompt_mode(),
        )

    def prompt_mode(self) -> str:
        """
        The training prompt, as the inference loader names it; subclasses with a choice override it.

        :return: ``"native"`` (the model's own transcription prompt).
        """
        return configured_prompt_mode(self.config, ("native",))

    def notes(self) -> dict:
        """
        Recipe facts recorded in the training state.

        :return: Notes (for adapters, what the wrapped model actually adapts).
        """
        if self.adapter:
            return {"adaptation": "NVIDIA recipe", **adapter_summary(self.model)}
        return {"adaptation": "full FT, fp32 master weights"}

    def export(self, path: Path) -> Path:
        """
        Save the model (or adapter) and processor for inference.

        :param path: Output dir.
        :return: *path*.
        """
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        return path

class Qwen3ASRTrainer(_QwenTrainerBase):
    """
    Qwen3-ASR (``*-hf`` checkpoints): full FT with fp32 master weights, or, with
    ``adaptation: nvidia_adapter``, the adapter control (frozen bf16 base,
    full-rank encoder + projector, LoRA on the dense LLM).
    """

    def __init__(self, config: dict):
        """
        Store the config and the adaptation.

        :param config: Job configuration dictionary.
        """
        super().__init__(config)
        self.adapter = config.get("adaptation") == "nvidia_adapter"

    def _load(self, path: str) -> None:
        """
        Load a transformers-native Qwen3-ASR checkpoint, encoding audio item by item as
        inference does, and wrap it in an adapter for the adapter recipe.

        :param path: Hub id or local checkpoint path.
        """
        logger.info(f"Loading Qwen3-ASR: {path} ({'adapter recipe' if self.adapter else 'full FT'})")
        self.processor = AutoProcessor.from_pretrained(path)
        # The adapter recipe's frozen base is bf16; ``base_dtype`` overrides it for CPU checks.
        dtype = getattr(torch, self.config.get("base_dtype") or "bfloat16") if self.adapter else torch.float32
        model = Qwen3ASRForConditionalGeneration.from_pretrained(path, dtype=dtype, device_map=engine.device_map())
        model.config.use_cache = False
        _encode_audio_per_item(model)
        # Qwen3-ASR ships sampling fields alongside do_sample=False, which
        # save_pretrained() rejects; they are unused under greedy decoding.
        gc = model.generation_config
        if gc is not None and not getattr(gc, "do_sample", False):
            for field in ("temperature", "top_p", "top_k", "typical_p", "min_p"):
                if getattr(gc, field, None) is not None:
                    setattr(gc, field, None)
        self.model = nvidia_adapter(model, QWEN3_ASR_LORA_MODULES, QWEN3_ASR_FULL_RANK) if self.adapter else model

    def collator(self):
        """
        The training collator.

        :return: :class:`Qwen3ASRCollator`.
        """
        return Qwen3ASRCollator(training_processor(self.processor), self.prompt_mode())

    def prompt_mode(self) -> str:
        """
        The training prompt, as :class:`~lyricscribe.transcribe.qwen_asr.Qwen3ASRTranscriber` names it.

        :return: The config's ``prompt_mode`` (default ``native``).
        """
        return configured_prompt_mode(self.config, Qwen3ASRTranscriber.prompt_modes)

    def export(self, path: Path) -> Path:
        """
        Save the full model (checking the inference architecture name) or the adapter, with the processor.

        :param path: Output dir.
        :return: *path*.
        """
        super().export(path)
        if not self.adapter:
            cfg = json.loads((path / "config.json").read_text())
            assert cfg.get("architectures") == ["Qwen3ASRForConditionalGeneration"], cfg.get("architectures")
        return path

def load_omni_thinker(base_model: str, device=None):
    """
    Load the Qwen3-Omni thinker (the module that actually has a ``forward``) in bf16.

    ``Qwen3OmniMoeForConditionalGeneration`` is a router over thinker/talker
    that defines only ``generate``; training it directly hits ``_forward_unimplemented``.

    :param base_model: Hub id or path of the full Omni checkpoint.
    :param device: Device map target (default: this rank's GPU).
    :return: ``(thinker, processor)``.
    """
    processor = AutoProcessor.from_pretrained(base_model)
    full = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        base_model, dtype=torch.bfloat16, device_map=device or engine.device_map())
    full.disable_talker()
    thinker = full.thinker
    thinker.config.use_cache = False
    assert type(thinker).forward is not torch.nn.Module.forward, \
        "Omni thinker has no forward() -- check the transformers version"
    return thinker, processor


class Qwen3OmniLoraTrainer(_QwenTrainerBase):
    """
    NVIDIA's adapter recipe on the Qwen3-Omni thinker (:func:`nvidia_adapter` with the Omni
    mapping); checkpoints are PEFT adapters holding the LoRA weights and the full-rank audio tower.
    """

    adapter = True

    def _load(self, path: str) -> None:
        """
        Load the Omni thinker and wrap it with the adapter.

        :param path: Hub id or path of the full Omni checkpoint.
        """
        thinker, self.processor = load_omni_thinker(path)
        self.model = nvidia_adapter(thinker, OMNI_LORA_MODULES, OMNI_FULL_RANK)

    def collator(self):
        """
        The training collator.

        :return: :class:`AudioLLMCollator` for Omni.
        """
        return AudioLLMCollator(
            training_processor(self.processor), "qwen3_omni", self.prompt_mode(), "feature_attention_mask"
        )

    def prompt_mode(self) -> str:
        """
        The training prompt.

        :return: The config's ``prompt_mode`` (default ``"lyrics"``).
        """
        return configured_prompt_mode(self.config, PROMPT_MODES, "lyrics")

    def notes(self) -> dict:
        """
        Recipe facts recorded in the training state.

        :return: Notes.
        """
        return {"adaptation": "NVIDIA recipe mapped to Omni", **adapter_summary(self.model)}


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
        ``"qwen3_omni"`` (checkpoints are adapters on *base_model*).
    :param base_model: Base hub id or path.
    :param checkpoints: Checkpoint dirs to compare against the base.
    :param manifest: Manifest to sample windows from (use one not trained on).
    :param rows: Number of windows.
    :param seed: Sampling seed.
    :param batch_size: Windows per forward pass.
    :return: ``{"base": ce, <checkpoint>: ce, ...}``.
    """
    torch.backends.cuda.enable_cudnn_sdp(False)
    dataset = ManifestDataset(manifest)
    if rows and rows < len(dataset.rows):
        dataset.rows = random.Random(seed).sample(dataset.rows, rows)
    results = {}
    if architecture == "qwen3_asr":
        collator = Qwen3ASRCollator(AutoProcessor.from_pretrained(base_model))
        for name in [base_model, *checkpoints]:
            model = Qwen3ASRForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16).to("cuda:0")
            _encode_audio_per_item(model)
            results["base" if name == base_model else name] = mean_token_ce(model, collator, dataset, batch_size)
            logger.info(f"{name}: CE {results['base' if name == base_model else name]:.4f}")
            del model
            torch.cuda.empty_cache()
    elif architecture == "qwen3_omni":
        thinker, processor = load_omni_thinker(base_model, device="cuda:0")
        collator = AudioLLMCollator(processor, "qwen3_omni", "lyrics", "feature_attention_mask")
        results["base"] = mean_token_ce(thinker, collator, dataset, batch_size)
        logger.info(f"base: CE {results['base']:.4f}")
        for path in checkpoints:
            peft = load_omni_adapter(thinker, path)
            results[path] = mean_token_ce(peft, collator, dataset, batch_size)
            logger.info(f"{path}: CE {results[path]:.4f}")
            thinker = peft.unload()
    else:
        raise ValueError(f"eval-loss supports qwen3_asr / qwen3_omni, got {architecture}")
    return results


# NVIDIA recipe mapped onto Music Flamingo's dense Qwen2.5 LLM; encoder and projector at full rank.
MF_LORA_MODULES = DENSE_LORA_MODULES
MF_FULL_RANK = ["audio_tower", "multi_modal_projector"]


class MusicFlamingoTrainer(_QwenTrainerBase):
    """
    NVIDIA's adapter recipe on Music Flamingo (~8B);
    checkpoints are PEFT adapters holding the LoRA weights, the audio tower and the projector.
    """

    adapter = True

    def _load(self, path: str) -> None:
        """
        Load Music Flamingo and wrap it with the adapter. The frozen base is bf16 by
        default (its checkpoint dtype, as for Omni and Nemotron); ``base_dtype`` overrides it.

        :param path: Hub id or path of the base checkpoint.
        """
        dtype = getattr(torch, self.config.get("base_dtype") or "bfloat16")
        logger.info(f"Loading Music Flamingo: {path} ({dtype} base)")
        self.processor = AutoProcessor.from_pretrained(path)
        model = MusicFlamingoForConditionalGeneration.from_pretrained(path, dtype=dtype, device_map=engine.device_map())
        model.config.use_cache = False
        self.model = nvidia_adapter(model, MF_LORA_MODULES, MF_FULL_RANK)

    def collator(self):
        """
        The training collator.

        :return: :class:`AudioLLMCollator` for Music Flamingo.
        """
        return AudioLLMCollator(
            training_processor(self.processor), "musicflamingo", self.prompt_mode(), "input_features_mask"
        )

    def prompt_mode(self) -> str:
        """
        The training prompt.

        :return: The config's ``prompt_mode`` (default ``"asr"``).
        """
        return configured_prompt_mode(self.config, PROMPT_MODES, "asr")

    def notes(self) -> dict:
        """
        Recipe facts recorded in the training state.

        :return: Notes.
        """
        return {
            "adaptation": "NVIDIA recipe mapped to Music Flamingo",
            "base_dtype": self.config.get("base_dtype") or "bfloat16",
            **adapter_summary(self.model),
        }
