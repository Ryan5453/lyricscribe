import json
import logging
from pathlib import Path

import torch
from transformers import AutoProcessor, GraniteSpeechForConditionalGeneration

from lyricscribe.finetune.engine import FeatureLayout, configured_prompt_mode, drop_bad, training_processor
from lyricscribe.finetune.qwen import _concat_prompt_target, _QwenTrainerBase, _target_ids
from lyricscribe.transcribe.base import shared_prompt
from lyricscribe.transcribe.chunking import language_name
from lyricscribe.transcribe.granite import (
    ASR_PROMPT,
    LANGUAGE_PROMPT,
    PROMPT_LANGUAGES,
    GraniteTranscriber,
    apply_logits_scaling,
)

logger = logging.getLogger(__name__)

# Granite stacks 2 frames of 80 mel bins per row; lengths are counted in rows.
GRANITE_LAYOUT = FeatureLayout(key="input_features", time_dim=1, freq_dim=2, stack=2)


def granite_prompt(tokenizer, language: str, prompt_mode: str = "asr") -> str:
    """
    The chat-templated prompt inference sends for *language*.

    :param tokenizer: Granite tokenizer.
    :param language: Language code.
    :param prompt_mode: ``asr`` (the card's language-ID ASR prompt) or ``shared``.
    :return: Prompt text ending in the generation prefix.
    """
    if prompt_mode == "shared":
        text = shared_prompt(language)
    else:
        name = language_name(language)
        text = LANGUAGE_PROMPT.format(language=name) if name in PROMPT_LANGUAGES else ASR_PROMPT
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": "<|audio|>" + text}], tokenize=False, add_generation_prompt=True
    )


class GraniteCollator:
    """
    Granite Speech LLM batches: the inference prompt (processed with its audio so the
    audio token is expanded) followed by the transcript + EOS; the prompt is masked out.

    :param processor: ``GraniteSpeechProcessor``.
    :param prompt_mode: ``asr`` or ``shared``.
    """

    def __init__(self, processor, prompt_mode: str = "asr"):
        """
        Keep the processor and resolve the pad id (EOS when the tokenizer has none).

        :param processor: ``GraniteSpeechProcessor``.
        :param prompt_mode: ``asr`` or ``shared``.
        """
        self.processor = processor
        self.prompt_mode = prompt_mode
        self.tokenizer = processor.tokenizer
        self.pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

    def __call__(self, features: list[dict]) -> dict:
        """
        Collate dataset items into a training batch.

        :param features: :class:`~lyricscribe.finetune.engine.ManifestDataset` items.
        :return: Token ids, attention mask, labels, stacked features and their mask, and the
            trainer's ``_feature_lengths`` / ``_placeholder``.
        """
        features = drop_bad(features)
        prompts, targets, feats, masks = [], [], [], []
        for item in features:
            enc = self.processor(
                [granite_prompt(self.tokenizer, item["language"], self.prompt_mode)],
                [torch.from_numpy(item["audio"])],
                return_tensors="pt",
            )
            prompts.append(enc["input_ids"][0])
            targets.append(_target_ids(self.tokenizer, item["text"]))
            feats.append(enc["input_features"][0])
            masks.append(enc["input_features_mask"][0])
        input_ids, attention, labels = _concat_prompt_target(prompts, targets, self.pad_id)
        rows = max(f.shape[0] for f in feats)
        width = max(m.shape[0] for m in masks)
        mask = torch.stack([torch.nn.functional.pad(m, (0, width - m.shape[0])) for m in masks])
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "input_features": torch.stack([torch.nn.functional.pad(f, (0, 0, 0, rows - f.shape[0])) for f in feats]),
            "input_features_mask": mask,
            # Each item's own rows (processed alone, so unpadded); the mask counts projector tokens.
            "_feature_lengths": torch.tensor([f.shape[0] for f in feats]),
            "_placeholder": any(it.get("placeholder") for it in features),
        }


class GraniteTrainer(_QwenTrainerBase):
    """
    Full finetune of the Granite Speech LLM (5.0-2B), fp32 master weights. Its built-in loss
    is a per-batch mean, so the engine's causal token loss is used instead. The encoder
    takes no padding mask, so it trains one window per device.
    """

    loss_mode = "causal"
    layout = GRANITE_LAYOUT
    logits_scaling = None

    def _load(self, path: str) -> None:
        """
        Load a Granite Speech LLM checkpoint in fp32.

        :param path: Hub id or local checkpoint path.
        """
        logger.info(f"Loading Granite Speech: {path}")
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = GraniteSpeechForConditionalGeneration.from_pretrained(path, dtype=torch.float32)
        self.model.config.use_cache = False
        if getattr(self.model.config, "has_lora_adapter", False):
            raise ValueError(f"{path} ships a LoRA adapter; full FT of adapter checkpoints is not wired")
        self.logits_scaling = apply_logits_scaling(self.model)

    def notes(self) -> dict:
        """
        Recipe facts recorded in the training state.

        :return: Notes, including the logits scaling divided out of the loss.
        """
        return {**super().notes(), "logits_scaling": self.logits_scaling}

    def _gc_args(self) -> dict:
        """
        No gradient checkpointing: ``GraniteSpeechForConditionalGeneration`` doesn't support it.

        :return: No extra ``TrainingArguments`` fields.
        """
        return {}

    def collator(self):
        """
        The training collator.

        :return: :class:`GraniteCollator`.
        """
        return GraniteCollator(training_processor(self.processor), self.prompt_mode())

    def prompt_mode(self) -> str:
        """
        The training prompt, as :class:`~lyricscribe.transcribe.granite.GraniteTranscriber` names it.

        :return: The config's ``prompt_mode`` (default ``asr``, the card's ASR prompt).
        """
        return configured_prompt_mode(self.config, GraniteTranscriber.prompt_modes)

    def parts(self, manifest: Path, seed: int, eval_manifest: Path | None = None):
        """
        The pieces of a run, refusing a per-device batch other than 1.

        :param manifest: Training manifest.
        :param seed: Run seed.
        :param eval_manifest: Held-out windows scored at each export, or ``None``.
        :return: The training parts.
        :raises ValueError: If ``batch_size`` is not 1 (the encoder takes no padding mask).
        """
        if int(self.config["batch_size"]) != 1:
            raise ValueError(f"Granite trains with per-device batch 1 (no encoder padding mask), not {self.config['batch_size']}")
        return super().parts(manifest, seed, eval_manifest)

    def export(self, path: Path) -> Path:
        """
        Save the full model and processor (checking the inference architecture name).

        :param path: Output dir.
        :return: *path*.
        """
        super().export(path)
        cfg = json.loads((path / "config.json").read_text())
        assert cfg.get("architectures") == ["GraniteSpeechForConditionalGeneration"], cfg.get("architectures")
        return path
