import json
import logging
from pathlib import Path

import torch
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
from transformers.models.cohere_asr.processing_cohere_asr import LANGUAGES

from lyricscribe.finetune.engine import FeatureLayout, configured_prompt_mode, drop_bad, training_processor
from lyricscribe.finetune.qwen import _QwenTrainerBase
from lyricscribe.transcribe.base import shared_prompt
from lyricscribe.transcribe.cohere import CohereTranscriber

logger = logging.getLogger(__name__)


class CohereCollator:
    """
    Cohere Transcribe batches: the decoder prompt inference sends
    (``get_decoder_prompt_ids`` for the window's language, punctuation on, and for
    ``shared`` the shared prompt in the context slot after ``<|startofcontext|>``)
    followed by the transcript and EOS, with labels on the transcript and EOS
    only. Audio is clipped to the processor's single-chunk limit so a window is
    never split into chunks that would share one label.

    :param processor: ``CohereAsrProcessor``.
    :param prompt_mode: ``native`` or ``shared``.
    """

    def __init__(self, processor, prompt_mode: str = "native"):
        """
        Keep the processor and resolve the special ids and the clip length.

        :param processor: ``CohereAsrProcessor``.
        :param prompt_mode: ``native`` or ``shared``.
        """
        self.processor = processor
        self.prompt_mode = prompt_mode
        self.tokenizer = processor.tokenizer
        self.pad_id = self.tokenizer.pad_token_id
        self.eos_id = self.tokenizer.eos_token_id
        fe = processor.feature_extractor
        self.max_samples = int(fe.sampling_rate * max(0.0, fe.max_audio_clip_s - fe.overlap_chunk_second))

    def __call__(self, features: list[dict]) -> dict:
        """
        Collate dataset items into a training batch.

        :param features: :class:`~lyricscribe.finetune.engine.ManifestDataset` items.
        :return: Audio features and mask, decoder inputs and mask, labels, and the
            trainer's ``_feature_lengths`` / ``_placeholder``.
        """
        features = drop_bad(features)
        audio = self.processor.feature_extractor(
            [item["audio"][: self.max_samples] for item in features],
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            padding="longest", return_attention_mask=True, return_tensors="pt",
        )
        sequences, prompt_lengths = [], []
        for item in features:
            language = item["language"] if item["language"] in LANGUAGES else "en"
            prompt = self.processor.get_decoder_prompt_ids(language=language)
            if self.prompt_mode == "shared":
                # Same insertion as inference (transcribe.canary.insert_context), with the window's own language.
                pos = prompt.index(self.tokenizer.convert_tokens_to_ids("<|startofcontext|>")) + 1
                ctx = self.tokenizer(shared_prompt(item["language"]), add_special_tokens=False)["input_ids"]
                prompt = prompt[:pos] + ctx + prompt[pos:]
            target = self.tokenizer(item["text"], add_special_tokens=False)["input_ids"] + [self.eos_id]
            sequences.append(prompt + target)
            prompt_lengths.append(len(prompt))
        width = max(len(s) for s in sequences) - 1
        decoder_input_ids = torch.full((len(sequences), width), self.pad_id, dtype=torch.long)
        decoder_attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
        labels = torch.full((len(sequences), width), -100, dtype=torch.long)
        for i, (seq, n_prompt) in enumerate(zip(sequences, prompt_lengths)):
            seq = torch.tensor(seq)
            decoder_input_ids[i, : len(seq) - 1] = seq[:-1]
            decoder_attention_mask[i, : len(seq) - 1] = 1
            # The forward does not shift: position i predicts token i + 1.
            labels[i, n_prompt - 1 : len(seq) - 1] = seq[n_prompt:]
        return {
            "input_features": audio["input_features"],
            "attention_mask": audio["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": labels,
            "_feature_lengths": audio["attention_mask"].sum(-1),
            "_placeholder": any(it.get("placeholder") for it in features),
        }


class CohereTrainer(_QwenTrainerBase):
    """
    Full finetune of Cohere Transcribe (Conformer encoder, Transformer decoder), fp32 master weights.
    """

    layout = FeatureLayout(key="input_features", time_dim=1, freq_dim=2)

    def _load(self, path: str) -> None:
        """
        Load a Cohere Transcribe checkpoint in fp32.

        :param path: Hub id or local checkpoint path.
        """
        logger.info(f"Loading Cohere Transcribe: {path}")
        self.processor = AutoProcessor.from_pretrained(path)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(path, dtype=torch.float32)
        self.model.config.use_cache = False

    def collator(self):
        """
        The training collator.

        :return: :class:`CohereCollator`.
        """
        # Its feature extractor dithers (1e-5) by default; training gets ManifestDataset's dither only.
        return CohereCollator(training_processor(self.processor), self.prompt_mode())

    def prompt_mode(self) -> str:
        """
        The training prompt, as :class:`~lyricscribe.transcribe.cohere.CohereTranscriber` names it.

        :return: The config's ``prompt_mode`` (default ``native``).
        """
        return configured_prompt_mode(self.config, CohereTranscriber.prompt_modes)

    def export(self, path: Path) -> Path:
        """
        Save the full model and processor (checking the inference architecture name).

        :param path: Output dir.
        :return: *path*.
        """
        super().export(path)
        cfg = json.loads((path / "config.json").read_text())
        assert cfg.get("architectures") == ["CohereAsrForConditionalGeneration"], cfg.get("architectures")
        return path
