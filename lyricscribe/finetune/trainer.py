import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.modeling_whisper import shift_tokens_right

from lyricscribe.finetune import engine
from lyricscribe.finetune.cohere import CohereTrainer
from lyricscribe.finetune.engine import (
    SAMPLE_RATE,
    FeatureLayout,
    ManifestDataset,
    TrainingParts,
    configured_prompt_mode,
    sample_rng,
    training_processor,
)
from lyricscribe.finetune.fastconformer import FastConformerTrainer
from lyricscribe.finetune.granite import GraniteTrainer
from lyricscribe.finetune.nemotron import NemotronLoraTrainer
from lyricscribe.finetune.qwen import MusicFlamingoTrainer, Qwen3ASRTrainer, Qwen3OmniLoraTrainer
from lyricscribe.transcribe.base import shared_prompt
from lyricscribe.transcribe.whisper import WhisperTranscriber

logger = logging.getLogger(__name__)


class WhisperFinetuneDataset(torch.utils.data.Dataset):
    """
    Whisper finetuning windows from a manifest: log-mel features and token labels.

    Labels follow Whisper's dual-mode format: with probability
    ``timestamp_prob`` (when the entry carries ``segments``) the label is
    built in timestamp mode, otherwise in ``<|notimestamps|>`` mode. The
    choice is drawn per window and epoch from the run seed, so it is the
    same across workers and resumes. Training only one mode unlearns the
    other; long-form (>30s) decoding requires timestamps.

    :param manifest_path: Path to the JSONL manifest file.
    :param processor: ``WhisperProcessor`` for features and tokenization.
    :param timestamp_prob: Probability of a timestamp-mode label for entries
        that have ``segments``. Use 0.0 for eval datasets.
    :param seed: Run seed.
    :param dither: Waveform dither standard deviation.
    :param prompt_mode: ``native``, or ``shared`` for the shared prompt as a previous-text
        prompt (``<|startofprev|>`` ids, as inference's ``prompt_ids``) before the label prefix.
    """

    MAX_LABEL_TOKENS = 448
    TIME_PRECISION = 0.02  # seconds per timestamp token step
    HOP = 160

    def __init__(
        self,
        manifest_path: Path,
        processor: WhisperProcessor,
        timestamp_prob: float = 0.0,
        seed: int = 0,
        dither: float = 0.0,
        prompt_mode: str = "native",
    ):
        """
        Read the manifest and resolve the timestamp token range.

        :param manifest_path: Path to the JSONL manifest file.
        :param processor: ``WhisperProcessor`` for features and tokenization.
        :param timestamp_prob: Probability of a timestamp-mode label.
        :param seed: Run seed.
        :param dither: Waveform dither.
        :param prompt_mode: ``native`` or ``shared``.
        """
        self.processor = processor
        self.prompt_mode = prompt_mode
        self.timestamp_prob = timestamp_prob
        self.windows = ManifestDataset(manifest_path, seed=seed, dither=dither)
        self.entries = self.windows.rows
        self.seed = seed
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

    def __getitem__(self, idx: int) -> dict:
        """
        Load and preprocess one window.

        :param idx: Index into the manifest, optionally epoch-encoded.
        :return: ``input_features``, ``labels`` (with the decoder prefix), ``prefix_len``,
            ``prompt_ids`` (empty for ``native``) and ``frames``; ``bad`` is set for unreadable
            or over-cap windows.
        """
        item = self.windows[idx]
        entry = item["row"]
        if item["audio"] is None:
            return {"bad": True}
        audio = item["audio"]
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features[0]
        tokenizer = self.processor.tokenizer
        # predict_timestamps persists on the tokenizer, so set it on every call.
        labels = None
        draw = sample_rng(self.seed, item["epoch"], item["index"], "timestamps").random()
        if self.timestamp_prob > 0 and entry.get("segments") and draw < self.timestamp_prob:
            tokenizer.set_prefix_tokens(language=entry["language"], task="transcribe", predict_timestamps=True)
            labels = self._build_timestamp_labels(entry)
        if labels is None:
            tokenizer.set_prefix_tokens(language=entry["language"], task="transcribe", predict_timestamps=False)
            text = " " + entry["text"] if entry["text"] else entry["text"]
            labels = tokenizer(text).input_ids
            if len(labels) > self.MAX_LABEL_TOKENS:
                logger.warning(f"BAD_WINDOW {entry['audio_filepath']} @{entry.get('offset', 0)}: "
                               f"{len(labels)} label tokens > {self.MAX_LABEL_TOKENS}")
                return {"bad": True}
        prompt_ids = []
        if self.prompt_mode == "shared":
            prompt_ids = [int(i) for i in tokenizer.get_prompt_ids(shared_prompt(entry["language"]))]
            # The decoder sees prompt + labels[:-1], and its positions end at MAX_LABEL_TOKENS.
            if len(prompt_ids) + len(labels) - 1 > self.MAX_LABEL_TOKENS:
                logger.warning(f"BAD_WINDOW {entry['audio_filepath']} @{entry.get('offset', 0)}: "
                               f"{len(prompt_ids)} prompt + {len(labels)} label tokens > {self.MAX_LABEL_TOKENS} + 1")
                return {"bad": True}
        return {
            "input_features": input_features,
            "labels": labels,
            "prefix_len": len(tokenizer.prefix_tokens),
            "prompt_ids": prompt_ids,
            "frames": min(input_features.shape[-1], -(-len(audio) // self.HOP)),
        }


@dataclass
class WhisperDataCollator:
    """
    Collate Whisper windows: padded features, teacher-forcing inputs, and labels
    with padding and the decoder prefix (language, task, no-timestamps) masked to
    ``-100``, so the loss covers the transcript (and its timestamps) only.

    :param processor: A ``WhisperProcessor`` instance.
    :param decoder_start_token_id: Whisper SOT token id used to seed
        teacher-forcing inputs.
    :param pad_token_id: Token id used to fill shifted-out positions.
    :param prompt_mode: ``native``, or ``shared``: each window's ``prompt_ids`` precede its
        labels in the decoder input and are masked like the prefix.
    """

    processor: WhisperProcessor
    decoder_start_token_id: int
    pad_token_id: int
    prompt_mode: str = "native"

    def _placeholder(self) -> dict:
        """
        A silent item for a batch with no readable window (its loss is zeroed).

        :return: A dataset-style item.
        """
        features = self.processor.feature_extractor(
            np.zeros(SAMPLE_RATE, dtype=np.float32), sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features[0]
        tokenizer = self.processor.tokenizer
        return {
            "input_features": features,
            "labels": [self.decoder_start_token_id, tokenizer.eos_token_id],
            "prefix_len": 1,
            "frames": 1,
        }

    def __call__(self, features: list[dict]) -> dict:
        """
        Collate a list of samples into a padded batch.

        :param features: Samples from :class:`WhisperFinetuneDataset`.
        :return: ``input_features``, ``labels``, ``decoder_input_ids`` and the trainer's
            ``_feature_lengths`` / ``_placeholder``.
        """
        items = [f for f in features if not f.get("bad")]
        placeholder = not items
        if placeholder:
            items = [self._placeholder()]
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in items], return_tensors="pt"
        )
        if self.prompt_mode == "shared":
            return self._prompted(items, batch, placeholder)
        labels_batch = self.processor.tokenizer.pad([{"input_ids": f["labels"]} for f in items], return_tensors="pt")
        label_ids = labels_batch["input_ids"]
        attention_mask = labels_batch.attention_mask
        # shift_tokens_right prepends SOT again.
        if (label_ids[:, 0] == self.decoder_start_token_id).all():
            label_ids = label_ids[:, 1:]
            attention_mask = attention_mask[:, 1:]
        decoder_input_ids = shift_tokens_right(label_ids, self.pad_token_id, self.decoder_start_token_id)
        labels = label_ids.masked_fill(attention_mask.ne(1), -100)
        for b, f in enumerate(items):
            labels[b, : f["prefix_len"] - 1] = -100
        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids
        batch["_feature_lengths"] = torch.tensor([f["frames"] for f in items])
        batch["_placeholder"] = placeholder
        return batch

    def _prompted(self, items: list[dict], batch, placeholder: bool):
        """
        Teacher forcing with a previous-text prompt: the decoder reads
        ``prompt + labels[:-1]`` and predicts ``(prompt + labels)[1:]``, with the prompt,
        SOT and the prefix masked; the same alignment as the native path once the prompt is empty.

        :param items: Windows (a placeholder has no ``prompt_ids``).
        :param batch: Padded ``input_features``.
        :param placeholder: Whether the batch is the placeholder.
        :return: The batch with ``labels``, ``decoder_input_ids`` and the trainer's keys.
        """
        seqs = [list(f.get("prompt_ids") or []) + list(f["labels"]) for f in items]
        width = max(len(s) for s in seqs) - 1
        decoder_input_ids = torch.full((len(seqs), width), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(seqs), width), -100, dtype=torch.long)
        for b, (f, seq) in enumerate(zip(items, seqs)):
            seq = torch.tensor(seq, dtype=torch.long)
            decoder_input_ids[b, : len(seq) - 1] = seq[:-1]
            labels[b, : len(seq) - 1] = seq[1:]
            labels[b, : len(f.get("prompt_ids") or []) + f["prefix_len"] - 1] = -100
        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids
        batch["_feature_lengths"] = torch.tensor([f["frames"] for f in items])
        batch["_placeholder"] = placeholder
        return batch


class WhisperTrainer:
    """
    Trainer for Whisper models: full fine-tune with fp32 master weights and bf16
    autocast, the token-level cross-entropy of :class:`~lyricscribe.finetune.engine.FinetuneTrainer`
    (Whisper's own loss is a per-batch mean), and no label smoothing.

    :param config: Job configuration dictionary.
    """

    layout = FeatureLayout(key="input_features", time_dim=2, freq_dim=1)

    def __init__(self, config: dict):
        """
        Store the config; the model loads in :meth:`parts`.

        :param config: Job configuration dictionary.
        """
        self.config = config
        self.model = None
        self.processor = None

    def _load(self, source) -> None:
        """
        Load the Whisper model (fp32 master weights) and processor.

        :param source: Hub id or checkpoint dir.
        """
        logger.info(f"Loading Whisper model: {source}")
        self.model = WhisperForConditionalGeneration.from_pretrained(source, dtype=torch.float32)
        self.processor = WhisperProcessor.from_pretrained(source)
        self.model.generation_config.language = None
        self.model.generation_config.task = "transcribe"
        self.model.generation_config.forced_decoder_ids = None

    def _dataset(self, manifest: Path, seed: int, train: bool = True) -> WhisperFinetuneDataset:
        """
        A Whisper dataset over *manifest*.

        :param manifest: Manifest path.
        :param seed: Run seed.
        :param train: Training (timestamp mix and dither) rather than evaluation.
        :return: The dataset.
        """
        return WhisperFinetuneDataset(
            manifest,
            training_processor(self.processor),
            timestamp_prob=self.config.get("whisper_timestamp_prob", 0.5) if train else 0.0,
            seed=seed,
            dither=engine.DITHER if train else 0.0,
            prompt_mode=self.prompt_mode(),
        )

    def _collator(self) -> WhisperDataCollator:
        """
        The training collator.

        :return: :class:`WhisperDataCollator`.
        """
        return WhisperDataCollator(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
            pad_token_id=self.model.config.pad_token_id,
            prompt_mode=self.prompt_mode(),
        )

    def prompt_mode(self) -> str:
        """
        The training prompt, as :class:`~lyricscribe.transcribe.whisper.WhisperTranscriber` names it.

        :return: The config's ``prompt_mode`` (default ``native``: no prompt).
        """
        return configured_prompt_mode(self.config, WhisperTranscriber.prompt_modes)

    def parts(self, manifest: Path, seed: int, eval_manifest: Path | None = None) -> TrainingParts:
        """
        The pieces of a run.

        :param manifest: Training manifest (with Whisper ``segments``).
        :param seed: Run seed.
        :param eval_manifest: Held-out windows scored at each export, or ``None``.
        :return: Model, datasets, collator, loss mode and feature layout.
        """
        if self.model is None:
            self._load(self.config["base_model"])
        dataset = self._dataset(manifest, seed)
        with_segments = sum(1 for e in dataset.entries if e.get("segments"))
        if self.config.get("whisper_timestamp_prob", 0.5) > 0 and with_segments == 0:
            raise ValueError(f"{manifest} has no Whisper segments; prepare it with --whisper-segments")
        gc = bool(self.config.get("gradient_checkpointing", False))
        return TrainingParts(
            model=self.model,
            dataset=dataset,
            eval_dataset=self._dataset(eval_manifest, seed, train=False) if eval_manifest else None,
            collator=self._collator(),
            loss_mode="seq2seq",
            layout=self.layout,
            per_device=int(self.config["batch_size"]),
            args={"gradient_checkpointing": gc, "gradient_checkpointing_kwargs": {"use_reentrant": False}} if gc else {},
            export=self.export,
            notes={
                "master_weights": "fp32",
                "timestamp_prob": self.config.get("whisper_timestamp_prob", 0.5),
                "windows_with_segments": with_segments,
                "label_smoothing": 0.0,
            },
            prompt_mode=self.prompt_mode(),
        )

    def export(self, path: Path) -> Path:
        """
        Save model and processor for inference.

        :param path: Output dir.
        :return: *path*.
        """
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        # save_pretrained() can write eos_token_id as a one-element list;
        # SuppressTokensAtBeginLogitsProcessor needs an int.
        gen_config_path = path / "generation_config.json"
        if gen_config_path.exists():
            with open(gen_config_path) as f:
                gc = json.load(f)
            eos = gc.get("eos_token_id")
            if isinstance(eos, list) and len(eos) == 1:
                gc["eos_token_id"] = eos[0]
                with open(gen_config_path, "w") as f:
                    json.dump(gc, f, indent=2)
        return path

def create_trainer(config: dict):
    """
    Factory function that returns the appropriate trainer for the given
    architecture.

    :param config: Job configuration dictionary with an ``"architecture"`` key.
    :return: The architecture's trainer.
    :raises ValueError: If the architecture is not recognized.
    """
    trainers = {
        "canary": FastConformerTrainer,
        "parakeet": FastConformerTrainer,
        "whisper": WhisperTrainer,
        "qwen3_asr": Qwen3ASRTrainer,
        "qwen3_omni": Qwen3OmniLoraTrainer,
        "nemotron": NemotronLoraTrainer,
        "cohere": CohereTrainer,
        "granite": GraniteTrainer,
        "musicflamingo": MusicFlamingoTrainer,
    }
    architecture = config.get("architecture", "parakeet")
    if architecture not in trainers:
        raise ValueError(f"Unknown architecture: {architecture}")
    return trainers[architecture](config)


def run_job(config: dict, job_dir: Path, save: bool = True) -> dict:
    """
    Run (or resume) a job with its model family's trainer (see :func:`lyricscribe.finetune.engine.run`).

    :param config: Job config.
    :param job_dir: Job directory.
    :param save: Write checkpoints, exports and the training state.
    :return: Run status.
    """
    return engine.run(create_trainer(config), config, job_dir, save=save)
