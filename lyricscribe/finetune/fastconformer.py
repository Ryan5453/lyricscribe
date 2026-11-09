import logging
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoProcessor, CanaryForConditionalGeneration, ParakeetForTDT
from transformers.loss.loss_rnnt import rnnt_loss

from lyricscribe.finetune import engine
from lyricscribe.finetune.engine import (
    SAMPLE_RATE,
    FeatureLayout,
    FinetuneTrainer,
    ManifestDataset,
    TrainingParts,
    configured_prompt_mode,
    drop_bad,
    training_processor,
)
from lyricscribe.finetune.transducer_loss import tdt_loss
from lyricscribe.transcribe.base import shared_prompt
from lyricscribe.transcribe.canary import CanaryTranscriber, add_nemo_prefix

logger = logging.getLogger(__name__)

# NeMo's released Parakeet TDT recipe (``loss.tdt_kwargs``).
TDT_SIGMA = 0.02
TDT_OMEGA = 0.1

# Released NeMo recipe dropouts that the transformers modules do not implement.
PREDNET_DROPOUT = 0.2
JOINT_DROPOUT = 0.2
CANARY_DECODER_DROPOUT = 0.1

# Both models' features are ``(batch, frames, mel bins)``.
LAYOUT = FeatureLayout(key="input_features", time_dim=1, freq_dim=2)


def _output_dropout(p: float):
    """
    Forward hook: dropout on a module's output (a tuple's first element) in training mode.

    :param p: Dropout probability.
    :return: The hook.
    """
    def hook(module, args, output):
        if isinstance(output, tuple):
            return (torch.nn.functional.dropout(output[0], p, module.training), *output[1:])
        return torch.nn.functional.dropout(output, p, module.training)

    return hook


def _input_dropout(p: float):
    """
    Forward pre-hook: dropout on a module's first input in training mode.

    :param p: Dropout probability.
    :return: The hook.
    """
    def hook(module, args):
        return (torch.nn.functional.dropout(args[0], p, module.training), *args[1:])

    return hook


def add_nemo_regularization(model, is_canary: bool) -> None:
    """
    Give a transformers Parakeet TDT or Canary model the training
    regularization of the NeMo recipes, which the transformers modules leave
    out: no layer drop (``stochastic_depth_drop_prob: 0``), dropout on every
    conformer sub-module's output before its residual add, and either 0.2
    dropout in the prediction-network LSTM and the joint (Parakeet) or 0.1
    dropout on the decoder embeddings and every attention and feed-forward
    output (Canary). The hooks act only in training mode and add no
    parameters, so inference and the checkpoint format are unchanged.

    :param model: ``ParakeetForTDT`` or ``CanaryForConditionalGeneration``.
    :param is_canary: Whether *model* is Canary.
    """
    encoder = model.model.encoder if is_canary else model.encoder
    encoder.layerdrop = 0.0
    encoder.config.layerdrop = 0.0
    for block in encoder.layers:
        for module in (block.feed_forward1, block.self_attn, block.conv, block.feed_forward2):
            module.register_forward_hook(_output_dropout(encoder.config.dropout))
    if is_canary:
        decoder = model.model.decoder
        decoder.embedding_layernorm.register_forward_hook(_output_dropout(CANARY_DECODER_DROPOUT))
        for layer in decoder.layers:
            for module in (layer.self_attn, layer.encoder_attn, layer.mlp):
                module.register_forward_hook(_output_dropout(CANARY_DECODER_DROPOUT))
    else:
        model.decoder.lstm.dropout = PREDNET_DROPOUT
        model.decoder.decoder_projector.register_forward_pre_hook(_input_dropout(PREDNET_DROPOUT))
        model.joint.head.register_forward_pre_hook(_input_dropout(JOINT_DROPOUT))


@dataclass
class ParakeetCollator:
    """
    Batch Parakeet TDT training windows: features, ``<blank>``-prefixed
    prediction-network inputs, and ``-100``-padded labels. Unreadable windows
    are dropped (see :func:`~lyricscribe.finetune.engine.drop_bad`).

    :param processor: ``ParakeetProcessor``.
    """

    processor: object

    def __call__(self, items: list[dict]) -> dict:
        """
        Collate a list of windows.

        :param items: Samples from :class:`~lyricscribe.finetune.engine.ManifestDataset`.
        :return: ``input_features``, ``attention_mask``, ``decoder_input_ids``, ``labels``
            and the trainer's ``_feature_lengths`` / ``_placeholder``.
        """
        items = drop_bad(items)
        feats = self.processor.feature_extractor(
            [it["audio"] for it in items], sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        tokenizer = self.processor.tokenizer
        ids = [tokenizer(it["text"], add_special_tokens=False).input_ids for it in items]
        width = max(1, max(len(i) for i in ids))
        labels = torch.full((len(ids), width), -100, dtype=torch.long)
        decoder_input_ids = torch.full(
            (len(ids), width + 1), self.processor.blank_token_id, dtype=torch.long
        )
        for b, row in enumerate(ids):
            if row:
                labels[b, :len(row)] = torch.tensor(row)
                decoder_input_ids[b, 1:len(row) + 1] = torch.tensor(row)
        return {
            "input_features": feats["input_features"],
            "attention_mask": feats["attention_mask"],
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "_feature_lengths": feats["attention_mask"].sum(-1),
            "_placeholder": any(it.get("placeholder") for it in items),
        }


@dataclass
class CanaryCollator:
    """
    Batch Canary training windows in NeMo's ``canary2`` format (dummy-prefix
    ``▁``, language tokens, punctuation on; for ``shared`` the shared prompt in the
    context slot after ``<|startofcontext|>``, as inference's ``insert_context``).
    Labels cover the transcript and end token only: the decoder prompt is masked,
    as for every model.

    :param processor: ``CanaryProcessor``.
    :param prompt_mode: ``native`` or ``shared``.
    """

    processor: object
    prompt_mode: str = "native"

    def __post_init__(self) -> None:
        """
        Start the per-language prompt-length cache.
        """
        self._prompt_ids: dict[str, list[int]] = {}

    def _conversation(self, language: str, text: str | None) -> list[dict]:
        """
        The chat-template conversation of one window.

        :param language: Language code.
        :param text: Transcript, or ``None`` for the prompt alone.
        :return: Conversation messages.
        """
        user = {
            "role": "user",
            "content": [{
                "type": "text", "source_language": language, "target_language": language, "punctuation": True,
            }],
        }
        return [user] if text is None else [user, {"role": "assistant", "content": text}]

    def context_ids(self, language: str) -> list[int]:
        """
        Token ids of the shared prompt for *language*, tokenized as inference does.

        :param language: Language code.
        :return: Ids for the context slot.
        """
        return self.processor.tokenizer(shared_prompt(language), add_special_tokens=False).input_ids

    def prompt_ids(self, language: str) -> list[int]:
        """
        Token ids of the decoder prompt for *language*.

        :param language: Language code.
        :return: Prompt ids, as the processor tokenizes the rendered prompt (with the
            context after ``<|startofcontext|>`` for ``shared``).
        """
        if language not in self._prompt_ids:
            text = self.processor.apply_chat_template(
                self._conversation(language, None), tokenize=False, add_generation_prompt=True
            )
            ids = self.processor.tokenizer(text, add_special_tokens=False).input_ids
            if self.prompt_mode == "shared":
                ids = ids[:1] + self.context_ids(language) + ids[1:]
            self._prompt_ids[language] = ids
        return self._prompt_ids[language]

    def _insert_context(self, items: list[dict], decoder_input_ids: torch.Tensor, labels: torch.Tensor):
        """
        Insert each window's shared-prompt ids after ``<|startofcontext|>`` (position 0 of both
        tensors, before the ``▁`` prefix) and re-pad: windows of different languages get
        contexts of different lengths.

        :param items: The batch's windows.
        :param decoder_input_ids: ``(B, W)`` decoder inputs.
        :param labels: ``(B, W + 1)`` labels.
        :return: The widened ``(decoder_input_ids, labels)``.
        :raises RuntimeError: If a row doesn't start with ``<|startofcontext|>``.
        """
        soc = self.processor.tokenizer.convert_tokens_to_ids("<|startofcontext|>")
        pad = self.processor.tokenizer.pad_token_id
        rows_in, rows_lab = [], []
        for b, it in enumerate(items):
            if int(decoder_input_ids[b, 0]) != soc:
                raise RuntimeError("Canary decoder inputs must start with <|startofcontext|>")
            ctx = torch.tensor(self.context_ids(it["language"]), dtype=decoder_input_ids.dtype)
            rows_in.append(torch.cat([decoder_input_ids[b, :1], ctx, decoder_input_ids[b, 1:]]))
            rows_lab.append(torch.cat([labels[b, :1], ctx.to(labels.dtype), labels[b, 1:]]))
        width = max(len(r) for r in rows_in)
        decoder_input_ids = torch.stack([torch.nn.functional.pad(r, (0, width - len(r)), value=pad) for r in rows_in])
        labels = torch.stack([torch.nn.functional.pad(r, (0, width + 1 - len(r)), value=-100) for r in rows_lab])
        return decoder_input_ids, labels

    def __call__(self, items: list[dict]) -> dict:
        """
        Collate a list of windows.

        :param items: Samples from :class:`~lyricscribe.finetune.engine.ManifestDataset`.
        :return: ``input_features``, ``attention_mask``, ``decoder_input_ids``, ``labels``
            and the trainer's ``_feature_lengths`` / ``_placeholder``.
        :raises RuntimeError: If a rendered prompt is not a prefix of its window's sequence.
        """
        items = drop_bad(items)
        texts = [
            self.processor.apply_chat_template(self._conversation(it["language"], it["text"]), tokenize=False)
            for it in items
        ]
        batch = self.processor(
            audio=[it["audio"] for it in items],
            text=texts,
            output_labels=True,
            sampling_rate=SAMPLE_RATE,
            text_kwargs={"padding": True, "padding_side": "right"},
        )
        decoder_input_ids = batch["decoder_input_ids"]
        # labels[j] is the token after decoder position j once the ``▁`` prefix is prepended.
        labels = torch.cat([decoder_input_ids[:, :1], batch["labels"]], dim=1)
        if self.prompt_mode == "shared":
            if "decoder_attention_mask" in batch:
                raise RuntimeError("Canary's training batch has a decoder_attention_mask the context insertion doesn't widen")
            decoder_input_ids, labels = self._insert_context(items, decoder_input_ids, labels)
        for b, it in enumerate(items):
            prompt = self.prompt_ids(it["language"])
            if labels[b, :len(prompt)].tolist() != prompt:
                raise RuntimeError(f"Canary prompt for {it['language']!r} is not a prefix of its sequence")
            labels[b, :len(prompt)] = -100
        batch["labels"] = labels
        batch["decoder_input_ids"] = add_nemo_prefix(decoder_input_ids, self.processor.tokenizer)
        batch = dict(batch)
        batch["_feature_lengths"] = batch["attention_mask"].sum(-1)
        batch["_placeholder"] = any(it.get("placeholder") for it in items)
        return batch


class _CanaryTrainer(FinetuneTrainer):
    """
    Canary: the model's own cross-entropy, normalised over the global batch.
    """


class _TDTTrainer(FinetuneTrainer):
    """
    Parakeet with NeMo's TDT objective, computed outside the
    (DDP-wrapped) forward: per training batch, with probability ``TDT_OMEGA``
    the standard RNN-T loss on the token logits, otherwise the TDT loss with
    ``TDT_SIGMA`` logit undernormalization. Per-sample NLLs are summed and
    divided by the label tokens of the whole global batch, the multi-batch
    form of the checkpoints' ``rnnt_reduction: mean_volume``.
    """

    def _token_loss(self, model, inputs, num_items_in_batch):
        """
        Forward pass, then the sampled TDT / RNN-T loss.

        :param model: Possibly DDP-wrapped ``ParakeetForTDT``.
        :param inputs: Collated batch.
        :param num_items_in_batch: Label tokens across the global batch.
        :return: ``(loss, outputs)``.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        inner = model.module if hasattr(model, "module") else model
        config = inner.config
        token_logits = outputs.logits[..., : config.vocab_size]
        logit_lengths = inner._get_subsampling_output_length(inputs["attention_mask"].sum(-1)).int()
        target_lengths = (labels != -100).sum(-1).int()
        targets = labels.masked_fill(labels == -100, config.pad_token_id)
        lattice = logit_lengths.long() * (target_lengths.long() + 1) * config.vocab_size
        if model.training and random.random() < TDT_OMEGA and int(lattice.max()) < 2**31:
            # Per sample: torchaudio's CUDA RNN-T kernel indexes with int32, so a batch with a
            # >2^31-element lattice (rare, very dense labels) keeps the TDT loss.
            losses = []
            for b in range(targets.shape[0]):
                t, u = int(logit_lengths[b]), int(target_lengths[b])
                if u == 0:
                    # torchaudio's CUDA kernel reads uninitialized memory for an empty target; its only path is t blanks.
                    blank = torch.log_softmax(token_logits[b, :t, 0].float(), dim=-1)[:, config.blank_token_id]
                    losses.append(-blank.sum())
                    continue
                losses.append(rnnt_loss(
                    logits=token_logits[b:b + 1, :t, : u + 1],
                    targets=targets[b:b + 1, :u].int().contiguous(),
                    logit_lengths=logit_lengths[b:b + 1],
                    target_lengths=target_lengths[b:b + 1],
                    blank_token_id=config.blank_token_id,
                    reduction="sum",
                ))
            losses = torch.stack(losses)
        else:
            losses = tdt_loss(
                token_logits=token_logits,
                duration_logits=outputs.logits[..., config.vocab_size:],
                targets=targets.int(),
                logit_lengths=logit_lengths,
                target_lengths=target_lengths,
                blank_token_id=config.blank_token_id,
                durations=config.durations,
                sigma=TDT_SIGMA,
            )
        return self._normalise(losses.sum(), labels, num_items_in_batch), outputs


class FastConformerTrainer:
    """
    Finetune NVIDIA Parakeet TDT or Canary (run by :func:`lyricscribe.finetune.engine.run`).

    :param config: Job configuration dictionary.
    """

    def __init__(self, config: dict) -> None:
        """
        Store the config; the model loads in :meth:`parts`.

        :param config: Job configuration dictionary.
        """
        self.config = config
        self.is_canary = config["architecture"] == "canary"
        self.model = None
        self.processor = None

    def _load(self, source: str | Path) -> None:
        """
        Load model (fp32 master weights) and processor from a hub id or checkpoint dir.

        :param source: Hub id or ``epoch_N`` directory.
        """
        model_cls = CanaryForConditionalGeneration if self.is_canary else ParakeetForTDT
        self.model = model_cls.from_pretrained(source, dtype=torch.float32)
        self.processor = AutoProcessor.from_pretrained(source)
        add_nemo_regularization(self.model, self.is_canary)
        if self.config.get("freeze_encoder"):
            encoder = self.model.model.encoder if self.is_canary else self.model.encoder
            encoder.requires_grad_(False)
        if not self.is_canary:
            # NeMo's blank_as_pad: the blank (start) embedding stays zero and untrained.
            self.model.decoder.embedding.padding_idx = self.model.config.blank_token_id

    def _collator(self):
        """
        The architecture's collator.

        :return: A collator instance.
        """
        processor = training_processor(self.processor)
        return CanaryCollator(processor, self.prompt_mode()) if self.is_canary else ParakeetCollator(processor)

    def prompt_mode(self) -> str:
        """
        The training prompt, as inference names it: Canary's from the config, Parakeet's always ``native``.

        :return: Prompt mode.
        """
        modes = CanaryTranscriber.prompt_modes if self.is_canary else ("native",)
        return configured_prompt_mode(self.config, modes)

    def _trainer_cls(self) -> type:
        """
        The architecture's trainer class.

        :return: Trainer class.
        """
        return _CanaryTrainer if self.is_canary else _TDTTrainer

    def _loss_mode(self) -> str:
        """
        The architecture's loss mode.

        :return: ``"model"`` (Canary) or ``"tdt"`` (Parakeet).
        """
        return "model" if self.is_canary else "tdt"

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
            collator=self._collator(),
            trainer_cls=self._trainer_cls(),
            loss_mode=self._loss_mode(),
            layout=LAYOUT,
            per_device=int(self.config["batch_size"]),
            export=self.export,
            notes={
                "native_regularization": "NeMo residual/prednet/joint/decoder dropouts, no layerdrop",
                "master_weights": "fp32",
            },
            prompt_mode=self.prompt_mode(),
        )

    def export(self, path: Path) -> Path:
        """
        Save model and processor in transformers format.

        :param path: Output dir.
        :return: *path*.
        """
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)
        return path
