import copy
import json
import logging
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from torch.jit import ScriptModule
from transformers import (
    AutoProcessor,
    GenerationConfig,
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
)
from transformers.models.qwen3_asr.modeling_qwen3_asr import _get_feat_extract_output_lengths

from lyricscribe.transcribe.adapter import TrainRecordMismatch, check_train_record, load_adapter
from lyricscribe.transcribe.base import (
    DECODE_KNOBS,
    MAX_NEW_TOKENS,
    Decode,
    Transcriber,
    describe_kwargs,
    generate_kwargs,
    shared_prompt,
)
from lyricscribe.transcribe.chunking import SAMPLE_RATE, language_name, song_chunks

logger = logging.getLogger(__name__)

# Native-format repo holding the config/processor for a legacy checkpoint,
# keyed by its text-model hidden size.
_NATIVE_REFERENCE = {
    2048: "Qwen/Qwen3-ASR-1.7B-hf",
    1024: "Qwen/Qwen3-ASR-0.6B-hf",
}

# qwen-asr limits (qwen_asr/inference/utils.py).
_MAX_ASR_INPUT_SECONDS = 1200
_MIN_ASR_INPUT_SECONDS = 0.5
_ASR_TEXT_TAG = "<asr_text>"
_LANG_PREFIX = "language "


def _resolve_dir(model_name: str, allow_patterns: list[str] | None = None) -> Path:
    """
    Local directory for a model path or (cached) hub id.

    :param model_name: Local path or hub repo id.
    :param allow_patterns: Hub file patterns to fetch.
    :return: Directory path.
    """
    path = Path(model_name)
    if path.is_dir():
        return path

    return Path(snapshot_download(model_name, allow_patterns=allow_patterns))


def _convert_legacy_key(key: str) -> str:
    """
    Map a qwen-asr parameter name onto the native transformers layout.

    :param key: Legacy ``thinker.*`` parameter name.
    :return: Native parameter name.
    """
    if key.startswith("thinker.audio_tower.proj1."):
        return "model.multi_modal_projector.linear_1." + key.rsplit(".", 1)[1]
    if key.startswith("thinker.audio_tower.proj2."):
        return "model.multi_modal_projector.linear_2." + key.rsplit(".", 1)[1]
    if key.startswith("thinker.audio_tower."):
        return "model.audio_tower." + key[len("thinker.audio_tower."):]
    if key.startswith("thinker.model."):
        return "model.language_model." + key[len("thinker.model."):]
    if key == "thinker.lm_head.weight":
        return "lm_head.weight"
    raise KeyError(f"Unexpected qwen-asr parameter {key!r}")


def load_qwen3_asr(model_name: str, dtype: torch.dtype, device: str):
    """
    Load a Qwen3-ASR model + processor, converting qwen-asr checkpoints.

    :param model_name: Hub id or local checkpoint dir (either layout).
    :param dtype: Model dtype.
    :param device: Torch device.
    :return: ``(model, processor, generation_config)``.
    """
    src = _resolve_dir(model_name)
    cfg = json.loads((src / "config.json").read_text())
    if "thinker_config" not in cfg:
        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            src, dtype=dtype, device_map=device, attn_implementation="sdpa"
        )
        return model, AutoProcessor.from_pretrained(src), model.generation_config

    legacy = cfg["thinker_config"]
    hidden = legacy["text_config"]["hidden_size"]
    if hidden not in _NATIVE_REFERENCE:
        raise ValueError(f"No native Qwen3-ASR reference for hidden_size={hidden}")
    ref = _resolve_dir(
        _NATIVE_REFERENCE[hidden], allow_patterns=["*.json", "*.jinja", "*.txt"]
    )
    config = Qwen3ASRConfig.from_pretrained(ref)
    pairs = [
        (config.audio_config.d_model, legacy["audio_config"]["d_model"]),
        (config.audio_config.encoder_layers, legacy["audio_config"]["encoder_layers"]),
        (config.audio_config.output_dim, legacy["audio_config"]["output_dim"]),
        (config.text_config.num_hidden_layers, legacy["text_config"]["num_hidden_layers"]),
        (config.text_config.vocab_size, legacy["text_config"]["vocab_size"]),
        (config.audio_token_id, legacy["audio_token_id"]),
    ]
    if any(a != b for a, b in pairs):
        raise ValueError(f"{ref} does not match {src}: {pairs}")
    config._attn_implementation = "sdpa"

    state: dict[str, torch.Tensor] = {}
    for shard in sorted(src.glob("*.safetensors")):
        for key, tensor in load_file(str(shard)).items():
            state[_convert_legacy_key(key)] = tensor
    embed = state["model.language_model.embed_tokens.weight"]
    head = state.pop("lm_head.weight", None)
    # The native class ties lm_head to the embeddings, so an untied head would be lost.
    if head is not None and not torch.equal(head, embed):
        raise ValueError(f"{src}: lm_head differs from embed_tokens; cannot tie")

    model = Qwen3ASRForConditionalGeneration._from_config(config, dtype=dtype)
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise ValueError(f"{src}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    model.tie_weights()
    model = model.to(device=device, dtype=dtype).eval()
    processor = AutoProcessor.from_pretrained(ref)
    return model, processor, GenerationConfig.from_pretrained(ref)


def _encode_audio_per_item(model) -> None:
    """
    Make the native model encode each audio of a batch separately.

    qwen-asr runs its audio encoder one item at a time on exactly that item's
    frames ("audio encoder do not support batch inference to keep
    precision"), so each item's last 100-frame chunk is zero-padded. The
    native encoder takes the whole batch, and an item's tail chunk then
    convolves over the feature extractor's padding instead (log-mel floor
    values, not zeros). Encoding item by item restores the reference
    behaviour.

    :param model: Native ``Qwen3ASRForConditionalGeneration``.
    """
    inner = model.model
    original = inner.get_audio_features
    n_window = inner.audio_tower.n_window
    chunk = n_window * 2

    def per_item(input_features, input_features_mask, **kwargs):
        """
        Encode each item of the batch alone and concatenate the pooled outputs.

        :param input_features: Padded log-mel features, one row per item.
        :param input_features_mask: Valid-frame mask per item.
        :param kwargs: Passed through to the original ``get_audio_features``.
        :return: The last item's output with ``pooler_output`` holding every item's pooled tokens.
        """
        pooled = []
        out = None
        for feat, mask in zip(input_features, input_features_mask):
            n = int(mask.sum())
            pad = (-n) % chunk
            feats = torch.nn.functional.pad(feat[:, :n], (0, pad)).unsqueeze(0)
            item_mask = torch.zeros(1, n + pad, dtype=mask.dtype, device=mask.device)
            item_mask[0, :n] = 1
            # Whole-item window: qwen-asr's sdpa path ignores n_window_infer.
            tokens = _get_feat_extract_output_lengths(
                torch.tensor([n], device=mask.device), n_window
            )
            cu_seqlens = torch.cat(
                [torch.zeros(1, dtype=torch.int32, device=mask.device), tokens.to(torch.int32)]
            )
            out = original(feats, item_mask, cu_seqlens=cu_seqlens, **kwargs)
            pooled.append(out.pooler_output)
        out.pooler_output = torch.cat(pooled, dim=0)
        out.last_hidden_state = None
        return out

    inner.get_audio_features = per_item


# _split_audio_into_chunks and _parse_asr_output are adapted from
# qwen-asr 0.0.6 (qwen_asr/inference/utils.py), Copyright 2026 The Alibaba Qwen team,
# Apache-2.0.
def _split_audio_into_chunks(
    wav: np.ndarray,
    sr: int,
    max_chunk_sec: float,
    search_expand_sec: float = 5.0,
    min_window_ms: float = 100.0,
) -> list[np.ndarray]:
    """
    Split at low-energy points near ``max_chunk_sec``; pad short chunks (qwen-asr).

    :param wav: Mono waveform.
    :param sr: Sample rate.
    :param max_chunk_sec: Longest chunk.
    :param search_expand_sec: How far either side of each cut point to search
        for the quietest window.
    :param min_window_ms: Width of the energy window.
    :return: Chunks, each padded to at least ``_MIN_ASR_INPUT_SECONDS``.
    """
    wav = np.asarray(wav, dtype=np.float32)
    total_len = int(wav.shape[0])
    chunks = []
    if total_len / float(sr) <= max_chunk_sec:
        chunks = [wav]
    else:
        max_len = int(max_chunk_sec * sr)
        expand = int(search_expand_sec * sr)
        win = max(4, int((min_window_ms / 1000.0) * sr))
        start = 0
        while (total_len - start) > max_len:
            cut = start + max_len
            left = max(start, cut - expand)
            right = min(total_len, cut + expand)
            if right - left <= win:
                boundary = cut
            else:
                seg_abs = np.abs(wav[left:right])
                sums = np.convolve(seg_abs, np.ones(win, dtype=np.float32), mode="valid")
                wstart = int(np.argmin(sums))
                inner = int(np.argmin(seg_abs[wstart:wstart + win]))
                boundary = left + wstart + inner
            boundary = int(min(max(boundary, start + 1), total_len))
            chunks.append(wav[start:boundary])
            start = boundary
        chunks.append(wav[start:total_len])
    min_len = int(_MIN_ASR_INPUT_SECONDS * sr)
    return [
        np.pad(c, (0, min_len - c.shape[0])).astype(np.float32)
        if c.shape[0] < min_len
        else c
        for c in chunks
    ]


def _parse_asr_output(raw: str, forced_language: str | None) -> str:
    """
    Strip the ``language X<asr_text>`` header (qwen-asr). qwen-asr's
    repetition fixer is not applied: no backend gets vendor post-processing.

    :param raw: Decoded model output.
    :param forced_language: Language forced in the prompt; the output then has
        no header to strip.
    :return: Transcription text.
    """
    s = (raw or "").strip()
    if not s:
        return ""
    if forced_language:
        return s
    if _ASR_TEXT_TAG not in s:
        return s.strip()
    meta, text = s.split(_ASR_TEXT_TAG, 1)
    if "language none" in meta.lower():
        return text.strip()
    return text.strip()


# How training records this backend's native prompt (forced language, empty context).
PROMPT_MODE = "native"


def _build_prompt(context: str, language: str | None) -> str:
    """
    The qwen-asr text prompt (its chat template, rendered).

    Forcing a language appends ``language X<asr_text>`` so the model emits
    transcription text only.

    :param context: ``context`` system message (may be empty).
    :param language: Language name to force, or ``None``.
    :return: The rendered prompt.
    """
    prompt = (
        f"<|im_start|>system\n{context or ''}<|im_end|>\n"
        "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if language:
        prompt += f"language {language}{_ASR_TEXT_TAG}"
    return prompt


class Qwen3ASRTranscriber(Transcriber):
    """
    Qwen3-ASR (0.6B / 1.7B and finetunes), RMS-VAD chunked or full-song.

    ``batch_size`` is qwen-asr's ``max_inference_batch_size``: consecutive
    chunks of one song decoded together.
    The per-song prompt (``initial_prompt``) is passed as Qwen's ``context``
    system message; ``prompt_mode="shared"`` passes the shared prompt there instead.
    """

    supports = frozenset(DECODE_KNOBS)
    prompt_modes = (PROMPT_MODE, "shared")

    def __init__(
        self,
        model_name: str,
        batch_size: int = 8,
        max_new_tokens: int = MAX_NEW_TOKENS,
        auto_language: bool = False,
        adapter: str | None = None,
        prompt_mode: str | None = None,
    ) -> None:
        """
        :param model_name: Hub id or checkpoint dir.
        :param batch_size: Chunks decoded per ``generate`` call.
        :param max_new_tokens: Generation cap per chunk.
        :param auto_language: Ignore the provided language and let the model
            identify it (qwen-asr's default behaviour).
        :param adapter: PEFT adapter dir (the NVIDIA-recipe control) merged into the model, or ``None``.
        :param prompt_mode: ``native`` (empty context, default) or ``shared``.
        """
        super().__init__(model_name, batch_size, prompt_mode)
        self.max_new_tokens = max_new_tokens
        self.auto_language = auto_language
        self.adapter = adapter
        self.train_record = None
        self.model = None
        self.processor = None
        self.generation_config = None

    def load(self) -> None:
        """
        Load the model and processor onto the GPU (CPU without CUDA), merging the adapter if any.

        :raises RuntimeError: If the checkpoint's training record has another prompt.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        logger.info(f"Loading Qwen3-ASR: {self.model_name} on {device}")
        # cuDNN SDPA rebuilds its graph per sequence shape, which stalls decoding.
        torch.backends.cuda.enable_cudnn_sdp(False)
        self.train_record = check_train_record(
            self.adapter or self.model_name, self.prompt_mode, self.model_name if self.adapter else None
        )
        self.model, self.processor, self.generation_config = load_qwen3_asr(
            self.model_name, dtype, device
        )
        if self.adapter:
            self.model = load_adapter(self.model, self.adapter).merge_and_unload()
            logger.info(f"Merged adapter {self.adapter}")
        # After the merge: the patch wraps the merged model's audio path.
        _encode_audio_per_item(self.model)

    @torch.inference_mode()
    def _decode(self, wavs: list[np.ndarray], context: str, language: str | None, decode: Decode) -> list[str]:
        """
        Decode chunks of one song in batches of ``batch_size``.

        :param wavs: 16 kHz mono chunks.
        :param context: Qwen ``context`` system message.
        :param language: Language name to force, or ``None`` to let the model identify it.
        :param decode: Decoding settings.
        :return: Parsed transcription text per chunk.
        :raises TrainRecordMismatch: For a context on a checkpoint trained with the native (empty-context) prompt.
        """
        if context and getattr(self, "train_record", None) and self.prompt_mode == PROMPT_MODE:
            raise TrainRecordMismatch(f"{self.adapter or self.model_name} was trained with an empty context; got {context[:60]!r}")
        prompt = _build_prompt(context, language)
        config = copy.deepcopy(self.generation_config)
        outs = []
        for i in range(0, len(wavs), max(1, self.batch_size)):
            sub = wavs[i:i + max(1, self.batch_size)]
            inputs = self.processor(
                text=[prompt] * len(sub), audio=sub, return_tensors="pt", padding=True
            )
            inputs = inputs.to(self.model.device).to(self.model.dtype)
            # Fresh per call: the repetition controls record where this call's prompt ends.
            kwargs = generate_kwargs(self.max_new_tokens, decode)
            self.last_generate_kwargs = {**describe_kwargs(kwargs), "prompt": prompt}
            ids = self.model.generate(**inputs, generation_config=config, **kwargs)
            outs.extend(
                self.processor.batch_decode(
                    ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        return [_parse_asr_output(o, language) for o in outs]

    def transcribe(
        self,
        audio_path: str,
        use_vad: bool = False,
        vad_model: ScriptModule | None = None,
        use_chunked: bool = False,
        language: str | None = None,
        vad_source: str | None = None,
        vad_method: str = "silero",
        decode: Decode = Decode(),
        condition_on_prev: bool = True,
        temperature_fallback: bool = False,
        compression_ratio_threshold: float | None = None,
        logprob_threshold: float | None = None,
        initial_prompt: str | None = None,
    ) -> str:
        """
        Transcribe one song. ``initial_prompt`` (or the shared prompt) becomes Qwen's ``context``.
        """
        del use_chunked, condition_on_prev, temperature_fallback
        del compression_ratio_threshold, logprob_threshold
        self.check_decode(decode)
        if self.prompt_mode == "shared":
            if initial_prompt:
                raise ValueError("A per-song initial_prompt and prompt_mode 'shared' can't be combined")
            initial_prompt = shared_prompt(language)
        forced = None if self.auto_language else language_name(language)
        segments = song_chunks(audio_path, use_vad, vad_method, vad_source, vad_model=vad_model)
        # Each VAD segment is one qwen-asr input; inputs over 20 min are split
        # further and re-joined without a separator, and short ones padded to 0.5 s.
        pieces: list[tuple[int, np.ndarray]] = []
        for idx, seg in enumerate(segments):
            for part in _split_audio_into_chunks(seg, SAMPLE_RATE, _MAX_ASR_INPUT_SECONDS):
                pieces.append((idx, part))
        if not pieces:
            return ""
        texts = self._decode([p for _, p in pieces], initial_prompt or "", forced, decode)
        per_segment = [""] * len(segments)
        for (idx, _), text in zip(pieces, texts):
            per_segment[idx] += text
        return " ".join(t.strip() for t in per_segment if t.strip())
