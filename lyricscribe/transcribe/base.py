import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.jit import ScriptModule
from transformers import LogitsProcessor, LogitsProcessorList, NoRepeatNGramLogitsProcessor

from lyricscribe.transcribe.chunking import language_name, load_mono, pad_short, song_chunks

logger = logging.getLogger(__name__)

# Per-chunk generation cap shared by every autoregressive backend (Whisper's decoder limit).
MAX_NEW_TOKENS = 448

DECODE_KNOBS = ("num_beams", "repetition_penalty", "no_repeat_ngram_size")

# One prompt, word for word the same for every backend with a text input.
SHARED_PROMPT = "Transcribe the sung lyrics in this audio exactly. The language is {language}. Output only the lyrics."


def shared_prompt(language: str | None) -> str:
    """
    The shared prompt with the song's language filled in.

    :param language: ``detected_language`` from ``lyrics.json``, or ``None`` (English).
    :return: Prompt text.
    """
    return SHARED_PROMPT.format(language=language_name(language) or "English")


@dataclass(frozen=True)
class Decode:
    """
    Decoding settings applied identically by every backend that supports them.

    :param num_beams: Beam width; ``1`` is greedy.
    :param repetition_penalty: Penalty on tokens already generated; ``1.0`` is off.
    :param no_repeat_ngram_size: Ban repeating a generated n-gram; ``0`` is off.
    """

    num_beams: int = 1
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0

    def active(self) -> set[str]:
        """
        Knobs set to something other than their off value.

        :return: Names from :data:`DECODE_KNOBS`.
        """
        off = Decode()
        return {k for k in DECODE_KNOBS if getattr(self, k) != getattr(off, k)}

    def as_dict(self) -> dict:
        """
        Plain-dict form for job records.

        :return: The three settings.
        """
        return asdict(self)


def penalize_generated(scores: torch.FloatTensor, generated: torch.LongTensor, penalty: float) -> torch.FloatTensor:
    """
    Multiply the log-probability of every generated token by *penalty* (see
    :class:`GeneratedRepetitionPenalty`), on logits or log-probabilities alike.

    :param scores: ``(rows, vocab)`` logits or log-probabilities.
    :param generated: ``(rows, length)`` token ids generated so far.
    :param penalty: Multiplicative penalty.
    :return: The penalised scores.
    """
    if generated.shape[1] == 0:
        return scores
    score = torch.gather(scores, 1, generated)
    logp = torch.gather(torch.log_softmax(scores, dim=-1), 1, generated)
    # Tokens already masked to -inf stay masked.
    score = torch.where(torch.isfinite(logp), score + (penalty - 1) * logp, score)
    return scores.scatter(1, generated, score)


class GeneratedRepetitionPenalty(LogitsProcessor):
    """
    Repetition penalty on log-probabilities, restricted to generated tokens:
    each already-generated token's log-probability is multiplied by
    ``penalty``, i.e. its probability is raised to that power.

    HF's version divides positive and multiplies negative scores, and
    ``generate()`` hands processors raw logits under greedy but
    log-probabilities under beam search, so one setting is two different
    penalties, and under greedy one that depends on each model's logit
    offset. Working on log-probabilities gives one operator for every search
    and every model. The prompt (chat template, instruction, decoder task
    tokens) is never penalised, so the knob also means the same for
    decoder-only and encoder-decoder models.

    The prompt length is taken from the first call, when ``input_ids`` is
    exactly the (padded) prompt.
    """

    def __init__(self, penalty: float) -> None:
        """
        :param penalty: Multiplicative penalty, > 0.
        """
        if penalty <= 0:
            raise ValueError(f"repetition_penalty must be > 0, got {penalty}")
        self.penalty = penalty
        self.start = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Penalise every token id generated so far.

        :param input_ids: ``(batch * beams, length)`` ids so far.
        :param scores: ``(batch * beams, vocab)`` logits (greedy) or
            log-probabilities (beam search).
        :return: The scores with each generated token's log-probability
            multiplied by ``penalty``.
        """
        if self.start is None:
            self.start = input_ids.shape[1]
        return penalize_generated(scores, input_ids[:, self.start:], self.penalty)


class GeneratedNoRepeatNGram(NoRepeatNGramLogitsProcessor):
    """
    HF's ``no_repeat_ngram_size`` restricted to generated tokens (see
    :class:`GeneratedRepetitionPenalty`).
    """

    def __init__(self, ngram_size: int) -> None:
        """
        :param ngram_size: N-gram length, > 0.
        """
        super().__init__(ngram_size)
        self.start = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Ban continuations that would repeat a generated n-gram.

        :param input_ids: ``(batch * beams, length)`` ids so far.
        :param scores: ``(batch * beams, vocab)`` logits.
        :return: The masked logits.
        """
        if self.start is None:
            self.start = input_ids.shape[1]
        return super().__call__(input_ids[:, self.start:], scores)


def generate_kwargs(max_new_tokens: int, decode: Decode) -> dict:
    """
    ``generate()`` kwargs for one call under the shared decoding settings.

    The HF built-in repetition controls are passed explicitly off, so a
    checkpoint's ``generation_config.json`` can't add its own on top.

    :param max_new_tokens: Generation cap for this call.
    :param decode: Decoding settings.
    :return: Keyword arguments for ``model.generate``.
    """
    processors = []
    if decode.repetition_penalty != 1.0:
        processors.append(GeneratedRepetitionPenalty(decode.repetition_penalty))
    if decode.no_repeat_ngram_size:
        processors.append(GeneratedNoRepeatNGram(decode.no_repeat_ngram_size))
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "num_beams": decode.num_beams,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
        "logits_processor": LogitsProcessorList(processors),
    }
    if decode.num_beams > 1:
        kwargs.update(length_penalty=1.0, early_stopping=False, num_return_sequences=1)
    return kwargs


def describe_kwargs(kwargs: dict) -> dict:
    """
    JSON-safe copy of :func:`generate_kwargs` output for job records.

    :param kwargs: ``generate()`` kwargs.
    :return: The kwargs with logits processors replaced by their settings.
    """
    out = {k: v for k, v in kwargs.items() if k != "logits_processor"}
    out["logits_processor"] = [
        {"type": type(p).__name__, **{k: v for k, v in vars(p).items() if k != "start"}}
        for p in kwargs.get("logits_processor", [])
    ]
    return out


class Transcriber(ABC):
    """
    Abstract base class for ASR transcription models.

    ``supports`` lists the :data:`DECODE_KNOBS` a backend applies; asking for
    any other knob raises instead of being silently ignored. ``prompt_modes``
    lists the prompts it can send, its default first: ``native`` is the
    model's own input (no text, or its vendor prompt) and ``shared`` is
    :data:`SHARED_PROMPT`.
    """

    supports: frozenset[str] = frozenset()
    prompt_modes: tuple[str, ...] = ("native",)

    def __init__(self, model_name: str, batch_size: int = 1, prompt_mode: str | None = None) -> None:
        """
        Initialize the transcriber with a model name and batch size.

        :param model_name: HuggingFace model identifier.
        :param batch_size: Batch size for inference.
        :param prompt_mode: One of :attr:`prompt_modes`, or ``None`` for the default.
        :raises ValueError: If this backend has no such prompt mode.
        """
        if prompt_mode is not None and prompt_mode not in self.prompt_modes:
            raise ValueError(
                f"{type(self).__name__} has no prompt_mode {prompt_mode!r}; expected one of {self.prompt_modes}"
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self.prompt_mode = prompt_mode or self.prompt_modes[0]
        self.last_generate_kwargs = None

    @abstractmethod
    def load(self) -> None:
        """
        Load the model and any required processors.

        Called once before processing begins. Implementations should
        handle device placement and precision settings.
        """
        ...

    def check_decode(self, decode: Decode) -> None:
        """
        Refuse decoding settings this backend can't apply.

        :param decode: Requested settings.
        :raises ValueError: If a knob outside :attr:`supports` is set.
        """
        unsupported = decode.active() - set(self.supports)
        if unsupported:
            raise ValueError(f"{type(self).__name__} cannot apply {sorted(unsupported)}")

    def effective(self) -> dict:
        """
        What the loaded backend actually runs with, for the job record.

        :return: Dtype, attention implementation, checkpoint revision and
            backend-specific settings.
        """
        model = getattr(self, "model", None)
        config = getattr(model, "config", None)
        info = {
            "class": type(self).__name__,
            "batch_size": self.batch_size,
            "supports": sorted(self.supports),
            "prompt_mode": self.prompt_mode,
            "last_generate_kwargs": self.last_generate_kwargs,
        }
        if model is not None and hasattr(model, "dtype"):
            info["dtype"] = str(model.dtype)
        if isinstance(model, torch.nn.Module):
            # ``model.dtype`` is the first parameter's; mixed-precision models need the full count.
            counts: dict[str, int] = {}
            for p in model.parameters():
                counts[str(p.dtype)] = counts.get(str(p.dtype), 0) + p.numel()
            info["param_dtypes"] = counts
        if config is not None:
            info["attn_implementation"] = getattr(config, "_attn_implementation", None)
            info["model_commit_hash"] = getattr(config, "_commit_hash", None)
            info["name_or_path"] = getattr(config, "_name_or_path", None)
        return info

    @abstractmethod
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
        Transcribe a single audio file, optionally with VAD and/or chunking.

        :param audio_path: Absolute path to the audio file.
        :param use_vad: Whether to use VAD to segment the audio.
        :param vad_model: Loaded Silero VAD model instance (required if use_vad=True).
        :param use_chunked: Whether to use fixed-length chunked inference.
        :param language: Optional language code hint.
        :param vad_source: Optional path to an audio file to use as VAD source
            (timestamps from this file, transcription from audio_path).
        :param vad_method: VAD algorithm — ``"silero"`` (default) or
            ``"rms"`` (RMS-amplitude on a separated vocal track).
        :param decode: Beam width and repetition controls; a backend raises
            on any it can't apply (:meth:`check_decode`).
        :param condition_on_prev: Condition each long-form window on the
            previous window's text. Whisper's pipeline path only.
        :param temperature_fallback: Re-decode degenerate windows at rising
            temperature. Whisper's pipeline path only.
        :param compression_ratio_threshold: Fallback trigger (default 2.4).
            Whisper's pipeline path only.
        :param logprob_threshold: Fallback trigger (default -1.0).
            Whisper's pipeline path only.
        :param initial_prompt: Per-song text prompt: Whisper's decoder
            prompt, or Qwen3-ASR's ``context``.
        :return: Transcribed text.
        """
        ...


class ChunkTranscriber(Transcriber):
    """
    Transcriber that decodes a song as its VAD chunks, ``batch_size`` at a time.

    Subclasses implement :meth:`_decode` and may override :attr:`load_audio`.
    Every chunk is zero-padded to at least 0.5 s.
    """

    load_audio = staticmethod(load_mono)

    @abstractmethod
    def _decode(self, wavs: list[np.ndarray], language: str | None, decode: Decode) -> list[str]:
        """
        Transcribe a batch of chunks from one song.

        :param wavs: 16 kHz mono chunks.
        :param language: ``detected_language`` from ``lyrics.json``, or ``None``.
        :param decode: Decoding settings (already checked against :attr:`supports`).
        :return: One transcription per chunk.
        """
        ...

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
        Transcribe one song chunk by chunk and join the chunk texts.
        """
        del use_chunked, condition_on_prev, temperature_fallback
        del compression_ratio_threshold, logprob_threshold, initial_prompt
        self.check_decode(decode)
        chunks = [
            pad_short(c)
            for c in song_chunks(
                audio_path, use_vad, vad_method, vad_source, load=self.load_audio, vad_model=vad_model
            )
        ]
        texts: list[str] = []
        step = max(1, self.batch_size)
        for i in range(0, len(chunks), step):
            texts.extend(self._decode(chunks[i:i + step], language, decode))
        return " ".join(t.strip() for t in texts if t.strip())
