import logging
from abc import ABC, abstractmethod

import torch
from torch.jit import ScriptModule

logger = logging.getLogger(__name__)


class Transcriber(ABC):
    """
    Abstract base class for ASR transcription models.
    """

    def __init__(self, model_name: str, batch_size: int = 1) -> None:
        """
        Initialize the transcriber with a model name and batch size.

        :param model_name: HuggingFace model identifier.
        :param batch_size: Batch size for inference.
        """
        self.model_name = model_name
        self.batch_size = batch_size

    @abstractmethod
    def load(self) -> None:
        """
        Load the model and any required processors.

        Called once before processing begins. Implementations should
        handle device placement and precision settings.
        """
        ...

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
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        num_beams: int = 5,
        condition_on_prev: bool = True,
        temperature_fallback: bool = False,
        compression_ratio_threshold: float | None = None,
        logprob_threshold: float | None = None,
        initial_prompt: str | None = None,
    ) -> str:
        """
        Transcribe a single audio file, optionally with VAD and/or chunking.

        Every backend accepts all decoding knobs and ignores the ones it
        doesn't support.

        :param audio_path: Absolute path to the audio file.
        :param use_vad: Whether to use VAD to segment the audio.
        :param vad_model: Loaded Silero VAD model instance (required if use_vad=True).
        :param use_chunked: Whether to use fixed-length chunked inference.
        :param language: Optional language code hint.
        :param vad_source: Optional path to an audio file to use as VAD source
            (timestamps from this file, transcription from audio_path).
        :param vad_method: VAD algorithm — ``"silero"`` (default) or
            ``"rms"`` (RMS-amplitude on a separated vocal track).
        :param repetition_penalty: Decoder ``repetition_penalty`` (Whisper,
            NeMo multitask). ``1.0`` disables it.
        :param no_repeat_ngram_size: Decoder ``no_repeat_ngram_size``
            (Whisper, NeMo multitask, audio-LLMs, Qwen3-ASR). ``0`` disables it.
        :param num_beams: Beam width. Whisper-only.
        :param condition_on_prev: Condition each long-form window on the
            previous window's text. Whisper-only.
        :param temperature_fallback: Re-decode degenerate windows at rising
            temperature. Whisper-only.
        :param compression_ratio_threshold: Fallback trigger (default 2.4).
            Whisper-only.
        :param logprob_threshold: Fallback trigger (default -1.0).
            Whisper-only.
        :param initial_prompt: Per-song text prompt: Whisper's decoder
            prompt, or Qwen3-ASR's ``context``.
        :return: Transcribed text.
        """
        ...

    def halve_batch_size(self) -> int:
        """
        Halve the current batch size as an OOM recovery strategy.

        Called by the job runner when a ``torch.cuda.OutOfMemoryError``
        is caught. Clears the CUDA cache before returning.

        :return: The new batch size after halving.
        """
        old = self.batch_size
        self.batch_size = max(1, self.batch_size // 2)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.warning(f"OOM recovery: batch_size {old} -> {self.batch_size}")
        return self.batch_size
