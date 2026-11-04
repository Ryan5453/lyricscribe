import logging

import torch
import torchaudio
from silero_vad import get_speech_timestamps
from torch.jit import ScriptModule
from transformers import pipeline

from lyricscribe.transcribe.base import Transcriber
from lyricscribe.transcribe.rms_vad import (
    RmsVadOptions,
    get_speech_timestamps_rms,
    merge_segments,
)

logger = logging.getLogger(__name__)


class WhisperTranscriber(Transcriber):
    """
    Transcriber using OpenAI Whisper via HuggingFace Transformers.

    Uses the ``automatic-speech-recognition`` pipeline with float16
    precision on CUDA and optional flash attention. Falls back to
    standard attention if flash attention is unavailable.
    """

    def __init__(self, model_name: str, batch_size: int = 1) -> None:
        """
        Initialize the Whisper transcriber.

        :param model_name: HuggingFace model identifier (e.g.
            ``openai/whisper-large-v3``).
        :param batch_size: Batch size for chunked pipeline processing.
        """
        super().__init__(model_name, batch_size)
        self.pipe = None

    def load(self) -> None:
        """
        Load the Whisper pipeline with optional flash attention.

        Attempts CUDA with float16 and flash attention first. Falls back
        to standard attention if flash_attn is not installed, and to CPU
        if CUDA is unavailable.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper model: {self.model_name} on {device}")
        # cuDNN SDPA drifts from the flash kernels and rebuilds per sequence shape.
        torch.backends.cuda.enable_cudnn_sdp(False)

        kwargs = {
            "task": "automatic-speech-recognition",
            "model": self.model_name,
            "device": device,
            "return_timestamps": True,
        }

        if device == "cuda":
            kwargs["torch_dtype"] = torch.float16
            try:
                kwargs["model_kwargs"] = {"attn_implementation": "flash_attention_2"}
                self.pipe = pipeline(**kwargs)
                logger.info("Loaded with flash attention")
                return
            except Exception:
                logger.info("Flash attention unavailable, falling back to default")
                kwargs.pop("model_kwargs", None)

        self.pipe = pipeline(**kwargs)

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
        """
        generate_kwargs = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language
        if initial_prompt:
            generate_kwargs["prompt_ids"] = self.pipe.tokenizer.get_prompt_ids(
                initial_prompt, return_tensors="pt"
            ).to(self.pipe.model.device)
        if repetition_penalty and repetition_penalty != 1.0:
            generate_kwargs["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size and no_repeat_ngram_size > 0:
            generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
        # The HF ASR pipeline defaults to 5 beams, so num_beams is always explicit.
        generate_kwargs["num_beams"] = num_beams
        if not condition_on_prev:
            generate_kwargs["condition_on_prev_tokens"] = False
        if temperature_fallback:
            generate_kwargs["temperature"] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            generate_kwargs["compression_ratio_threshold"] = (
                compression_ratio_threshold if compression_ratio_threshold is not None else 2.4
            )
            generate_kwargs["logprob_threshold"] = (
                logprob_threshold if logprob_threshold is not None else -1.0
            )

        kwargs = {
            # bs>1 needs ``chunk_length_s``. It leaves RMS-VAD's <=30s segments whole
            # but would re-chunk uncapped Silero segments, so Silero stays at bs=1.
            "batch_size": self.batch_size if (use_chunked or (use_vad and vad_method == "rms")) else 1,
            "generate_kwargs": generate_kwargs,
        }
        if use_chunked or (use_vad and vad_method == "rms"):
            kwargs["chunk_length_s"] = 30
            kwargs["stride_length_s"] = (4, 2)

        if use_vad:
            if vad_method == "silero" and vad_model is None:
                raise ValueError(
                    "vad_model must be provided when use_vad=True and vad_method='silero'"
                )
            if vad_method not in ("silero", "rms"):
                raise ValueError(f"Unknown vad_method: {vad_method!r}")

            wav, sample_rate = torchaudio.load(audio_path)
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sample_rate != 16000:
                wav = torchaudio.functional.resample(wav, sample_rate, 16000)
                sample_rate = 16000

            if vad_source is not None:
                vad_wav, vad_sr = torchaudio.load(vad_source)
                if vad_wav.shape[0] > 1:
                    vad_wav = vad_wav.mean(dim=0, keepdim=True)
                if vad_sr != 16000:
                    vad_wav = torchaudio.functional.resample(vad_wav, vad_sr, 16000)
                vad_1d = vad_wav.squeeze(0)
            else:
                vad_1d = wav.squeeze(0)

            if vad_method == "silero":
                timestamps = get_speech_timestamps(
                    vad_1d,
                    vad_model,
                    sampling_rate=16000,
                    return_seconds=False,
                )
            else:
                rms_segments = get_speech_timestamps_rms(
                    vad_1d.numpy(),
                    vad_options=RmsVadOptions(),
                    window_size_samples=512,
                    sampling_rate=16000,
                )
                timestamps = merge_segments(
                    rms_segments, max_length_s=30, sampling_rate=16000
                )

            if not timestamps:
                return ""

            wav_1d = wav.squeeze(0)

            max_samples = wav_1d.shape[0]
            timestamps = [
                {
                    "start": min(seg["start"], max_samples),
                    "end": min(seg["end"], max_samples),
                }
                for seg in timestamps
                if seg["start"] < max_samples
            ]

            segments = [
                {
                    "raw": wav_1d[seg["start"] : seg["end"]].numpy(),
                    "sampling_rate": 16000,
                }
                for seg in timestamps
            ]

            results = self.pipe(segments, **kwargs)
            if isinstance(results, dict):
                results = [results]
            parts = [r["text"].strip() for r in results if r["text"].strip()]
            return " ".join(parts)
        else:
            result = self.pipe(audio_path, **kwargs)
            return result["text"].strip()
