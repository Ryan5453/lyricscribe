import logging

import torch
import torchaudio
from silero_vad import get_speech_timestamps
from torch.jit import ScriptModule
from transformers import pipeline

from lyricscribe.transcribe.base import Transcriber

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

    def transcribe(self, audio_path: str, language: str | None = None) -> str:
        """
        Transcribe a full audio file using sequential decoding.

        Always uses Whisper's native sequential long-form algorithm
        (no chunked batching) for consistent results regardless of
        batch size.

        :param audio_path: Path to the audio file.
        :param language: Optional ISO 639-1 language code hint.
        :return: Transcribed text.
        """
        generate_kwargs = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language
        result = self.pipe(audio_path, generate_kwargs=generate_kwargs)
        return result["text"].strip()

    def transcribe_with_vad(
        self,
        audio_path: str,
        vad_model: ScriptModule,
        language: str | None = None,
    ) -> str:
        """
        Transcribe with VAD by batching speech segments through the pipeline.

        Loads the audio, resamples to 16 kHz mono, runs Silero VAD to
        identify speech regions, then passes all segments as a list to
        the pipeline for batched processing. Each segment uses sequential
        decoding independently.

        :param audio_path: Path to the audio file.
        :param vad_model: Loaded Silero VAD model.
        :param language: Optional language code hint.
        :return: Concatenated transcription of speech segments.
        """
        wav, sample_rate = torchaudio.load(audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            wav = torchaudio.functional.resample(wav, sample_rate, 16000)
            sample_rate = 16000

        wav_1d = wav.squeeze(0)
        timestamps = get_speech_timestamps(
            wav_1d,
            vad_model,
            sampling_rate=sample_rate,
            return_seconds=False,
        )

        if not timestamps:
            return ""

        segments = [
            {
                "raw": wav_1d[seg["start"] : seg["end"]].numpy(),
                "sampling_rate": sample_rate,
            }
            for seg in timestamps
        ]

        generate_kwargs = {"task": "transcribe"}
        if language:
            generate_kwargs["language"] = language

        results = self.pipe(
            segments,
            batch_size=self.batch_size,
            generate_kwargs=generate_kwargs,
        )

        parts = [r["text"].strip() for r in results if r["text"].strip()]
        return " ".join(parts)
