import logging

import numpy as np
import torch
import torchaudio
from silero_vad import get_speech_timestamps
from torch.jit import ScriptModule
from transformers import pipeline

from lyricscribe.transcribe.adapter import check_train_record
from lyricscribe.transcribe.base import (
    DECODE_KNOBS,
    MAX_NEW_TOKENS,
    ChunkTranscriber,
    Decode,
    describe_kwargs,
    generate_kwargs,
    shared_prompt,
)
from lyricscribe.transcribe.chunking import SAMPLE_RATE
from lyricscribe.transcribe.rms_vad import (
    RmsVadOptions,
    get_speech_timestamps_rms,
    merge_segments,
)

logger = logging.getLogger(__name__)


DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}


class WhisperTranscriber(ChunkTranscriber):
    """
    Transcriber using OpenAI Whisper via HuggingFace Transformers.

    RMS-VAD chunks (each <= 30 s) go through the shared chunk path: the
    common loader, one short-form ``generate`` per batch with timestamps on,
    and the shared decoding settings. Silero-VAD and full-song inputs keep the
    ``automatic-speech-recognition`` pipeline, whose long-form knobs
    (``condition_on_prev``, temperature fallback) only exist there.
    ``prompt_mode="shared"`` sends the shared prompt as Whisper's previous-text
    prompt (``prompt_ids``) on every chunk.
    """

    supports = frozenset(DECODE_KNOBS)
    prompt_modes = ("native", "shared")

    def __init__(
        self, model_name: str, batch_size: int = 1, dtype: str | None = None, prompt_mode: str | None = None
    ) -> None:
        """
        Initialize the Whisper transcriber.

        :param model_name: HuggingFace model identifier (e.g.
            ``openai/whisper-large-v3``).
        :param batch_size: Chunks decoded per ``generate`` call.
        :param dtype: ``float16`` (default), ``bfloat16`` or ``float32`` on CUDA.
        :param prompt_mode: ``native`` (no prompt, default) or ``shared``.
        """
        super().__init__(model_name, batch_size, prompt_mode)
        if dtype is not None and dtype not in DTYPES:
            raise ValueError(f"Unknown dtype {dtype!r}; expected one of {sorted(DTYPES)}")
        self.dtype = DTYPES[dtype or "float16"]
        self.pipe = None

    def load(self) -> None:
        """
        Load the Whisper pipeline with optional flash attention.

        Attempts CUDA with float16 and flash attention first. Falls back
        to standard attention if flash_attn is not installed, and to CPU
        if CUDA is unavailable.

        :raises TrainRecordMismatch: If a fine-tuned checkpoint was trained with another prompt.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.train_record = check_train_record(self.model_name, self.prompt_mode)
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
            kwargs["torch_dtype"] = self.dtype
            try:
                kwargs["model_kwargs"] = {"attn_implementation": "flash_attention_2"}
                self.pipe = pipeline(**kwargs)
                logger.info("Loaded with flash attention")
                return
            except Exception:
                logger.info("Flash attention unavailable, falling back to default")
                kwargs.pop("model_kwargs", None)

        self.pipe = pipeline(**kwargs)

    @property
    def model(self):
        """
        The pipeline's Whisper model.

        :return: ``WhisperForConditionalGeneration``, or ``None`` before :meth:`load`.
        """
        return self.pipe.model if self.pipe is not None else None

    @torch.inference_mode()
    def _decode(self, wavs: list[np.ndarray], language: str | None, decode: Decode) -> list[str]:
        """
        Decode <= 30 s chunks in one short-form ``generate`` call.

        :param wavs: 16 kHz mono chunks.
        :param language: ``detected_language`` from ``lyrics.json``, or ``None``
            (Whisper then detects it).
        :param decode: Decoding settings.
        :return: One transcription per chunk.
        """
        model = self.pipe.model
        features = self.pipe.feature_extractor(
            [np.asarray(w, dtype=np.float32) for w in wavs], sampling_rate=SAMPLE_RATE, return_tensors="pt"
        )
        input_features = features.input_features.to(model.device, dtype=model.dtype)
        # The decoder prompt is [<|startofprev|> prompt] <|startoftranscript|><|lang|><|transcribe|>;
        # max_target_positions counts all of it.
        prompt = shared_prompt(language) if self.prompt_mode == "shared" else None
        prompt_ids = (
            self.pipe.tokenizer.get_prompt_ids(prompt, return_tensors="pt").to(model.device) if prompt else None
        )
        room = model.config.max_target_positions - 3 - (len(prompt_ids) if prompt_ids is not None else 0)
        kwargs = generate_kwargs(min(MAX_NEW_TOKENS, room), decode)
        extra = {"task": "transcribe", "return_timestamps": True}
        if language:
            extra["language"] = language
        self.last_generate_kwargs = {**describe_kwargs(kwargs), **extra, "prompt": prompt}
        if prompt_ids is not None:
            kwargs["prompt_ids"] = prompt_ids
        ids = model.generate(input_features, **extra, **kwargs)
        ids = ids.sequences if hasattr(ids, "sequences") else ids
        return self.pipe.tokenizer.batch_decode(ids, skip_special_tokens=True)

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
        Transcribe a single audio file: RMS-VAD chunks through the shared chunk
        path, anything else through the HF pipeline.
        """
        if initial_prompt and self.prompt_mode != "native":
            raise ValueError("A per-song initial_prompt and prompt_mode 'shared' can't be combined")
        if use_vad and vad_method == "rms" and not initial_prompt:
            return super().transcribe(
                audio_path, use_vad=use_vad, vad_model=vad_model, language=language,
                vad_source=vad_source, vad_method=vad_method, decode=decode,
            )
        if self.prompt_mode != "native":
            raise ValueError("prompt_mode 'shared' is wired into the RMS-VAD chunk path only")
        self.check_decode(decode)
        pipe_kwargs = {"task": "transcribe"}
        if language:
            pipe_kwargs["language"] = language
        if initial_prompt:
            pipe_kwargs["prompt_ids"] = self.pipe.tokenizer.get_prompt_ids(
                initial_prompt, return_tensors="pt"
            ).to(self.pipe.model.device)
        if decode.repetition_penalty != 1.0:
            pipe_kwargs["repetition_penalty"] = decode.repetition_penalty
        if decode.no_repeat_ngram_size:
            pipe_kwargs["no_repeat_ngram_size"] = decode.no_repeat_ngram_size
        # The HF ASR pipeline defaults to 5 beams, so num_beams is always explicit.
        pipe_kwargs["num_beams"] = decode.num_beams
        if not condition_on_prev:
            pipe_kwargs["condition_on_prev_tokens"] = False
        if temperature_fallback:
            pipe_kwargs["temperature"] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
            pipe_kwargs["compression_ratio_threshold"] = (
                compression_ratio_threshold if compression_ratio_threshold is not None else 2.4
            )
            pipe_kwargs["logprob_threshold"] = (
                logprob_threshold if logprob_threshold is not None else -1.0
            )

        kwargs = {
            # bs>1 needs ``chunk_length_s``. It leaves RMS-VAD's <=30s segments whole
            # but would re-chunk uncapped Silero segments, so Silero stays at bs=1.
            "batch_size": self.batch_size if (use_chunked or (use_vad and vad_method == "rms")) else 1,
            "generate_kwargs": pipe_kwargs,
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
