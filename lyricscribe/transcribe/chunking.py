from collections.abc import Callable

import numpy as np
import torch
import torchaudio
from silero_vad import get_speech_timestamps
from torch.jit import ScriptModule

from lyricscribe.transcribe.rms_vad import (
    RmsVadOptions,
    get_speech_timestamps_rms,
    merge_segments,
)

SAMPLE_RATE = 16000

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
}


def language_code(language: str | None) -> str | None:
    """
    Reduce a ``detected_language`` value (``en``, ``en-US``, ``es_ES``) to
    its two-letter code.

    :param language: Language string from ``lyrics.json``, or ``None``.
    :return: Lower-case base code, or ``None``.
    """
    if not language:
        return None
    return language.split("-")[0].split("_")[0].lower()


def language_name(language: str | None) -> str | None:
    """
    English name of a language code, as the LLM prompts expect.

    :param language: Language string from ``lyrics.json``, or ``None``.
    :return: Language name (unknown codes are passed through), or ``None``.
    """
    code = language_code(language)
    if code is None:
        return None
    return LANGUAGE_NAMES.get(code, code)


def load_mono(path: str) -> np.ndarray:
    """
    Load an audio file as 16 kHz mono float32 with torchaudio's decoder and
    resampler: the one loader every inference backend shares. Trainers read
    windows with ``finetune.engine.load_window`` (the same mono mean and
    torchaudio resample, per window).

    Parakeet and Canary match their NeMo transcriptions only with this
    loader. Samples are not clipped or peak-normalised.

    :param path: Audio file path.
    :return: 1-D waveform.
    """
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    return wav.numpy().astype(np.float32, copy=False)


load_mono_torchaudio = load_mono


def pad_short(chunk: np.ndarray, min_seconds: float = 0.5) -> np.ndarray:
    """
    Zero-pad a chunk to at least ``min_seconds``.

    RMS-VAD has no minimum speech duration, so it can emit segments too short
    for a feature extractor's STFT window.

    :param chunk: 16 kHz mono waveform.
    :param min_seconds: Minimum length after padding.
    :return: The chunk, padded at the end when shorter than ``min_seconds``.
    """
    min_len = int(min_seconds * SAMPLE_RATE)
    if len(chunk) >= min_len:
        return chunk
    return np.pad(chunk, (0, min_len - len(chunk)))


def vad_chunks(
    audio_path: str,
    vad_source: str | None = None,
    max_length_s: float = 30,
    load: Callable[[str], np.ndarray] = load_mono,
) -> list[np.ndarray]:
    """
    RMS-VAD segments of ``audio_path`` merged into ``<= max_length_s`` chunks.

    Boundaries come from ``vad_source`` when given (e.g. separated vocals),
    and are clipped to the transcription audio's length.

    :param audio_path: Audio to transcribe.
    :param vad_source: Audio to run the VAD on (defaults to ``audio_path``).
    :param max_length_s: Maximum merged chunk length in seconds.
    :param load: Audio loader returning 16 kHz mono float32.
    :return: Chunk waveforms in order; empty if no speech was found.
    """
    audio = load(audio_path)
    if vad_source and str(vad_source) != str(audio_path):
        vad_audio = load(vad_source)
    else:
        vad_audio = audio
    timestamps = get_speech_timestamps_rms(
        vad_audio, RmsVadOptions(), sampling_rate=SAMPLE_RATE
    )
    timestamps = merge_segments(
        timestamps, max_length_s=max_length_s, sampling_rate=SAMPLE_RATE
    )
    return _slice_segments(audio, timestamps)


def silero_chunks(
    audio_path: str,
    vad_model: ScriptModule,
    vad_source: str | None = None,
    load: Callable[[str], np.ndarray] = load_mono,
) -> list[np.ndarray]:
    """
    Silero-VAD speech segments of ``audio_path``, unmerged, as the Whisper
    backend cuts them.

    :param audio_path: Audio to transcribe.
    :param vad_model: Loaded Silero VAD model.
    :param vad_source: Audio to run the VAD on (defaults to ``audio_path``).
    :param load: Audio loader returning 16 kHz mono float32.
    :return: Chunk waveforms in order; empty if no speech was found.
    """
    audio = load(audio_path)
    vad_audio = load(vad_source) if vad_source and str(vad_source) != str(audio_path) else audio
    timestamps = get_speech_timestamps(
        torch.from_numpy(np.ascontiguousarray(vad_audio, dtype=np.float32)),
        vad_model,
        sampling_rate=SAMPLE_RATE,
    )
    return _slice_segments(audio, timestamps)


def _slice_segments(audio: np.ndarray, timestamps: list[dict]) -> list[np.ndarray]:
    """
    Cut sample-index segments out of ``audio``, clipped to its length.

    :param audio: 16 kHz mono waveform.
    :param timestamps: ``{"start", "end"}`` sample-index dicts.
    :return: Non-empty chunk waveforms in order.
    """
    chunks = []
    for segment in timestamps:
        start = min(int(segment["start"]), len(audio))
        end = min(int(segment["end"]), len(audio))
        if end > start:
            chunks.append(np.asarray(audio[start:end], dtype=np.float32))
    return chunks


def song_chunks(
    audio_path: str,
    use_vad: bool,
    vad_method: str,
    vad_source: str | None,
    load: Callable[[str], np.ndarray] = load_mono,
    vad_model: ScriptModule | None = None,
) -> list[np.ndarray]:
    """
    Chunks for one song: RMS-VAD chunks merged to at most 30 s, Silero-VAD
    speech segments, or the whole song when VAD is off.

    :param audio_path: Audio to transcribe.
    :param use_vad: Whether to segment with VAD.
    :param vad_method: ``"rms"`` or ``"silero"``.
    :param vad_source: Optional separate VAD source file.
    :param load: Audio loader returning 16 kHz mono float32.
    :param vad_model: Loaded Silero VAD model, required for ``"silero"``.
    :return: Chunk waveforms.
    :raises ValueError: For an unknown method, or Silero without a model.
    """
    if not use_vad:
        return [load(audio_path)]
    if vad_method == "rms":
        return vad_chunks(audio_path, vad_source, load=load)
    if vad_method == "silero":
        if vad_model is None:
            raise ValueError("vad_method='silero' needs a loaded Silero VAD model")
        return silero_chunks(audio_path, vad_model, vad_source, load=load)
    raise ValueError(f"Unknown vad_method: {vad_method!r}")
