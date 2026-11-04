import numpy as np

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
    Load an audio file as 16 kHz mono float32.

    :param path: Audio file path.
    :return: 1-D waveform.
    """
    import librosa

    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    return audio


def vad_chunks(
    audio_path: str,
    vad_source: str | None = None,
    max_length_s: float = 30,
) -> list[np.ndarray]:
    """
    RMS-VAD segments of ``audio_path`` merged into ``<= max_length_s`` chunks.

    Boundaries come from ``vad_source`` when given (e.g. separated vocals),
    and are clipped to the transcription audio's length.

    :param audio_path: Audio to transcribe.
    :param vad_source: Audio to run the VAD on (defaults to ``audio_path``).
    :param max_length_s: Maximum merged chunk length in seconds.
    :return: Chunk waveforms in order; empty if no speech was found.
    """
    audio = load_mono(audio_path)
    if vad_source and str(vad_source) != str(audio_path):
        vad_audio = load_mono(vad_source)
    else:
        vad_audio = audio
    timestamps = get_speech_timestamps_rms(
        vad_audio, RmsVadOptions(), sampling_rate=SAMPLE_RATE
    )
    timestamps = merge_segments(
        timestamps, max_length_s=max_length_s, sampling_rate=SAMPLE_RATE
    )
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
) -> list[np.ndarray]:
    """
    Chunks for one song: RMS-VAD chunks, or the whole song when VAD is off.

    :param audio_path: Audio to transcribe.
    :param use_vad: Whether to segment with RMS-VAD.
    :param vad_method: Must be ``"rms"`` when ``use_vad`` is set.
    :param vad_source: Optional separate VAD source file.
    :return: Chunk waveforms.
    :raises ValueError: For Silero VAD, which these backends don't support.
    """
    if not use_vad:
        return [load_mono(audio_path)]
    if vad_method != "rms":
        raise ValueError(
            f"vad_method={vad_method!r} is not supported by this backend; use 'rms'"
        )
    return vad_chunks(audio_path, vad_source)
