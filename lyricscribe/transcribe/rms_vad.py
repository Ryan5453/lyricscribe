"""
RMS-amplitude voice activity detection on separated vocals.

Ported from https://github.com/jaza-syed/mss-alt
(Syed et al., "Exploiting Music Source Separation for Automatic Lyrics
Transcription with Whisper", arXiv:2506.15514).

Unlike speech-trained VAD (Silero), this thresholds the normalized RMS
amplitude of the (already separated) vocal track. Works on sustained
tones, melismas, and non-lexical vocables that speech VAD under-fires on.
"""

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class RmsVadOptions:
    """Threshold and shape options for RMS-based VAD."""

    onset: float = 0.1
    offset: float = 0.1
    min_speech_duration_ms: float = 0
    max_speech_duration_s: float = 30
    min_silence_duration_ms: float = 1000
    speech_pad_ms: float = 200


def get_speech_timestamps_rms(
    audio: np.ndarray,
    vad_options: RmsVadOptions,
    window_size_samples: int = 512,
    sampling_rate: int = 16000,
) -> list[dict]:
    """
    Threshold normalized RMS amplitude into speech segments.

    Mirrors ``get_speech_timestamps`` in ``silero_vad`` so this can drop
    into the same downstream slicing logic.

    :param audio: 1-D float array of separated vocals.
    :param vad_options: Threshold + duration options.
    :param window_size_samples: Hop length of the RMS feature.
    :param sampling_rate: Sample rate of ``audio``.
    :returns: List of ``{"start": int, "end": int}`` dicts with sample
        indices.
    """
    onset = vad_options.onset
    offset = vad_options.offset
    min_speech_duration_ms = vad_options.min_speech_duration_ms
    max_speech_duration_s = vad_options.max_speech_duration_s
    min_silence_duration_ms = vad_options.min_silence_duration_ms
    speech_pad_ms = vad_options.speech_pad_ms
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = (
        sampling_rate * max_speech_duration_s
        - window_size_samples
        - 2 * speech_pad_samples
    )
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

    audio_length_samples = len(audio)

    rms = np.mean(
        librosa.feature.rms(y=audio, frame_length=2048, hop_length=window_size_samples),
        axis=0,
    )
    probs = rms / np.max(rms) if np.max(rms) > 0 else rms

    triggered = False
    speeches: list[dict] = []
    current_speech: dict = {}
    temp_end = 0
    prev_end = next_start = 0

    for i, speech_prob in enumerate(probs):
        if (speech_prob >= onset) and temp_end:
            temp_end = 0
            if next_start < prev_end:
                next_start = window_size_samples * i

        if (speech_prob >= onset) and not triggered:
            triggered = True
            current_speech["start"] = window_size_samples * i
            continue

        if (
            triggered
            and (window_size_samples * i) - current_speech["start"] > max_speech_samples
        ):
            if prev_end:
                current_speech["end"] = prev_end
                speeches.append(current_speech)
                current_speech = {}
                if next_start < prev_end:
                    triggered = False
                else:
                    current_speech["start"] = next_start
                prev_end = next_start = temp_end = 0
            else:
                # Min-cut on the second half if no prior silence to fall back on.
                start_frame = current_speech["start"] // window_size_samples
                end_frame = i + 1
                segment_scores = probs[start_frame:end_frame]
                second_half_start = len(segment_scores) // 2
                search_segment = segment_scores[second_half_start:]
                min_val = search_segment.min()
                min_indices = np.where(search_segment == min_val)[0]
                min_index_in_half = min_indices[-1]
                chosen_frame = start_frame + second_half_start + min_index_in_half
                min_cut_sample = window_size_samples * chosen_frame
                current_speech["end"] = min_cut_sample
                speeches.append(current_speech)
                current_speech = {"start": min_cut_sample}
                triggered = True
                prev_end = next_start = temp_end = 0
                continue

        if (speech_prob < offset) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            if (window_size_samples * i) - temp_end > min_silence_samples_at_max_speech:
                prev_end = temp_end
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            current_speech["end"] = temp_end
            if (current_speech["end"] - current_speech["start"]) > min_speech_samples:
                speeches.append(current_speech)
            current_speech = {}
            prev_end = next_start = temp_end = 0
            triggered = False
            continue

    if (
        current_speech
        and (audio_length_samples - current_speech["start"]) > min_speech_samples
    ):
        current_speech["end"] = audio_length_samples
        speeches.append(current_speech)

    # Apply symmetric padding without crossing into neighbouring segments.
    for i, speech in enumerate(speeches):
        if i == 0:
            speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i + 1]["start"] - speech["end"]
            if silence_duration < 2 * speech_pad_samples:
                speech["end"] += int(silence_duration // 2)
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - silence_duration // 2)
                )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )
                speeches[i + 1]["start"] = int(
                    max(0, speeches[i + 1]["start"] - speech_pad_samples)
                )
        else:
            speech["end"] = int(
                min(audio_length_samples, speech["end"] + speech_pad_samples)
            )

    return speeches


def merge_segments(
    segments: list[dict],
    max_length_s: float = 30,
    sampling_rate: int = 16000,
) -> list[dict]:
    """
    Merge consecutive speech segments up to ``max_length_s``.

    Whisper's input is capped at 30s; the small RMS-VAD segments are
    concatenated into ≤30s chunks to maximise context per inference call.

    :param segments: ``{"start", "end"}`` sample-index dicts from
        :func:`get_speech_timestamps_rms`.
    :param max_length_s: Maximum chunk length (seconds).
    :param sampling_rate: Sample rate of the original audio.
    :returns: Merged ``{"start", "end"}`` dicts with sample indices.
    """
    if not segments:
        return []

    chunk_length = int(max_length_s * sampling_rate)
    merged: list[dict] = []
    curr_start = segments[0]["start"]
    curr_end = segments[0]["end"]

    for seg in segments[1:]:
        if seg["end"] - curr_start > chunk_length and curr_end - curr_start > 0:
            merged.append({"start": curr_start, "end": curr_end})
            curr_start = seg["start"]
        curr_end = seg["end"]

    merged.append({"start": curr_start, "end": curr_end})
    return merged
