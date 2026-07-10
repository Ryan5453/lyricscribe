import json
import logging
import math
import random
from pathlib import Path
from typing import Iterator

import soundfile as sf
from transformers import WhisperTokenizer

logger = logging.getLogger(__name__)

_RNG = random.Random(42)


def _window_has_vocals(
    synced: list[dict], window_start_ms: float, window_end_ms: float
) -> bool:
    """
    True if any non-empty synced line has most of its duration inside
    the window.
    """
    for line in synced:
        if not line.get("text", "").strip():
            continue
        line_start = line.get("start", 0)
        line_dur = line.get("duration", 0)
        if line_dur <= 0:
            if window_start_ms <= line_start < window_end_ms:
                return True
            continue
        overlap = min(line_start + line_dur, window_end_ms) - max(
            line_start, window_start_ms
        )
        if overlap > 0.5 * line_dur:
            return True
    return False


class LyricsDataset:
    """
    Dataset that iterates over song directories, yielding audio paths
    and transcripts for finetuning.

    Each song directory must contain a ``lyrics.json`` file and at least one
    of the requested audio filenames.

    :param dataset_dir: Root directory containing song subdirectories.
    :param filenames: Audio filenames to look for in each song directory.
        When multiple are provided, one is chosen at random per iteration.
    """

    def __init__(
        self,
        dataset_dir: Path,
        filenames: list[str],
    ):
        self.dataset_dir = Path(dataset_dir)
        self.filenames = filenames

        self.songs = []
        for song_dir in sorted(self.dataset_dir.iterdir()):
            if not song_dir.is_dir():
                continue

            lyrics_path = song_dir / "lyrics.json"
            if not lyrics_path.exists():
                continue

            # Song must have at least one of the requested audio files
            available = [f for f in filenames if (song_dir / f).exists()]
            if not available:
                continue

            self.songs.append({
                "id": song_dir.name,
                "dir": song_dir,
                "available": available,
            })

        logger.info(f"Loaded {len(self.songs)} songs from {dataset_dir} ({', '.join(filenames)})")

    def __len__(self) -> int:
        """
        Return the number of songs in the dataset.

        :return: Song count.
        """
        return len(self.songs)

    def __iter__(self) -> Iterator[dict]:
        """
        Yield dicts with ``song_id``, ``audio_path``, and ``transcript``
        for each song that has a non-empty unsynced transcript.

        :return: Iterator of song dicts.
        """
        for song in self.songs:
            lyrics_path = song["dir"] / "lyrics.json"

            try:
                with open(lyrics_path) as f:
                    lyrics_data = json.load(f)
                transcript = lyrics_data["unsynced"]["data"].strip()

                if not transcript:
                    continue

                audio_file = song["dir"] / _RNG.choice(song["available"])

                yield {
                    "song_id": song["id"],
                    "audio_path": str(audio_file),
                    "transcript": transcript,
                }

            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Failed to load {song['id']}: {e}")
                continue


def create_manifest(
    dataset: LyricsDataset,
    output_path: Path,
    architecture: str = "parakeet",
    model_name: str | None = None,
    window_seconds: float = 30.0,
    max_tokens: int = 440,
    windows_per_song_multiplier: int = 1,
    line_overlap_threshold: float = 0.7,
    dedup_max_run: int = 0,
) -> int:
    """
    Create a manifest JSONL by sampling random fixed-length audio windows
    from each song. Synced lyric lines whose start time falls within a
    window become that window's text label; windows that land on purely
    instrumental regions get an empty text label and are kept as-is so
    the model learns to stay silent during non-vocal audio.

    This matches the inference-time distribution: HF's Whisper pipeline
    slices long songs into arbitrary 30s windows at inference, so
    training must expose the model to arbitrary boundaries too.
    Parakeet's NeMo transcribe path handles variable-length audio at
    inference natively, but still benefits from seeing a mix of vocal
    and instrumental contexts during training.

    The number of windows per song is proportional to song duration so
    longer songs contribute more samples. All offsets are drawn from the
    module-level ``_RNG`` for reproducibility.

    For Whisper, windows whose tokenized text exceeds *max_tokens* are
    dropped (the 448-token decoder cap). Parakeet/Canary skip this check.

    For Canary, each entry additionally carries ``source_lang``,
    ``target_lang``, ``pnc``, and ``answer`` fields for the Lhotse
    prompt-based data pipeline.

    :param dataset: Dataset instance providing song directories with
        ``lyrics.json`` files.
    :param output_path: Path to write the manifest JSONL file.
    :param architecture: Model architecture name (``"whisper"``,
        ``"parakeet"``, or ``"canary"``).
    :param model_name: HuggingFace model identifier, required for
        ``"whisper"`` to load the tokenizer for token-count checking.
    :param window_seconds: Fixed window duration in seconds.
    :param max_tokens: Whisper decoder token cap. Only applied when
        ``architecture == "whisper"``.
    :param windows_per_song_multiplier: Multiplier on the base
        ``ceil(song_duration / window_seconds)`` window count. Default 1
        gives ~1 random window per ``window_seconds`` of song. Raising
        it toward the epoch count approximates per-epoch randomization
        (the model sees different slices of each song within the
        manifest, just in shuffled order across epochs).
    :param line_overlap_threshold: Minimum fraction of a synced line's
        duration that must fall inside a window for that line to be
        kept as part of the window's label. Used only when a song has
        no MFA alignment in its ``lyrics.json`` (fallback path).
        Guards against the model being asked to transcribe lyrics from
        partial audio (hallucination bias) or to stay silent on
        audible vocals that have no text label (deletion bias).
        Default 0.7 means we keep a line only if at least 70% of its
        duration is inside the window.
    :param dedup_max_run: If > 0, cap runs of consecutive identical
        words in the label at this length. Off by default — capping
        measurably hurt WER.

    When a song's ``lyrics.json`` carries an ``alignment`` field (from
    ``lyricscribe dataset align``), word-level MFA times are used
    directly: a word is kept only if its full ``[start, start+duration]``
    falls inside the window. Windows with vocals (per synced lines) but
    no aligned words are skipped — the gap is an alignment failure, not
    silence.

    For Whisper, each entry also carries ``segments``
    (``[start_s, end_s, text]`` relative to window start) for
    timestamp-mode labels.

    :return: Number of manifest entries written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    if architecture == "whisper":
        if model_name is None:
            raise ValueError(
                "model_name is required for whisper manifest creation"
            )
        logger.info(f"  Loading tokenizer: {model_name}")
        tokenizer = WhisperTokenizer.from_pretrained(model_name)

    window_ms = window_seconds * 1000

    count = 0
    skipped_no_sync = 0
    skipped_too_short = 0
    skipped_too_long = 0
    skipped_mfa_gap = 0
    total = len(dataset)

    with open(output_path, "w") as f:
        for idx, song in enumerate(dataset.songs):
            if (idx + 1) % 500 == 0:
                logger.info(f"  Processing manifest: {idx + 1}/{total}")

            lyrics_path = song["dir"] / "lyrics.json"
            try:
                with open(lyrics_path) as lf:
                    lyrics_data = json.load(lf)

                synced = lyrics_data.get("synced", {}).get("data", [])
                if not synced:
                    skipped_no_sync += 1
                    continue

                # Opt-in word-level filtering when MFA alignment is present.
                aligned_words = None
                alignment = lyrics_data.get("alignment")
                if alignment:
                    aligned_words = alignment.get("words")

                detected_language = lyrics_data["detected_language"]
                audio_file = song["dir"] / _RNG.choice(song["available"])

                try:
                    info = sf.info(str(audio_file))
                    song_duration = info.duration
                except Exception as e:
                    logger.warning(
                        f"Failed to read audio info for {song['id']}: {e}"
                    )
                    continue

                if song_duration < 1.0:
                    skipped_too_short += 1
                    continue

                # Songs shorter than one window collapse to a single entry
                # at offset 0 (no point in sampling the same position
                # multiple times).
                if song_duration <= window_seconds:
                    offsets = [0.0]
                else:
                    base_windows = math.ceil(song_duration / window_seconds)
                    num_windows = base_windows * windows_per_song_multiplier
                    max_offset = song_duration - window_seconds
                    offsets = [
                        _RNG.uniform(0.0, max_offset)
                        for _ in range(num_windows)
                    ]

                for offset in offsets:
                    window_start_ms = offset * 1000
                    window_end_ms = window_start_ms + window_ms

                    # (start_ms, end_ms, [words]) segments; joined words
                    # form the text label.
                    segments_raw: list[list] = []
                    if aligned_words is not None:
                        kept = [
                            w for w in aligned_words
                            if w["start"] >= window_start_ms
                            and w["start"] + w["duration"] <= window_end_ms
                        ]
                        # Split into segments at pauses >1s.
                        for w in kept:
                            w_start = w["start"]
                            w_end = w["start"] + w["duration"]
                            if (
                                segments_raw
                                and w_start - segments_raw[-1][1] <= 1000
                            ):
                                segments_raw[-1][1] = max(
                                    segments_raw[-1][1], w_end
                                )
                                segments_raw[-1][2].append(w["word"])
                            else:
                                segments_raw.append(
                                    [w_start, w_end, [w["word"]]]
                                )

                        # Vocals present but no aligned words: alignment
                        # gap, skip.
                        if not segments_raw and _window_has_vocals(
                            synced, window_start_ms, window_end_ms
                        ):
                            skipped_mfa_gap += 1
                            continue
                    else:
                        for line in synced:
                            if not line.get("text", "").strip():
                                continue
                            line_start = line.get("start", 0)
                            line_dur = line.get("duration", 0)
                            if line_dur <= 0:
                                if window_start_ms <= line_start < window_end_ms:
                                    segments_raw.append(
                                        [line_start, line_start,
                                         line["text"].strip().split()]
                                    )
                                continue
                            line_end = line_start + line_dur
                            overlap_start = max(line_start, window_start_ms)
                            overlap_end = min(line_end, window_end_ms)
                            overlap = max(0, overlap_end - overlap_start)
                            if overlap / line_dur >= line_overlap_threshold:
                                segments_raw.append(
                                    [max(line_start, window_start_ms),
                                     min(line_end, window_end_ms),
                                     line["text"].strip().split()]
                                )

                    if dedup_max_run > 0:
                        for seg in segments_raw:
                            deduped: list[str] = []
                            for w in seg[2]:
                                wl = w.lower()
                                if (
                                    len(deduped) >= dedup_max_run
                                    and all(
                                        d.lower() == wl
                                        for d in deduped[-dedup_max_run:]
                                    )
                                ):
                                    continue
                                deduped.append(w)
                            seg[2] = deduped

                    segments_raw = [s for s in segments_raw if s[2]]
                    text = " ".join(
                        w for seg in segments_raw for w in seg[2]
                    ).strip()

                    if tokenizer is not None and text:
                        token_count = len(tokenizer(text).input_ids)
                        if token_count > max_tokens:
                            skipped_too_long += 1
                            continue

                    duration = min(
                        window_seconds, song_duration - offset
                    )
                    if duration <= 0.1:
                        continue

                    entry = {
                        "audio_filepath": str(audio_file),
                        "offset": offset,
                        "duration": duration,
                        "text": text,
                        "language": detected_language,
                    }
                    if architecture == "whisper" and segments_raw:
                        # Window-relative seconds, clamped and monotonic.
                        segs = []
                        prev_end = 0.0
                        for s_ms, e_ms, words in segments_raw:
                            s = min(max((s_ms - window_start_ms) / 1000.0, prev_end), duration)
                            e = min(max((e_ms - window_start_ms) / 1000.0, s), duration)
                            segs.append([round(s, 3), round(e, 3), " ".join(words)])
                            prev_end = e
                        entry["segments"] = segs
                    if architecture == "canary":
                        entry["answer"] = text
                        entry["source_lang"] = detected_language
                        entry["target_lang"] = detected_language
                        # pnc must reflect the target's actual style.
                        has_style = any(c.isupper() for c in text) or any(
                            c in text for c in ".,!?;:"
                        )
                        entry["pnc"] = "yes" if has_style else "no"

                    f.write(json.dumps(entry) + "\n")
                    count += 1

            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Failed to process {song['id']}: {e}")
                continue

    logger.info(
        f"Created manifest with {count} windows "
        f"({skipped_no_sync} songs skipped for missing synced lyrics, "
        f"{skipped_too_short} songs skipped for being too short, "
        f"{skipped_too_long} windows dropped for exceeding {max_tokens} tokens, "
        f"{skipped_mfa_gap} windows dropped for MFA alignment gaps over vocals)"
    )
    return count


def write_subset_manifest(
    source: Path,
    dest: Path,
    size: int,
    seed: int = 42,
) -> int:
    """
    Copy a deterministic shuffled subset of a JSONL manifest.

    Used for the during-training validation subset: we want Whisper and
    NeMo to evaluate on the exact same set of samples each epoch, so we
    materialize one shared subset manifest at setup time instead of
    letting each trainer pick its own window (``torch.Subset`` vs
    ``limit_val_batches``, which produce different sizes *and*
    different contents).

    The shuffle is seeded so re-running setup on the same full manifest
    always produces the same subset — reproducible, and subsetting a
    subset is a no-op.

    :param source: Path to the full manifest (JSONL).
    :param dest: Path to write the subset to (JSONL).
    :param size: Maximum number of entries in the subset. If the source
        has fewer entries, the whole thing is copied.
    :param seed: Shuffle seed (default 42).
    :return: Number of entries written to the subset manifest.
    """
    import random as _random

    with open(source) as f:
        lines = f.readlines()

    rng = _random.Random(seed)
    rng.shuffle(lines)
    lines = lines[:size]

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        f.writelines(lines)

    return len(lines)
