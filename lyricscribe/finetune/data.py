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
        gives one random window per ``window_seconds`` of song on
        average; raising it trades training time for coverage diversity.
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

                    # Collect synced lines whose start time falls inside
                    # the window. Lines that only partially overlap (start
                    # before or after the window) are excluded to keep
                    # audio/text aligned to what the model will see.
                    lines_in_window = [
                        line for line in synced
                        if window_start_ms <= line.get("start", 0) < window_end_ms
                        and line.get("text", "").strip()
                    ]
                    text = " ".join(
                        line["text"].strip() for line in lines_in_window
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
                    if architecture == "canary":
                        entry["answer"] = text
                        entry["source_lang"] = detected_language
                        entry["target_lang"] = detected_language
                        entry["pnc"] = "no"

                    f.write(json.dumps(entry) + "\n")
                    count += 1

            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Failed to process {song['id']}: {e}")
                continue

    logger.info(
        f"Created manifest with {count} windows "
        f"({skipped_no_sync} songs skipped for missing synced lyrics, "
        f"{skipped_too_short} songs skipped for being too short, "
        f"{skipped_too_long} windows dropped for exceeding {max_tokens} tokens)"
    )
    return count
