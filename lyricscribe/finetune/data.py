import json
import logging
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
    max_duration: float = 300.0,
) -> int:
    """
    Create a NeMo-format manifest JSONL file from a dataset, with one entry
    per song.

    For Canary models, each entry additionally includes ``source_lang``,
    ``target_lang``, ``pnc``, and ``answer`` fields required by the Lhotse
    prompt-based data pipeline.

    :param dataset: Dataset instance to iterate over.
    :param output_path: Path to write the manifest JSONL file.
    :param architecture: Model architecture name. ``"canary"`` triggers
        extra manifest fields.
    :param max_duration: Maximum audio duration in seconds. Songs exceeding
        this are skipped.
    :return: Number of entries written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped_duration = 0
    total = len(dataset)

    with open(output_path, "w") as f:
        for idx, item in enumerate(dataset):
            if (idx + 1) % 500 == 0:
                logger.info(f"  Processing manifest: {idx + 1}/{total}")

            try:
                audio_path = item["audio_path"]
                info = sf.info(audio_path)
                duration = info.duration

                if duration > max_duration:
                    skipped_duration += 1
                    continue

                entry = {
                    "audio_filepath": audio_path,
                    "duration": duration,
                    "text": item["transcript"],
                }

                if architecture == "canary":
                    entry["answer"] = item["transcript"]
                    entry["source_lang"] = "en"
                    entry["target_lang"] = "en"
                    entry["pnc"] = "no"

                f.write(json.dumps(entry) + "\n")
                count += 1

            except Exception as e:
                logger.warning(f"Failed to process {item['song_id']}: {e}")
                continue

    logger.info(f"Created manifest with {count} entries ({skipped_duration} skipped for duration)")
    return count


def _chunk_synced_lines(
    lines: list[dict],
    tokenizer: WhisperTokenizer,
    max_tokens: int = 440,
    max_chunk_seconds: float = 30.0,
) -> list[dict]:
    """
    Group consecutive synced lyric lines into chunks that fit within
    Whisper's decoder token limit.

    Lines are accumulated until adding the next line would exceed
    *max_tokens* or *max_chunk_seconds*, whichever is hit first. The
    token limit is the primary constraint; the time limit is a secondary
    safeguard so that Whisper's 30-second feature extractor window is
    not wasted on silence.

    :param lines: Synced line dicts, each with ``text`` (str),
        ``start`` (ms), and ``duration`` (ms).
    :param tokenizer: A Whisper tokenizer instance used to count tokens.
    :param max_tokens: Maximum number of tokens per chunk. Defaults to
        440, leaving headroom below Whisper's 448-token decoder limit for
        special tokens.
    :param max_chunk_seconds: Secondary time cap per chunk in seconds.
    :return: List of chunk dicts, each with ``text`` (str),
        ``offset`` (seconds), and ``duration`` (seconds).
    """
    max_ms = max_chunk_seconds * 1000
    chunks = []
    current_lines = []
    current_text = ""
    current_tokens = 0
    chunk_start_ms = None

    for line in lines:
        text = line.get("text", "").strip()
        if not text:
            continue

        line_start = line["start"]
        line_end = line_start + line["duration"]

        # Tokenize the candidate text to get exact token count
        candidate_text = f"{current_text} {text}".strip() if current_text else text
        candidate_tokens = len(tokenizer(candidate_text).input_ids)

        if chunk_start_ms is None:
            chunk_start_ms = line_start

        chunk_duration_ms = line_end - chunk_start_ms
        exceeds_tokens = candidate_tokens > max_tokens
        exceeds_time = chunk_duration_ms > max_ms

        if (exceeds_tokens or exceeds_time) and current_lines:
            # Flush current chunk
            last_end = current_lines[-1]["start"] + current_lines[-1]["duration"]
            chunks.append({
                "text": current_text,
                "offset": chunk_start_ms / 1000.0,
                "duration": (last_end - chunk_start_ms) / 1000.0,
            })
            current_lines = [line]
            current_text = text
            current_tokens = len(tokenizer(text).input_ids)
            chunk_start_ms = line_start
        else:
            current_lines.append(line)
            current_text = candidate_text
            current_tokens = candidate_tokens

    # Flush remaining lines
    if current_lines and chunk_start_ms is not None:
        last_end = current_lines[-1]["start"] + current_lines[-1]["duration"]
        chunks.append({
            "text": current_text,
            "offset": chunk_start_ms / 1000.0,
            "duration": (last_end - chunk_start_ms) / 1000.0,
        })

    return chunks


def create_whisper_manifest(
    dataset: LyricsDataset,
    output_path: Path,
    model_name: str = "openai/whisper-large-v3",
    max_tokens: int = 440,
    max_chunk_seconds: float = 30.0,
) -> int:
    """
    Create a manifest JSONL for Whisper finetuning by chunking songs into
    segments that fit within Whisper's 448-token decoder limit, using
    synced line-level lyrics for audio boundaries.

    The tokenizer is loaded from *model_name* to ensure chunks are sized
    correctly for the target model. Songs without synced lyrics are
    skipped entirely.

    :param dataset: Dataset instance to iterate over.
    :param output_path: Path to write the manifest JSONL file.
    :param model_name: HuggingFace model identifier for loading the
        tokenizer (e.g. ``"openai/whisper-large-v3"``).
    :param max_tokens: Maximum tokens per chunk (default 440, leaving
        headroom for special tokens below Whisper's 448 limit).
    :param max_chunk_seconds: Secondary time cap per chunk in seconds.
    :return: Number of chunk entries written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"  Loading tokenizer: {model_name}")
    tokenizer = WhisperTokenizer.from_pretrained(model_name)

    count = 0
    skipped_no_sync = 0
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

                audio_file = song["dir"] / _RNG.choice(song["available"])
                chunks = _chunk_synced_lines(
                    synced, tokenizer, max_tokens, max_chunk_seconds,
                )

                for chunk in chunks:
                    if not chunk["text"].strip():
                        continue
                    if chunk["duration"] < 0.1:
                        continue
                    entry = {
                        "audio_filepath": str(audio_file),
                        "offset": chunk["offset"],
                        "duration": chunk["duration"],
                        "text": chunk["text"],
                    }
                    f.write(json.dumps(entry) + "\n")
                    count += 1

            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Failed to process {song['id']}: {e}")
                continue

    logger.info(
        f"Created Whisper manifest with {count} chunks "
        f"({skipped_no_sync} songs skipped for missing synced lyrics)"
    )
    return count
