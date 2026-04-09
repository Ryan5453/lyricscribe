import json
import logging
import random
from pathlib import Path
from typing import Iterator

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


def _chunk_synced_lines(
    lines: list[dict],
    max_chunk_seconds: float = 30.0,
    tokenizer: WhisperTokenizer | None = None,
    max_tokens: int = 440,
) -> list[dict]:
    """
    Group consecutive synced lyric lines into chunks bounded by time and,
    if a tokenizer is provided, by a maximum token count.

    Lines are accumulated until adding the next line would exceed
    *max_chunk_seconds* (primary bound) or, when a tokenizer is given,
    *max_tokens* (Whisper decoder safety net). The time bound keeps
    chunks short enough to fit within Whisper's 30-second feature
    extractor window and small enough to train Parakeet/Canary without
    running out of GPU memory on long songs.

    :param lines: Synced line dicts, each with ``text`` (str),
        ``start`` (ms), and ``duration`` (ms).
    :param max_chunk_seconds: Maximum chunk duration in seconds.
    :param tokenizer: Optional Whisper tokenizer for enforcing the
        decoder token cap. Pass ``None`` for architectures that don't
        have a fixed decoder length (e.g. Parakeet, Canary).
    :param max_tokens: Maximum tokens per chunk when *tokenizer* is
        provided. Defaults to 440, leaving headroom below Whisper's
        448-token decoder limit for special tokens.
    :return: List of chunk dicts, each with ``text`` (str),
        ``offset`` (seconds), and ``duration`` (seconds).
    """
    max_ms = max_chunk_seconds * 1000
    chunks = []
    current_lines = []
    current_text = ""
    chunk_start_ms = None

    for line in lines:
        text = line.get("text", "").strip()
        if not text:
            continue

        # Skip individual synced lines longer than the chunk cap. These are
        # almost always data anomalies (e.g. a provider returning the full
        # song as a single "line"). Letting them through produces oversized
        # chunks that OOM Parakeet/Canary at training time.
        if line["duration"] > max_ms:
            continue

        line_start = line["start"]
        line_end = line_start + line["duration"]

        candidate_text = f"{current_text} {text}".strip() if current_text else text

        if chunk_start_ms is None:
            chunk_start_ms = line_start

        chunk_duration_ms = line_end - chunk_start_ms
        exceeds_time = chunk_duration_ms > max_ms
        exceeds_tokens = False
        if tokenizer is not None:
            candidate_tokens = len(tokenizer(candidate_text).input_ids)
            exceeds_tokens = candidate_tokens > max_tokens

        if (exceeds_time or exceeds_tokens) and current_lines:
            last_end = current_lines[-1]["start"] + current_lines[-1]["duration"]
            chunks.append({
                "text": current_text,
                "offset": chunk_start_ms / 1000.0,
                "duration": (last_end - chunk_start_ms) / 1000.0,
            })
            current_lines = [line]
            current_text = text
            chunk_start_ms = line_start
        else:
            current_lines.append(line)
            current_text = candidate_text

    if current_lines and chunk_start_ms is not None:
        last_end = current_lines[-1]["start"] + current_lines[-1]["duration"]
        chunks.append({
            "text": current_text,
            "offset": chunk_start_ms / 1000.0,
            "duration": (last_end - chunk_start_ms) / 1000.0,
        })

    return chunks


def create_manifest(
    dataset: LyricsDataset,
    output_path: Path,
    architecture: str = "parakeet",
    model_name: str | None = None,
    max_chunk_seconds: float = 30.0,
    max_tokens: int = 440,
) -> int:
    """
    Create a manifest JSONL by chunking songs into segments aligned to
    synced lyric line boundaries. Songs without synced lyrics are
    skipped.

    All architectures share the same chunking strategy so that Whisper,
    Parakeet, and Canary train on comparable inputs: without this,
    NeMo's dataloader loads whole songs and Parakeet hits OOM on long
    audio. Chunks are bounded by *max_chunk_seconds* (primary memory /
    feature-window constraint). Whisper additionally enforces
    *max_tokens* as a safety net for its 448-token decoder cap.

    Each entry includes the per-song ``language`` from
    ``lyrics.json["detected_language"]`` so that the trainer can set the
    correct decoder prefix tokens per sample.

    For Canary, each entry includes ``source_lang``, ``target_lang``,
    ``pnc``, and ``answer`` fields required by the Lhotse prompt-based
    data pipeline. ``source_lang``/``target_lang`` are populated from the
    detected language so multilingual training works correctly.

    :param dataset: Dataset instance providing song directories.
    :param output_path: Path to write the manifest JSONL file.
    :param architecture: Model architecture name (``"whisper"``,
        ``"parakeet"``, or ``"canary"``).
    :param model_name: HuggingFace model identifier, required for
        ``"whisper"`` to load the tokenizer. Ignored otherwise.
    :param max_chunk_seconds: Maximum chunk duration in seconds.
    :param max_tokens: Whisper decoder token cap (applied only when
        ``architecture == "whisper"``).
    :return: Number of chunk entries written.
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

                detected_language = lyrics_data["detected_language"]

                audio_file = song["dir"] / _RNG.choice(song["available"])
                chunks = _chunk_synced_lines(
                    synced,
                    max_chunk_seconds=max_chunk_seconds,
                    tokenizer=tokenizer,
                    max_tokens=max_tokens,
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
                        "language": detected_language,
                    }
                    if architecture == "canary":
                        entry["answer"] = chunk["text"]
                        entry["source_lang"] = detected_language
                        entry["target_lang"] = detected_language
                        entry["pnc"] = "no"
                    f.write(json.dumps(entry) + "\n")
                    count += 1

            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"Failed to process {song['id']}: {e}")
                continue

    logger.info(
        f"Created manifest with {count} chunks "
        f"({skipped_no_sync} songs skipped for missing synced lyrics)"
    )
    return count
