import json
import logging
import random
from pathlib import Path
from typing import Iterator

import soundfile as sf

logger = logging.getLogger(__name__)

_RNG = random.Random(42)


class LyricsDataset:
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
        return len(self.songs)

    def __iter__(self) -> Iterator[dict]:
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
    max_duration: float = 300.0,
) -> int:
    """
    Create manifest JSONL file from dataset. One entry per song.

    :param dataset: LyricsDataset instance
    :param output_path: Path to write manifest
    :param max_duration: Maximum duration in seconds (songs longer are skipped)
    :return: Number of entries written
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
                info = sf.info(item["audio_path"])
                duration = info.duration

                if duration > max_duration:
                    skipped_duration += 1
                    continue

                entry = {
                    "audio_filepath": item["audio_path"],
                    "duration": duration,
                    "text": item["transcript"],
                }
                f.write(json.dumps(entry) + "\n")
                count += 1

            except Exception as e:
                logger.warning(f"Failed to process {item['song_id']}: {e}")
                continue

    logger.info(f"Created manifest with {count} entries ({skipped_duration} skipped for duration)")
    return count


def _chunk_synced_lines(lines: list[dict], max_chunk_seconds: float = 30.0) -> list[dict]:
    """
    Group consecutive synced lyric lines into chunks of ~max_chunk_seconds.

    :param lines: List of synced line dicts with 'text', 'start' (ms), 'duration' (ms)
    :param max_chunk_seconds: Target max chunk duration in seconds
    :return: List of chunk dicts with 'text', 'offset' (seconds), 'duration' (seconds)
    """
    max_ms = max_chunk_seconds * 1000
    chunks = []
    current_lines = []
    chunk_start_ms = None

    for line in lines:
        if not line.get("text", "").strip():
            continue

        line_start = line["start"]
        line_end = line_start + line["duration"]

        if chunk_start_ms is None:
            chunk_start_ms = line_start

        chunk_duration_ms = line_end - chunk_start_ms

        # If adding this line would exceed the limit and we already have lines, flush
        if chunk_duration_ms > max_ms and current_lines:
            last_end = current_lines[-1]["start"] + current_lines[-1]["duration"]
            chunks.append({
                "text": " ".join(l["text"].strip() for l in current_lines),
                "offset": chunk_start_ms / 1000.0,
                "duration": (last_end - chunk_start_ms) / 1000.0,
            })
            current_lines = [line]
            chunk_start_ms = line_start
        else:
            current_lines.append(line)

    # Flush remaining lines
    if current_lines and chunk_start_ms is not None:
        last_end = current_lines[-1]["start"] + current_lines[-1]["duration"]
        chunks.append({
            "text": " ".join(l["text"].strip() for l in current_lines),
            "offset": chunk_start_ms / 1000.0,
            "duration": (last_end - chunk_start_ms) / 1000.0,
        })

    return chunks


def create_whisper_manifest(
    dataset: LyricsDataset,
    output_path: Path,
    max_chunk_seconds: float = 30.0,
) -> int:
    """
    Create manifest JSONL for Whisper finetuning, chunking songs into ~30s segments
    using synced line-level lyrics.

    Songs without synced lyrics are skipped.

    :param dataset: LyricsDataset instance
    :param output_path: Path to write manifest
    :param max_chunk_seconds: Target max chunk duration
    :return: Number of entries written
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
                chunks = _chunk_synced_lines(synced, max_chunk_seconds)

                for chunk in chunks:
                    if not chunk["text"].strip():
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
