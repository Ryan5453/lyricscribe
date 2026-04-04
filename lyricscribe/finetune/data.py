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


def create_nemo_manifest(
    dataset: LyricsDataset,
    output_path: Path,
    max_duration: float = 300.0,
) -> int:
    """
    Create NeMo manifest JSONL file from dataset.
    
    :param dataset: LyricsDataset instance
    :param output_path: Path to write manifest
    :param max_duration: Maximum duration in seconds
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
