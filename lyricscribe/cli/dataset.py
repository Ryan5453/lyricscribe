import logging
import re
import shutil
import zipfile
from pathlib import Path

import datasets as hf_datasets
import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from lyricscribe.schemas import Lyrics, SyncedLine, SyncedLyrics, UnsyncedLyrics

logger = logging.getLogger(__name__)

MUSDB18HQ_URL = "https://zenodo.org/records/3338373/files/musdb18hq.zip?download=1"
MUSDB18HQ_CACHE_DIR = Path("/tmp") / "lyricscribe" / "musdb18hq"
MUSDB18HQ_ZIP_NAME = "musdb18hq.zip"


def _sanitize_dirname(name: str) -> str:
    """
    Convert a song/track name into a filesystem-safe directory name.

    Replaces whitespace runs with underscores and strips characters that
    are problematic on common filesystems.

    :param name: Raw song or track name.
    :return: A sanitized string safe for use as a directory name.
    """
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^\w\-.]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def _build_lyrics(
    text: str,
    lines: list[dict],
    language: str,
    provider: str,
) -> Lyrics:
    """
    Build a :class:`Lyrics` model from common source fields.

    Converts line-level ``start``/``end`` times from seconds to
    millisecond-based ``start``/``duration``.

    :param text: Full lyrics as a single string.
    :param lines: List of dicts with ``start`` (seconds), ``end`` (seconds),
        and ``text`` keys.
    :param language: ISO 639-1 language code.
    :param provider: Provider name for attribution.
    :return: A populated :class:`Lyrics` instance.
    """
    synced_lines = []
    for line in lines:
        start_ms = round(line["start"] * 1000)
        end_ms = round(line["end"] * 1000)
        duration_ms = end_ms - start_ms
        synced_lines.append(
            SyncedLine(text=line["text"], start=start_ms, duration=duration_ms)
        )

    return Lyrics(
        unsynced=UnsyncedLyrics(data=text, provider=provider),
        synced=SyncedLyrics(data=synced_lines, provider=provider),
        detected_language=language,
        language_confidence=1.0,
    )


def _download_file(url: str, dest: Path) -> None:
    """
    Stream-download a file with a Rich progress bar.

    Writes to a ``.partial`` file first, then renames on completion
    to avoid leaving corrupt files on interrupted downloads.

    :param url: URL to download.
    :param dest: Local path to write the file to.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")

    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with (
        Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            DownloadColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
        ) as progress,
        open(partial, "wb") as f,
    ):
        task = progress.add_task("Downloading MUSDB18-HQ", total=total or None)
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            progress.update(task, advance=len(chunk))

    partial.rename(dest)


def _ensure_musdb18hq(cache_dir: Path) -> Path:
    """
    Ensure MUSDB18-HQ test tracks are available locally.

    Downloads the zip from Zenodo if necessary, extracts the ``test/``
    subdirectory, and deletes the zip to free disk space.

    :param cache_dir: Base cache directory for MUSDB18-HQ files.
    :return: Path to the extracted ``test/`` directory.
    """
    test_dir = cache_dir / "test"

    if test_dir.is_dir() and any(test_dir.iterdir()):
        logger.info(f"MUSDB18-HQ test tracks already cached at {test_dir}")
        return test_dir

    zip_path = cache_dir / MUSDB18HQ_ZIP_NAME

    if not zip_path.exists():
        logger.info(
            "Downloading MUSDB18-HQ from Zenodo (~30 GB). "
            "This is a one-time download."
        )
        _download_file(MUSDB18HQ_URL, zip_path)
        logger.info(f"Download complete: {zip_path}")
    else:
        logger.info(f"Using cached zip: {zip_path}")

    logger.info("Extracting test tracks from zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        test_members = [m for m in zf.namelist() if m.startswith("musdb18hq/test/")]
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total} files"),
        ) as progress:
            task = progress.add_task("Extracting", total=len(test_members))
            for member in test_members:
                rel = Path(member).relative_to("musdb18hq")
                target = cache_dir / rel
                if member.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                progress.update(task, advance=1)

    logger.info(f"Extracted test tracks to {test_dir}")

    zip_path.unlink()
    logger.info("Deleted zip to free disk space")

    return test_dir


def download_jam_alt(output_dir: Path) -> None:
    """
    Download the Jam-ALT dataset and convert it to per-song directories.

    Each song gets a directory containing ``audio.mp3`` and ``lyrics.json``
    in the standard schema.

    :param output_dir: Root directory for the output dataset.
    """
    logger.info("Loading Jam-ALT dataset from HuggingFace...")
    ds = hf_datasets.load_dataset("jamendolyrics/jam-alt", split="test")
    ds = ds.cast_column("audio", hf_datasets.Audio(decode=False))

    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0

    for row in ds:
        name = _sanitize_dirname(row["name"])
        song_dir = output_dir / name
        lyrics_path = song_dir / "lyrics.json"

        if lyrics_path.exists():
            skipped += 1
            continue

        song_dir.mkdir(parents=True, exist_ok=True)

        audio_info = row["audio"]
        audio_bytes = audio_info["bytes"]
        audio_path = song_dir / "audio.mp3"

        if audio_bytes:
            audio_path.write_bytes(audio_bytes)
        elif audio_info.get("path"):
            shutil.copy2(audio_info["path"], audio_path)

        lyrics = _build_lyrics(
            text=row["text"],
            lines=row["lines"],
            language=row["language"],
            provider="jam-alt",
        )
        lyrics_path.write_text(lyrics.model_dump_json(indent=2))

        success += 1
        logger.info(f"  [{success + skipped}/{len(ds)}] {name}")

    logger.info(
        f"Jam-ALT complete: {success} downloaded, {skipped} skipped (already exist)"
    )


def download_musdb_alt(output_dir: Path) -> None:
    """
    Download the MUSDB-ALT dataset and convert it to per-song directories.

    Lyrics come from HuggingFace. Audio (``mixture.wav`` and ``vocals.wav``)
    comes from MUSDB18-HQ, which is automatically downloaded from Zenodo
    and cached.

    Each song gets a directory containing ``mixture.wav``, ``vocals.wav``,
    and ``lyrics.json``.

    :param output_dir: Root directory for the output dataset.
    """
    logger.info("Loading MUSDB-ALT lyrics from HuggingFace...")
    ds = hf_datasets.load_dataset("jazasyed/musdb-alt", split="test")

    test_dir = _ensure_musdb18hq(MUSDB18HQ_CACHE_DIR)

    available_tracks = {d.name: d for d in test_dir.iterdir() if d.is_dir()}

    output_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    unmatched = []

    for row in ds:
        track_name = row["name"]
        safe_name = _sanitize_dirname(track_name)
        song_dir = output_dir / safe_name
        lyrics_path = song_dir / "lyrics.json"

        if lyrics_path.exists():
            skipped += 1
            continue

        musdb_track_dir = available_tracks.get(track_name)
        if musdb_track_dir is None:
            unmatched.append(track_name)
            logger.warning(f"No MUSDB18-HQ match for: {track_name}")
            continue

        mixture_src = musdb_track_dir / "mixture.wav"
        vocals_src = musdb_track_dir / "vocals.wav"

        if not mixture_src.exists() or not vocals_src.exists():
            logger.warning(
                f"Missing audio files for {track_name} "
                f"(mixture={mixture_src.exists()}, vocals={vocals_src.exists()})"
            )
            continue

        song_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(mixture_src, song_dir / "mixture.wav")
        shutil.copy2(vocals_src, song_dir / "vocals.wav")

        lyrics = _build_lyrics(
            text=row["text"],
            lines=row["lines"],
            language="en",
            provider="musdb-alt",
        )
        lyrics_path.write_text(lyrics.model_dump_json(indent=2))

        success += 1
        logger.info(f"  [{success + skipped}/{len(ds)}] {track_name}")

    logger.info(
        f"MUSDB-ALT complete: {success} downloaded, {skipped} skipped (already exist)"
    )
    if unmatched:
        logger.warning(
            f"{len(unmatched)} tracks had no MUSDB18-HQ match: "
            + ", ".join(unmatched)
        )
