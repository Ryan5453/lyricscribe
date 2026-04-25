import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import torchaudio

from lyricscribe.schemas import AlignedWord, Alignment, Lyrics

logger = logging.getLogger(__name__)

_SILENCE_LABELS = {"", "sp", "sil", "<eps>"}

DEFAULT_MFA_DOCKER_REF = "docker://mmcauliffe/montreal-forced-aligner:latest"

_MFA_MODEL_BY_LANGUAGE = {
    "en": "english_mfa",
    "es": "spanish_mfa",
}
_WORK_MOUNT = "/mfa_work"
_MFA_ROOT_MOUNT = "/mfa_root"
_SEGMENT_PADDING_S = 0.15
_MIN_SEGMENT_S = 0.2


def _language_key(lyrics_data: dict) -> str:
    lang = (lyrics_data.get("detected_language") or "en").lower()
    return lang.split("-")[0].split("_")[0]


def _clean_lyrics(text: str) -> str:
    """Strip punctuation and normalise whitespace for MFA."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_mfa_json(json_path: Path) -> list[dict]:
    """
    Extract word-level timestamps from a single MFA JSON alignment file.

    :param json_path: path to a ``.json`` file produced by MFA with
        ``output_format="json"``.
    :returns: list of dicts each containing ``word``, ``start``, and ``end``.
    """
    with open(json_path) as f:
        data = json.load(f)

    words: list[dict] = []
    tiers = data.get("tiers", [])

    # MFA JSON has appeared in two shapes:
    # 1) a list of tier objects with {"name": ..., "entries": ...}
    # 2) a dict keyed by tier name, e.g. {"words": {"entries": ...}, ...}
    if isinstance(tiers, dict):
        word_tier = tiers.get("words", {})
        entries = word_tier.get("entries", [])
        for entry in entries:
            start, end, label = entry[0], entry[1], entry[2]
            if label not in _SILENCE_LABELS:
                words.append({"word": label, "start": start, "end": end})
        return words

    for tier in tiers:
        if tier.get("name") != "words":
            continue
        for entry in tier.get("entries", []):
            start, end, label = entry[0], entry[1], entry[2]
            if label not in _SILENCE_LABELS:
                words.append({"word": label, "start": start, "end": end})
    return words


def _find_container_runtime() -> str:
    """Prefer Apptainer (common on HPC), then Singularity."""
    for cmd in ("apptainer", "singularity"):
        if shutil.which(cmd):
            return cmd
    raise RuntimeError(
        "Neither 'apptainer' nor 'singularity' was found in PATH. "
        "Load the Singularity/Apptainer module on your cluster, or install Singularity."
    )


def _iter_synced_segments(lyrics_data: dict) -> list[dict]:
    """
    Return cleaned line-level segments from lyrics.json synced entries.

    Expected input shape is ``lyrics_data["synced"]["data"]`` where each item
    has ``text``, ``start``, and ``duration`` in milliseconds.
    """
    synced = lyrics_data.get("synced", {}).get("data", [])
    if not isinstance(synced, list):
        return []

    segments: list[dict] = []
    for i, entry in enumerate(synced):
        if not isinstance(entry, dict):
            continue
        text = _clean_lyrics(str(entry.get("text", "")))
        if not text:
            continue

        try:
            start_ms = float(entry.get("start", 0))
            duration_ms = float(entry.get("duration", 0))
        except (TypeError, ValueError):
            continue

        if duration_ms <= 0:
            continue

        start_s = start_ms / 1000.0
        end_s = (start_ms + duration_ms) / 1000.0
        if end_s - start_s < _MIN_SEGMENT_S:
            continue

        segments.append(
            {
                "index": i,
                "text": text,
                "start_s": start_s,
                "end_s": end_s,
            }
        )
    return segments


def align(
    dataset_dir: Path,
    *,
    container: str | Path | None = None,
    mfa_root: Path | None = None,
    filename: str = "vocals.wav",
    num_chunks: int = 1,
    chunk_id: int = 0,
    skip_existing: bool = True,
) -> None:
    """
    Run Montreal Forced Aligner inside a Singularity/Apptainer container and
    write word-level alignments back into each song's ``lyrics.json``.

    Prepares a temporary MFA corpus directory (wav + lab files), runs
    ``mfa align`` in the container with ``--output_format json``, then mutates
    each song's ``lyrics.json`` in place to populate the
    :attr:`lyricscribe.schemas.Lyrics.alignment` field.

    :param dataset_dir: root dataset directory containing one subdirectory per
        song, each with an audio file (see ``filename``) and ``lyrics.json``.
    :param container: path to a ``.sif`` image or ``docker://`` URI, or ``None``
        to read from ``LYRICSCRIBE_MFA_CONTAINER``.
    :param mfa_root: host directory bound as MFA's model cache. If omitted, a
        temp directory is used (models re-downloaded each run).
    :param filename: audio filename inside each song subdirectory (e.g.
        ``"vocals.wav"`` or ``"htdemucs_ft_vocals.wav"``).
    :param num_chunks: total number of shards this dataset is partitioned into.
        When >1, only the subset matching ``chunk_id`` is processed. Used by
        the SLURM array wrapper to parallelize large datasets across nodes.
    :param chunk_id: 0-indexed shard id to process (requires ``num_chunks > 1``).
    :param skip_existing: if True, songs whose ``lyrics.json`` already has a
        non-null ``alignment`` field are skipped — makes resume-on-failure
        cheap and lets the same command run across partial datasets.
    """
    raw = container
    if raw is None:
        raw = os.environ.get("LYRICSCRIBE_MFA_CONTAINER")
    if not raw:
        raise ValueError(
            "No container image specified. Pass --container /path/to/mfa.sif "
            "or set LYRICSCRIBE_MFA_CONTAINER. "
            f"Example: singularity pull mfa.sif {DEFAULT_MFA_DOCKER_REF}"
        )
    raw_s = str(raw).strip()
    if raw_s.startswith("docker://"):
        image_arg = raw_s
    else:
        image_path = Path(raw_s).expanduser().resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"MFA container image not found: {image_path}")
        image_arg = str(image_path)

    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
    if chunk_id < 0 or chunk_id >= num_chunks:
        raise ValueError(
            f"chunk_id must be in [0, {num_chunks - 1}] for num_chunks={num_chunks}, got {chunk_id}"
        )

    runtime = _find_container_runtime()

    song_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    total_found = len(song_dirs)

    # Deterministic chunk partition: take every ``num_chunks``-th song starting
    # at ``chunk_id``. Deterministic so reruns hit the same songs, cheap to
    # compute, and naturally balanced for a uniformly-sized dataset.
    if num_chunks > 1:
        song_dirs = song_dirs[chunk_id::num_chunks]

    # Group songs by the MFA model their detected_language dictates.
    # Songs whose existing alignment already used the correct model are
    # dropped when skip_existing=True; this lets reruns be safe across
    # languages without re-aligning the already-correct ones.
    songs_by_model: dict[str, list[Path]] = {}
    unmapped: list[str] = []
    for song_dir in song_dirs:
        lyrics_path = song_dir / "lyrics.json"
        if not lyrics_path.exists():
            continue
        try:
            data = json.loads(lyrics_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        lang_key = _language_key(data)
        mfa_model = _MFA_MODEL_BY_LANGUAGE.get(lang_key)
        if mfa_model is None:
            unmapped.append(song_dir.name)
            continue
        if skip_existing:
            existing = data.get("alignment")
            if existing is not None and existing.get("mfa_model") == mfa_model:
                continue
        songs_by_model.setdefault(mfa_model, []).append(song_dir)

    if unmapped:
        logger.warning(
            "Skipping %s songs with unsupported detected_language "
            "(no MFA model mapped)",
            len(unmapped),
        )

    total_to_align = sum(len(v) for v in songs_by_model.values())
    logger.info(
        f"Found {total_found} songs in {dataset_dir}; "
        f"aligning {total_to_align} across {len(songs_by_model)} language group(s) "
        f"(chunk {chunk_id}/{num_chunks}, skip_existing={skip_existing})"
    )
    if not songs_by_model:
        logger.info("Nothing to align — exiting.")
        return

    mfa_root_temp: tempfile.TemporaryDirectory[str] | None = None
    if mfa_root is None:
        mfa_root_temp = tempfile.TemporaryDirectory(prefix="lyricscribe_mfa_root_")
        mfa_host = Path(mfa_root_temp.name)
    else:
        mfa_host = mfa_root.expanduser().resolve()
        mfa_host.mkdir(parents=True, exist_ok=True)

    try:
        for mfa_model, group_song_dirs in songs_by_model.items():
            logger.info(
                "=== MFA pass: %s (%s songs) ===", mfa_model, len(group_song_dirs)
            )
            _run_mfa_pass(
                song_dirs=group_song_dirs,
                mfa_model=mfa_model,
                filename=filename,
                mfa_host=mfa_host,
                image_arg=image_arg,
                runtime=runtime,
            )
    finally:
        if mfa_root_temp is not None:
            mfa_root_temp.cleanup()


def _run_mfa_pass(
    *,
    song_dirs: list[Path],
    mfa_model: str,
    filename: str,
    mfa_host: Path,
    image_arg: str,
    runtime: str,
) -> None:
    """Build an MFA corpus from *song_dirs*, run alignment with *mfa_model*,
    then write per-song alignments back into each song's ``lyrics.json``."""

    with tempfile.TemporaryDirectory(prefix="lyricscribe_mfa_") as tmp:
        tmp_path = Path(tmp)
        corpus_dir = tmp_path / "corpus"
        mfa_output = tmp_path / "aligned"
        corpus_dir.mkdir()
        mfa_output.mkdir()

        prepared = 0
        prepared_segments = 0
        segment_index: dict[str, dict] = {}
        for song_dir in song_dirs:
            vocals_path = song_dir / filename
            lyrics_path = song_dir / "lyrics.json"

            if not vocals_path.exists() or not lyrics_path.exists():
                logger.warning(
                    f"Missing {filename} or lyrics.json in {song_dir.name}"
                )
                continue

            with open(lyrics_path) as f:
                lyrics_data = json.load(f)

            segments = _iter_synced_segments(lyrics_data)
            if not segments:
                logger.warning(f"No usable synced lyric segments for {song_dir.name}")
                continue

            name = song_dir.name
            waveform, sample_rate = torchaudio.load(vocals_path)
            num_frames = waveform.shape[1]

            for segment in segments:
                start_s = max(0.0, segment["start_s"] - _SEGMENT_PADDING_S)
                end_s = min(
                    num_frames / sample_rate,
                    segment["end_s"] + _SEGMENT_PADDING_S,
                )
                if end_s - start_s < _MIN_SEGMENT_S:
                    continue

                start_frame = max(0, int(start_s * sample_rate))
                end_frame = min(num_frames, int(end_s * sample_rate))
                if end_frame <= start_frame:
                    continue

                utterance_id = f"{name}__seg{segment['index']:04d}"
                segment_waveform = waveform[:, start_frame:end_frame]
                torchaudio.save(
                    str(corpus_dir / f"{utterance_id}.wav"),
                    segment_waveform,
                    sample_rate,
                )
                (corpus_dir / f"{utterance_id}.lab").write_text(segment["text"])
                segment_index[utterance_id] = {
                    "song_id": name,
                    "offset_s": start_s,
                }
                prepared_segments += 1

            prepared += 1

        logger.info(
            "Prepared %s songs and %s synced lyric segments for %s",
            prepared,
            prepared_segments,
            mfa_model,
        )

        if prepared == 0:
            return

        cmd: list[str] = [
            runtime,
            "exec",
            "--env",
            f"MFA_ROOT_DIR={_MFA_ROOT_MOUNT}",
            "-B",
            f"{tmp_path.resolve()}:{_WORK_MOUNT}",
            "-B",
            f"{mfa_host.resolve()}:{_MFA_ROOT_MOUNT}",
            image_arg,
            "mfa",
            "align",
            f"{_WORK_MOUNT}/corpus",
            mfa_model,
            mfa_model,
            f"{_WORK_MOUNT}/aligned",
            "--output_format",
            "json",
            "--clean",
            "--temporary_directory",
            f"{_WORK_MOUNT}/mfa_tmp",
        ]

        logger.info(f"Running: {runtime} exec ... mfa align ({mfa_model}) ...")
        result = subprocess.run(cmd, text=True, check=False)
        if result.returncode != 0:
            logger.warning(
                f"MFA align ({mfa_model}) exited {result.returncode}; "
                "some songs may not be aligned. Re-run with --skip-existing to "
                "retry only the gaps."
            )

        song_words: dict[str, list[dict]] = {}
        for json_path in sorted(mfa_output.glob("**/*.json")):
            utterance_id = json_path.stem
            if utterance_id not in segment_index:
                logger.warning("Skipping unexpected MFA output: %s", utterance_id)
                continue

            words = _parse_mfa_json(json_path)
            mapping = segment_index[utterance_id]
            song_id = mapping["song_id"]
            offset_s = mapping["offset_s"]
            song_words.setdefault(song_id, []).extend(
                {
                    "word": word["word"],
                    "start": word["start"] + offset_s,
                    "end": word["end"] + offset_s,
                }
                for word in words
            )

        generated_at = datetime.now(timezone.utc)
        song_dirs_by_name = {d.name: d for d in song_dirs}
        success = 0
        for song_id, words in sorted(song_words.items()):
            words.sort(key=lambda x: (x["start"], x["end"]))
            aligned_words = [
                AlignedWord(
                    word=w["word"],
                    start=round(w["start"] * 1000),
                    duration=round((w["end"] - w["start"]) * 1000),
                )
                for w in words
            ]
            alignment = Alignment(
                words=aligned_words,
                source_audio=filename,
                mfa_model=mfa_model,
                generated_at=generated_at,
            )

            song_dir = song_dirs_by_name.get(song_id)
            if song_dir is None:
                logger.warning(
                    "MFA output for %s has no matching song dir (post-chunk?)",
                    song_id,
                )
                continue

            lyrics_path = song_dir / "lyrics.json"
            lyrics = Lyrics.model_validate_json(lyrics_path.read_text())
            lyrics.alignment = alignment
            lyrics_path.write_text(lyrics.model_dump_json(indent=2))
            success += 1

        logger.info(
            "Wrote %s alignments (%s) back into lyrics.json files",
            success,
            mfa_model,
        )
