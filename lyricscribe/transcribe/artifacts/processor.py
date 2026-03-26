import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import torchaudio

logger = logging.getLogger(__name__)

_SILENCE_LABELS = {"", "sp", "sil", "<eps>"}

DEFAULT_MFA_DOCKER_REF = "docker://mmcauliffe/montreal-forced-aligner:latest"

_DICTIONARY = "english_mfa"
_ACOUSTIC_MODEL = "english_mfa"
_WORK_MOUNT = "/mfa_work"
_MFA_ROOT_MOUNT = "/mfa_root"
_SEGMENT_PADDING_S = 0.15
_MIN_SEGMENT_S = 0.2


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
    musdb_dir: Path,
    output_dir: Path,
    *,
    container: str | Path | None = None,
    mfa_root: Path | None = None,
) -> None:
    """
    Run Montreal Forced Aligner inside a Singularity/Apptainer container and
    write per-song alignment JSON files.

    Prepares a temporary MFA corpus directory (wav + lab files), runs
    ``mfa align`` in the container with ``--output_format json``, then converts
    MFA's output into the per-song format expected by
    :func:`~lyricscribe.transcribe.artifacts.correlation._load_alignments`.

    :param musdb_dir: root MUSDB directory containing one subdirectory per song,
        each with ``vocals.wav`` and ``lyrics.json``.
    :param output_dir: directory to write one ``.json`` alignment file per song.
    :param container: path to a ``.sif`` image or ``docker://`` URI, or ``None``
        to read from ``LYRICSCRIBE_MFA_CONTAINER``.
    :param mfa_root: host directory bound as MFA's model cache. If omitted, a
        temp directory is used (models re-downloaded each run).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

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

    runtime = _find_container_runtime()

    song_dirs = sorted([d for d in musdb_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(song_dirs)} songs in {musdb_dir}")

    mfa_root_temp: tempfile.TemporaryDirectory[str] | None = None
    if mfa_root is None:
        mfa_root_temp = tempfile.TemporaryDirectory(prefix="lyricscribe_mfa_root_")
        mfa_host = Path(mfa_root_temp.name)
    else:
        mfa_host = mfa_root.expanduser().resolve()
        mfa_host.mkdir(parents=True, exist_ok=True)

    try:
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
                vocals_path = song_dir / "vocals.wav"
                lyrics_path = song_dir / "lyrics.json"

                if not vocals_path.exists() or not lyrics_path.exists():
                    logger.warning(
                        f"Missing vocals.wav or lyrics.json in {song_dir.name}"
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
                "Prepared %s songs and %s synced lyric segments for MFA alignment",
                prepared,
                prepared_segments,
            )

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
                _DICTIONARY,
                _ACOUSTIC_MODEL,
                f"{_WORK_MOUNT}/aligned",
                "--output_format",
                "json",
                "--clean",
            ]

            logger.info(f"Running: {runtime} exec ... mfa align ...")
            result = subprocess.run(
                cmd,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"MFA align failed with exit code {result.returncode}. "
                    "Check that english_mfa acoustic model and dictionary are "
                    "downloaded in your --mfa-root directory."
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

            success = 0
            for song_id, words in sorted(song_words.items()):
                words.sort(key=lambda x: (x["start"], x["end"]))
                out_path = output_dir / f"{song_id}.json"
                out_path.write_text(
                    json.dumps(
                        {
                            "song_id": song_id,
                            "words": words,
                        },
                        indent=2,
                    )
                )
                success += 1

            logger.info(f"Wrote alignments for {success} songs to {output_dir}")
    finally:
        if mfa_root_temp is not None:
            mfa_root_temp.cleanup()
