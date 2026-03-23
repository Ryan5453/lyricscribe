import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_SILENCE_LABELS = {"", "sp", "sil", "<eps>"}

DEFAULT_MFA_DOCKER_REF = "docker://mmcauliffe/montreal-forced-aligner:latest"

_DICTIONARY = "english_mfa"
_ACOUSTIC_MODEL = "english_mfa"
_WORK_MOUNT = "/mfa_work"
_MFA_ROOT_MOUNT = "/mfa_root"


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
    for tier in data.get("tiers", []):
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

                text = lyrics_data.get("unsynced", {}).get("data", "")
                if not text:
                    logger.warning(f"Empty lyrics for {song_dir.name}")
                    continue

                name = song_dir.name
                shutil.copy2(vocals_path, corpus_dir / f"{name}.wav")
                (corpus_dir / f"{name}.lab").write_text(_clean_lyrics(text))
                prepared += 1

            logger.info(f"Prepared {prepared} songs for MFA alignment")

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
                "--quiet",
            ]

            logger.info(f"Running: {runtime} exec ... mfa align ...")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                logger.error("MFA stderr:\n%s", result.stderr)
                logger.error("MFA stdout:\n%s", result.stdout)
                raise RuntimeError(
                    f"MFA align failed with exit code {result.returncode}. "
                    "Check that english_mfa acoustic model and dictionary are "
                    "downloaded in your --mfa-root directory."
                )

            success = 0
            for json_path in sorted(mfa_output.glob("**/*.json")):
                song_id = json_path.stem
                words = _parse_mfa_json(json_path)
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
