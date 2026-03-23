import json
import logging
import re
import shutil
import tempfile
from pathlib import Path

from montreal_forced_aligner.alignment import PretrainedAligner

logger = logging.getLogger(__name__)

_SILENCE_LABELS = {"", "sp", "sil", "<eps>"}


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


def align(
    musdb_dir: Path,
    output_dir: Path,
    *,
    dictionary: str = "english_mfa",
    acoustic_model: str = "english_mfa",
) -> None:
    """
    Run Montreal Forced Aligner on a MUSDB dataset and write per-song
    alignment JSON files.

    Prepares a temporary MFA corpus directory (wav + lab files), runs
    alignment using :class:`PretrainedAligner`, exports to JSON, then
    converts MFA's output into the per-song format expected by
    :func:`~lyricscribe.transcribe.artifacts.correlation._load_alignments`.

    :param musdb_dir: root MUSDB directory containing one subdirectory
        per song, each with ``vocals.wav`` and ``lyrics.json``.
    :param output_dir: directory to write one ``.json`` alignment file
        per song.
    :param dictionary: MFA dictionary name or path (default ``english_mfa``).
    :param acoustic_model: MFA acoustic model name or path
        (default ``english_mfa``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    song_dirs = sorted([d for d in musdb_dir.iterdir() if d.is_dir()])
    logger.info(f"Found {len(song_dirs)} songs in {musdb_dir}")

    with tempfile.TemporaryDirectory(prefix="lyricscribe_mfa_") as tmp:
        corpus_dir = Path(tmp) / "corpus"
        mfa_output = Path(tmp) / "aligned"
        corpus_dir.mkdir()
        mfa_output.mkdir()

        prepared = 0
        for song_dir in song_dirs:
            vocals_path = song_dir / "vocals.wav"
            lyrics_path = song_dir / "lyrics.json"

            if not vocals_path.exists() or not lyrics_path.exists():
                logger.warning(f"Missing vocals.wav or lyrics.json in {song_dir.name}")
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

        aligner = PretrainedAligner(
            corpus_directory=str(corpus_dir),
            dictionary_path=dictionary,
            acoustic_model_path=acoustic_model,
            output_directory=str(mfa_output),
            clean=True,
            quiet=True,
        )
        aligner.setup()
        aligner.align()
        aligner.export_files(
            output_directory=mfa_output,
            output_format="json",
        )

        success = 0
        for json_path in sorted(mfa_output.glob("**/*.json")):
            song_id = json_path.stem
            words = _parse_mfa_json(json_path)
            out_path = output_dir / f"{song_id}.json"
            out_path.write_text(json.dumps({
                "song_id": song_id,
                "words": words,
            }, indent=2))
            success += 1

        logger.info(f"Wrote alignments for {success} songs to {output_dir}")
