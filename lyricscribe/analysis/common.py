"""
Shared loaders for job dirs and dataset references.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import jiwer

from lyricscribe.evaluate import _language_for, _load_dataset, _normalized

# Keyed by dataset directory name; jobs with fewer songs are refused as partial.
EXPECTED_SONGS = {"jam-alt": 79, "jam_alt": 79, "musdb_alt": 39}


@dataclass
class Job:
    """
    One transcription job dir: its config, raw hypotheses and references.

    :ivar path: Job directory.
    :ivar config: Parsed ``config.json``.
    :ivar hyps: Raw hypothesis text per song (``None`` = transcription failed).
    :ivar corrupt_lines: Unparseable ``results*.jsonl`` lines (e.g. from
        concurrent writers).
    :ivar refs: Normalised unsynced reference per song.
    :ivar langs: Language code per song.
    """

    path: Path
    config: dict
    hyps: dict[str, str | None]
    corrupt_lines: int
    refs: dict[str, str] = field(default_factory=dict)
    langs: dict[str, str] = field(default_factory=dict)

    @property
    def dataset_dirs(self) -> list[Path]:
        return [Path(d) for d in self.config.get("directories", [])]

    @property
    def expected(self) -> int | None:
        """Song count this job should have, or ``None`` if unknown."""
        for d in self.dataset_dirs:
            if d.name in EXPECTED_SONGS:
                return EXPECTED_SONGS[d.name]
        total = self.config.get("total_files")
        return int(total) if total else None

    def normalized_hyp(self, song_id: str) -> str:
        return _normalized(self.hyps.get(song_id) or "", self.langs[song_id])


def load_job(job_dir: Path) -> Job:
    """
    Load a job dir's config, hypotheses and (cached) normalised references.

    :param job_dir: Directory containing ``config.json`` and ``results*.jsonl``.
    :return: The loaded :class:`Job`.
    :raises FileNotFoundError: If ``config.json`` is missing.
    """
    job_dir = Path(job_dir)
    config_path = job_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{job_dir}: no config.json (wrong path?)")
    config = json.loads(config_path.read_text())

    hyps: dict[str, str | None] = {}
    corrupt = 0
    for results_path in sorted(job_dir.glob("results*.jsonl")):
        for line in results_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                corrupt += 1
                continue
            hyps[row["song_id"]] = row.get("transcription")

    job = Job(path=job_dir, config=config, hyps=hyps, corrupt_lines=corrupt)
    for d in job.dataset_dirs:
        cache = _load_dataset(d)
        job.refs.update(cache["refs_normalized"])
        job.langs.update(cache["languages"])
    return job


def load_selfwer_keep(selfwer_file: Path, max_self_wer: float) -> set[str]:
    """
    Songs whose reference self-WER is within the threshold.

    Songs with no self-WER (missing synced lyrics, or an error row) are
    excluded: their reference quality cannot be checked.

    :param selfwer_file: Artifact from ``lyricscribe analyze selfwer``.
    :param max_self_wer: Inclusive cutoff, e.g. 0.3.
    :return: Song IDs to keep.
    """
    keep = set()
    for line in Path(selfwer_file).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            sw = row.get("self_wer")
            if sw is not None and sw <= max_self_wer:
                keep.add(row["song_id"])
    return keep


def corpus_stats(refs: list[str], hyps: list[str]) -> dict:
    """
    Corpus-level WER and I/D/S counts over aligned normalised strings.

    :param refs: Normalised references.
    :param hyps: Normalised hypotheses, same order.
    :return: Dict with ``n``, ``wer``, ``I``/``D``/``S``/``H`` and the I/D/S
        ``share`` of errors in percent.
    """
    m = jiwer.process_words(refs, hyps)
    ref_len = m.substitutions + m.deletions + m.hits
    errors = m.substitutions + m.deletions + m.insertions
    tot = max(errors, 1)
    return {
        "n": len(refs),
        "wer": errors / ref_len if ref_len else 0.0,
        "I": m.insertions,
        "D": m.deletions,
        "S": m.substitutions,
        "H": m.hits,
        "share": {
            "I": 100 * m.insertions / tot,
            "D": 100 * m.deletions / tot,
            "S": 100 * m.substitutions / tot,
        },
    }


def per_song_counts(refs: list[str], hyps: list[str]) -> tuple[list[int], list[int]]:
    """
    Per-song error and reference-word counts.

    :param refs: Normalised references.
    :param hyps: Normalised hypotheses, same order.
    :return: ``(errors, ref_words)`` lists.
    """
    errors, words = [], []
    for r, h in zip(refs, hyps):
        m = jiwer.process_words([r], [h])
        errors.append(m.substitutions + m.deletions + m.insertions)
        words.append(m.substitutions + m.deletions + m.hits)
    return errors, words


def raw_references(dataset_dir: Path) -> dict[str, tuple[str, str, str]]:
    """
    Un-normalised reference texts per song.

    :param dataset_dir: Dataset directory with per-song ``lyrics.json``.
    :return: ``{song_id: (language, unsynced_text, synced_text)}``; the synced
        text is the line texts joined with spaces ("" if absent).
    """
    out = {}
    for song_dir in Path(dataset_dir).iterdir():
        path = song_dir / "lyrics.json"
        if not song_dir.is_dir() or not path.exists():
            continue
        data = json.loads(path.read_text())
        unsynced = (data.get("unsynced") or {}).get("data")
        synced = (data.get("synced") or {}).get("data") or []
        synced_text = (
            " ".join(x.get("text", "") for x in synced)
            if isinstance(synced, list)
            else str(synced)
        )
        out[song_dir.name] = (
            _language_for(data),
            unsynced if isinstance(unsynced, str) else "",
            synced_text,
        )
    return out
