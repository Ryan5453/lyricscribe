"""
Audio-domain assignment for multi-file finetune manifests.

``create_manifest`` picks a random available ``--filename`` per song, so a
multi-file experiment ends up with an uncontrolled mix of audio domains. The
``"balanced"`` mode instead assigns every song *wholly* to one domain and
balances the two domains 50/50 by window count; ``"fallback"`` puts every
song in domain B. Both are a post-pass over the finished manifests: offsets,
labels and filtering are unchanged and only ``audio_filepath`` is swapped,
which requires every audio file of a song to be a time-aligned full-song
render (e.g. separator outputs and the mix).

Domains are defined by the experiment's ``--filename`` list: the first
filename is domain A (e.g. ``vocals.wav``); the remaining filenames form
domain B in preference order (e.g. ``audio.wav`` then ``audio.mp3``, the
first one that exists is used).
"""

import hashlib
import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

AUDIO_DOMAIN_MODES = ("balanced", "fallback")


def _domain_path(song_dir: Path, names: list[str]) -> Path:
    """
    First existing file among *names* in *song_dir*.

    :param song_dir: Song directory.
    :param names: Candidate filenames in preference order.
    :return: Path to the first existing file.
    :raises FileNotFoundError: If none of the files exist.
    """
    for name in names:
        path = song_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"none of {names} in {song_dir}")


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _song_hash(song: str) -> str:
    return hashlib.sha256(song.encode()).hexdigest()


def balanced_assignment(rows: list[dict]) -> dict[str, str]:
    """
    Assign each song to domain ``"a"`` or ``"b"`` so window counts balance.

    Songs are visited largest-first (sha256 of the song name breaks ties, so
    the result is independent of directory iteration order) and greedily
    given to whichever domain currently has fewer windows.

    :param rows: Manifest rows.
    :return: Mapping of song directory name to ``"a"`` or ``"b"``.
    """
    counts = Counter(Path(r["audio_filepath"]).parent.name for r in rows)
    ordered = sorted(counts, key=lambda song: (-counts[song], _song_hash(song)))
    totals = {"a": 0, "b": 0}
    assignment = {}
    for song in ordered:
        domain = "a" if totals["a"] <= totals["b"] else "b"
        assignment[song] = domain
        totals[domain] += counts[song]
    logger.info(
        f"Balanced assignment: {sum(v == 'a' for v in assignment.values())} / "
        f"{sum(v == 'b' for v in assignment.values())} songs, "
        f"{totals['a']} / {totals['b']} windows"
    )
    return assignment


def rewrite_manifest(
    path: Path,
    filenames: list[str],
    mode: str,
    assignment: dict[str, str] | None = None,
) -> Counter:
    """
    Rewrite ``audio_filepath`` in one manifest according to the domain mode.

    :param path: Manifest JSONL (rewritten in place via a temp file).
    :param filenames: The experiment's ``--filename`` list (domain A first).
    :param mode: ``"balanced"`` (per-song A/B from *assignment*) or
        ``"fallback"`` (every song uses domain B).
    :param assignment: Song -> domain mapping for ``"balanced"``. Songs
        missing from it (only expected for a val-subset song absent from the
        val manifest) get a deterministic hash-based domain.
    :return: Window count per domain.
    """
    rows = _load_rows(path)
    if not rows:
        return Counter()
    domain_files = {"a": filenames[:1], "b": filenames[1:]}
    domains = Counter()
    seen = {}
    for row in rows:
        directory = Path(row["audio_filepath"]).parent
        song = directory.name
        if mode == "fallback":
            domain = "b"
        elif assignment is not None and song in assignment:
            domain = assignment[song]
        else:
            domain = "a" if int(_song_hash(song), 16) % 2 == 0 else "b"
        if seen.setdefault(song, domain) != domain:
            raise RuntimeError(f"song {song} assigned to both domains")
        row["audio_filepath"] = str(_domain_path(directory, domain_files[domain]))
        domains[domain] += 1

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)
    logger.info(f"{path.name}: {len(rows)} rows, {len(seen)} songs, {dict(domains)}")
    return domains


def apply_audio_domains(job_dir: Path, filenames: list[str], mode: str) -> dict:
    """
    Apply an audio-domain mode to an experiment's train/val/val-subset manifests.

    :param job_dir: Experiment directory holding the manifests.
    :param filenames: The experiment's ``--filename`` list (domain A first).
    :param mode: One of :data:`AUDIO_DOMAIN_MODES`.
    :return: ``{manifest name: {domain: windows}}`` summary for config.json.
    """
    if mode not in AUDIO_DOMAIN_MODES:
        raise ValueError(f"audio domain mode must be one of {AUDIO_DOMAIN_MODES}")
    if len(filenames) < 2:
        raise ValueError("audio domain modes need at least two --filename values")

    train = job_dir / "train_manifest.jsonl"
    val = job_dir / "val_manifest.jsonl"
    subset = job_dir / "val_subset_manifest.jsonl"
    summary = {}
    if mode == "fallback":
        for path in (train, val, subset):
            summary[path.name] = dict(rewrite_manifest(path, filenames, mode))
        return summary

    train_assignment = balanced_assignment(_load_rows(train))
    val_rows = _load_rows(val)
    val_assignment = balanced_assignment(val_rows) if val_rows else {}
    summary[train.name] = dict(rewrite_manifest(train, filenames, mode, train_assignment))
    summary[val.name] = dict(rewrite_manifest(val, filenames, mode, val_assignment))
    summary[subset.name] = dict(rewrite_manifest(subset, filenames, mode, val_assignment))
    return summary
