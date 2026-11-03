"""
Batch scoring of job dirs: count verification, corrupt-line guard, the
reference-quality slice, common-song pairing, per-language and raw WER.
"""

import logging
from pathlib import Path

from lyricscribe.analysis.common import (
    Job,
    corpus_stats,
    load_job,
    raw_references,
)

logger = logging.getLogger(__name__)


class IncompleteJobError(ValueError):
    """A job dir has fewer usable hypotheses than its dataset has songs."""


def scorable_songs(job: Job, require_complete: bool = True) -> list[str]:
    """
    Songs that can be scored: a non-``None`` hypothesis and a non-empty ref.

    :param job: Loaded job.
    :param require_complete: Raise if the job has fewer successful songs
        than its dataset size, or any corrupt result lines.
    :return: Sorted song IDs.
    :raises IncompleteJobError: When ``require_complete`` and the job is
        partial or corrupt.
    """
    ok = sorted(
        s for s, h in job.hyps.items() if h is not None and job.refs.get(s)
    )
    if require_complete:
        if job.corrupt_lines:
            raise IncompleteJobError(
                f"{job.path}: {job.corrupt_lines} unparseable result line(s) — "
                "parallel chunks probably raced on one results.jsonl"
            )
        want = job.expected
        have = sum(1 for h in job.hyps.values() if h is not None)
        if want is not None and have != want:
            raise IncompleteJobError(f"{job.path}: {have}/{want} songs transcribed")
    return ok


def score_job(
    job: Job,
    songs: list[str] | None = None,
    keep: set[str] | None = None,
    by_language: bool = False,
    raw: bool = False,
) -> dict:
    """
    Score one job, optionally restricted to a song set.

    :param job: Loaded job.
    :param songs: Songs to score (default: :func:`scorable_songs` without the
        completeness check).
    :param keep: Optional reference-quality keep-set (filtered slice).
    :param by_language: Also report per-language stats.
    :param raw: Also report WER on un-normalised text (whitespace split only)
        — the gap to the normalised number is surface style the metric
        discards (casing, punctuation, contractions).
    :return: ``{"all": stats, "filtered": stats?, "by_language": {...}?,
        "raw": stats?}``; stats as from :func:`corpus_stats`.
    """
    if songs is None:
        songs = scorable_songs(job, require_complete=False)
    out = {"all": _stats(job, songs)}
    if keep is not None:
        out["filtered"] = _stats(job, [s for s in songs if s in keep])
    if by_language:
        langs = sorted({job.langs[s] for s in songs})
        out["by_language"] = {
            lang: _stats(job, [s for s in songs if job.langs[s] == lang])
            for lang in langs
        }
    if raw:
        raw_refs = {}
        for d in job.dataset_dirs:
            raw_refs.update({k: v[1] for k, v in raw_references(d).items()})
        ids = [s for s in songs if raw_refs.get(s, "").strip()]
        out["raw"] = corpus_stats(
            [raw_refs[s] for s in ids], [job.hyps[s] or "" for s in ids]
        )
    return out


def _stats(job: Job, songs: list[str]) -> dict:
    return corpus_stats(
        [job.refs[s] for s in songs], [job.normalized_hyp(s) for s in songs]
    )


def score_many(
    job_dirs: list[Path],
    keep: set[str] | None = None,
    common: bool = False,
    by_language: bool = False,
    raw: bool = False,
    require_complete: bool = True,
) -> dict[str, dict]:
    """
    Score several job dirs, skipping (and reporting) partial or corrupt ones.

    :param job_dirs: Job directories.
    :param keep: Optional reference-quality keep-set.
    :param common: Restrict every job to the songs common to all of them, so
        partially finished runs compare without a song-set confound (this
        implies ``require_complete=False``).
    :param by_language: Per-language breakdown.
    :param raw: Un-normalised WER alongside.
    :param require_complete: Refuse partial/corrupt jobs.
    :return: ``{job_dir: result}``; a skipped job maps to ``{"error": msg}``.
    """
    jobs, out = {}, {}
    for d in job_dirs:
        try:
            job = load_job(d)
            scorable_songs(job, require_complete=require_complete and not common)
            jobs[str(d)] = job
        except (FileNotFoundError, IncompleteJobError) as e:
            out[str(d)] = {"error": str(e)}
    shared = None
    if common and jobs:
        shared = set.intersection(
            *(set(scorable_songs(j, require_complete=False)) for j in jobs.values())
        )
    for name, job in jobs.items():
        songs = sorted(shared) if shared is not None else None
        out[name] = score_job(job, songs, keep, by_language, raw)
    return out


def discover_job_dirs(root: Path, pattern: str) -> list[Path]:
    """
    Job dirs under *root* whose path matches a glob and that hold results.

    :param root: Search root (e.g. the jobs directory).
    :param pattern: Glob for the job dir relative to *root*, e.g.
        ``**/jam-alt*``.
    :return: Sorted job dirs containing both ``config.json`` and results.
    """
    found = set()
    for d in Path(root).glob(pattern):
        if d.is_dir() and (d / "config.json").exists() and any(d.glob("results*.jsonl")):
            found.add(d)
    return sorted(found)


def format_stats(label: str, s: dict) -> str:
    """One-line human summary of a stats dict."""
    sh = s["share"]
    return (
        f"{label}: n={s['n']} WER={s['wer']:.4f} "
        f"I/D/S={sh['I']:.1f}/{sh['D']:.1f}/{sh['S']:.1f} "
        f"(I={s['I']} D={s['D']} S={s['S']})"
    )
