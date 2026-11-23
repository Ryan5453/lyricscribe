import logging
import math
from pathlib import Path

from lyricscribe.analysis.common import (
    Job,
    corpus_stats,
    load_job,
    raw_references,
)

logger = logging.getLogger(__name__)


class IncompleteJobError(ValueError):
    """
    A job dir has fewer usable hypotheses than its dataset has songs.
    """


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
        want = len(job.refs)
        have = sum(1 for h in job.hyps.values() if h is not None)
        if have != want:
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
    """
    Corpus stats over *songs* of *job*.

    :param job: Loaded job.
    :param songs: Songs to include.
    :return: Stats from :func:`corpus_stats`.
    """
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


def readability_job(
    job_dir: Path,
    *,
    keep: set[str] | None = None,
    by_language: bool = False,
    require_complete: bool = True,
) -> dict:
    """
    Compute the full Jam-ALT readability metric suite (``alt_eval.compute_metrics``):
    WER, case-sensitive WER / case error rate, and punctuation, parentheses, line-break
    and section-break precision/recall/F1 on un-normalised hypotheses and references.

    :param job_dir: Transcription job directory.
    :param keep: Optional reference-quality song ID keep-set.
    :param by_language: Also compute readability metrics per language.
    :param require_complete: Refuse partial or corrupt job dirs.
    :return: ``{"n_songs": int, "all": {...}, "by_language": {...}?}``.
    """
    from alt_eval import compute_metrics

    job = load_job(job_dir)
    songs = scorable_songs(job, require_complete=require_complete)
    if keep is not None:
        songs = [s for s in songs if s in keep]

    raw_refs: dict[str, tuple[str, str, str]] = {}
    for d in job.dataset_dirs:
        raw_refs.update(raw_references(d))

    ids = [s for s in songs if raw_refs.get(s, ("", "", ""))[1].strip()]
    if not ids:
        raise IncompleteJobError(f"{job_dir}: no songs with non-empty raw references")

    refs = [raw_refs[s][1] for s in ids]
    hyps = [job.hyps[s] or "" for s in ids]
    langs = [raw_refs[s][0] for s in ids]

    def _clean(m: dict) -> dict:
        out_m = {}
        for k, v in m.items():
            if isinstance(v, (int, float)):
                fv = float(v)
                out_m[k] = None if math.isnan(fv) else fv
            else:
                out_m[k] = v
        return out_m

    out: dict = {
        "n_songs": len(ids),
        "all": _clean(compute_metrics(refs, hyps, languages=langs)),
    }
    if by_language:
        out["by_language"] = {}
        for lang in sorted(set(langs)):
            idx = [i for i, lg in enumerate(langs) if lg == lang]
            out["by_language"][lang] = _clean(
                compute_metrics([refs[i] for i in idx], [hyps[i] for i in idx], languages=lang)
            )
    return out


def readability_many(
    job_dirs: list[Path],
    *,
    keep: set[str] | None = None,
    by_language: bool = False,
    require_complete: bool = True,
) -> dict[str, dict]:
    """
    Compute Jam-ALT readability metrics across several job directories.

    :param job_dirs: Job directories.
    :param keep: Optional song ID keep-set.
    :param by_language: Per-language breakdown.
    :param require_complete: Refuse partial or corrupt jobs.
    :return: ``{job_dir: readability_dict}``; skipped jobs map to ``{"error": msg}``.
    """
    out: dict[str, dict] = {}
    for d in job_dirs:
        try:
            out[str(d)] = readability_job(
                d, keep=keep, by_language=by_language, require_complete=require_complete
            )
        except (FileNotFoundError, IncompleteJobError) as e:
            out[str(d)] = {"error": str(e)}
    return out


def qualitative_examples(
    job_dirs: list[Path],
    *,
    songs: list[str] | None = None,
    top_diff: int = 10,
    deloop_max_run: int | None = None,
    labels: list[str] | None = None,
    max_words: int = 80,
) -> list[dict]:
    """
    Build side-by-side multi-model song comparison records for qualitative error analysis.

    Ranks common songs by the largest cross-system WER spread (unless explicit *songs* are
    requested) and includes each system's WER, I/D/S counts and normalised/raw excerpts.

    :param job_dirs: Two or more job directories on the same dataset.
    :param songs: Explicit song IDs to inspect (overrides *top_diff* ranking).
    :param top_diff: Number of highest-divergence songs to return when *songs* is unset.
    :param deloop_max_run: Optional deloop cap applied to hypotheses.
    :param labels: Optional display names matching *job_dirs*.
    :param max_words: Word cap for the preview excerpt of each song.
    :return: List of per-song comparison dictionaries.
    """
    from lyricscribe.evaluate import job_song_pairs

    if not job_dirs:
        raise ValueError("Need at least one job directory")
    names = labels if (labels and len(labels) == len(job_dirs)) else [str(d) for d in job_dirs]

    per_job: dict[str, dict[str, tuple[str, str, str]]] = {}
    for name, d in zip(names, job_dirs):
        loaded = job_song_pairs(Path(d), verbose=False, deloop_max_run=deloop_max_run)
        if loaded is None:
            raise ValueError(f"{d}: no evaluable songs")
        per_job[name] = {sid: (ref, hyp, lang) for sid, ref, hyp, lang in loaded["songs"]}

    common_ids = sorted(set.intersection(*(set(m) for m in per_job.values())))
    if songs:
        target_ids = [s for s in songs if s in common_ids]
    else:
        target_ids = common_ids

    records: list[dict] = []
    for sid in target_ids:
        ref, _, lang = per_job[names[0]][sid]
        systems: dict[str, dict] = {}
        wers: list[float] = []
        for name in names:
            r, h, _ = per_job[name][sid]
            st = corpus_stats([r], [h])
            wers.append(st["wer"])
            systems[name] = {
                "wer": round(st["wer"], 4),
                "I": st["I"],
                "D": st["D"],
                "S": st["S"],
                "H": st["H"],
                "excerpt": " ".join(h.split()[:max_words]),
            }
        spread = (max(wers) - min(wers)) if wers else 0.0
        records.append({
            "song_id": sid,
            "language": lang,
            "ref_words": len(ref.split()),
            "wer_spread": round(spread, 4),
            "ref_excerpt": " ".join(ref.split()[:max_words]),
            "systems": systems,
        })

    if not songs:
        records.sort(key=lambda x: (-x["wer_spread"], x["song_id"]))
        records = records[:top_diff]
    return records
