import json
import logging
from pathlib import Path

import jiwer
from alt_eval.tokenizer import WORD, LyricsTokenizer, tokens_as_words

from lyricscribe.ensemble.deloop import deloop

logger = logging.getLogger(__name__)

_TOKENIZER = LyricsTokenizer()

_DATASET_CACHE: dict[Path, dict] = {}


def _language_for(lyrics_data: dict) -> str:
    """
    Base language code of a song from its ``lyrics.json``.

    :param lyrics_data: Parsed ``lyrics.json``.
    :return: Lowercase base language code, ``"en"`` if none is set.
    """
    lang = lyrics_data.get("detected_language") or "en"
    return lang.split("-")[0].split("_")[0].lower() or "en"


def _normalized(text: str, language: str) -> str:
    """
    Normalise text with the alt-eval tokenizer into space-joined lowercase words.

    :param text: Lyrics or transcription text.
    :param language: Language code.
    :return: The normalised text.
    """
    tokens = _TOKENIZER(text, language=language)
    return " ".join(t.text.lower() for t in tokens_as_words(tokens) if WORD in t.tags)


def _load_dataset(directory: Path) -> dict:
    """
    Walk *directory* once, load every song's ``lyrics.json``, and
    pre-normalize the reference text. Cached by resolved path so repeated
    calls for the same dataset are free.

    :param directory: Dataset directory of song subdirectories.
    :return: ``{"refs_normalized": {song: ref}, "languages": {song: code}}``.
    """
    key = directory.resolve()
    cached = _DATASET_CACHE.get(key)
    if cached is not None:
        return cached

    refs_normalized: dict[str, str] = {}
    languages: dict[str, str] = {}
    for song_dir in directory.iterdir():
        if not song_dir.is_dir():
            continue
        lyrics_path = song_dir / "lyrics.json"
        if not lyrics_path.exists():
            continue
        with open(lyrics_path) as f:
            lyrics_data = json.load(f)
        lang = _language_for(lyrics_data)
        ref = lyrics_data["unsynced"]["data"]
        languages[song_dir.name] = lang
        refs_normalized[song_dir.name] = _normalized(ref, lang)

    result = {"refs_normalized": refs_normalized, "languages": languages}
    _DATASET_CACHE[key] = result
    return result


def _registered_songs(job_dir: Path) -> set[str]:
    """
    Songs a job registered across its ``chunk_*.json`` manifests.

    :param job_dir: Transcription job directory.
    :return: Song ids (empty if the job has no manifests).
    """
    names: set[str] = set()
    for chunk_path in job_dir.glob("chunk_*.json"):
        try:
            with open(chunk_path) as f:
                names.update(entry["name"] for entry in json.load(f).get("files", []))
        except (json.JSONDecodeError, OSError, KeyError):
            continue
    return names


def _counts(refs: list[str], hyps: list[str]) -> dict:
    """
    Corpus WER and I/D/S/H counts.

    :param refs: Normalised references.
    :param hyps: Normalised hypotheses.
    :return: ``n_songs``, ``wer``, ``insertions``, ``deletions``, ``substitutions``, ``hits``.
    """
    measures = jiwer.process_words(refs, hyps)
    ref_length = measures.substitutions + measures.deletions + measures.hits
    errors = measures.substitutions + measures.deletions + measures.insertions
    return {
        "n_songs": len(refs),
        "wer": errors / ref_length if ref_length else 0.0,
        "insertions": measures.insertions,
        "deletions": measures.deletions,
        "substitutions": measures.substitutions,
        "hits": measures.hits,
    }


def job_song_pairs(job_dir: Path, verbose: bool = False, deloop_max_run: int | None = None) -> dict | None:
    """
    The normalised reference and hypothesis of every scored song in a job,
    under :func:`evaluate_job`'s rules (failed or missing songs are empty
    hypotheses; songs without a reference are skipped).

    :param job_dir: Transcription job directory.
    :param verbose: Log why songs or the job are skipped.
    :param deloop_max_run: If set, collapse loops in each normalised hypothesis.
    :return: ``songs`` (``(song_id, ref, hyp, language)`` tuples, sorted),
        ``n_failed``, ``n_missing``, ``n_delooped`` and ``config``, or ``None``.
    """
    config_path = job_dir / "config.json"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = json.load(f)

    directories = [Path(d) for d in config.get("directories", [])]
    if not directories:
        if verbose:
            logger.warning("Job config has no directories")
        return None

    refs_normalized: dict[str, str] = {}
    languages: dict[str, str] = {}
    for directory in directories:
        cache = _load_dataset(directory)
        refs_normalized.update(cache["refs_normalized"])
        languages.update(cache["languages"])

    if not refs_normalized:
        if verbose:
            logger.warning("No ground-truth lyrics found in dataset directories")
        return None

    results_map = {}
    for results_path in job_dir.glob("results*.jsonl"):
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    results_map[r["song_id"]] = r
                except json.JSONDecodeError:
                    if verbose:
                        logger.warning(f"Skipping invalid JSON line in {results_path}")
                    continue

    if not results_map:
        if verbose:
            logger.warning("No results found in job directory")
        return None

    songs = sorted(_registered_songs(job_dir) | set(results_map))
    pairs: list[tuple[str, str, str, str]] = []
    n_failed = n_missing = n_delooped = 0
    for song_id in songs:
        ref_norm = refs_normalized.get(song_id)
        if ref_norm is None:
            if verbose:
                logger.warning(f"{song_id}: skipped (no ground truth)")
            continue
        if not ref_norm:
            if verbose:
                logger.warning(f"{song_id}: skipped (empty reference after normalization)")
            continue
        lang = languages[song_id]
        r = results_map.get(song_id)
        hypothesis = r.get("transcription") if r else None
        if hypothesis is None:
            if r is None:
                n_missing += 1
            else:
                n_failed += 1
            if verbose:
                logger.warning(f"{song_id}: scored as empty ({'missing' if r is None else 'failed'})")
            hypothesis = ""
        hyp_norm = _normalized(hypothesis, lang)
        if deloop_max_run:
            words = hyp_norm.split()
            fixed = deloop(words, max_run=int(deloop_max_run))
            if fixed != words:
                n_delooped += 1
                hyp_norm = " ".join(fixed)
        pairs.append((song_id, ref_norm, hyp_norm, lang))

    if not pairs:
        if verbose:
            logger.warning("No songs could be evaluated")
        return None
    return {
        "songs": pairs,
        "n_failed": n_failed,
        "n_missing": n_missing,
        "n_delooped": n_delooped,
        "config": config,
    }


def evaluate_job(job_dir: Path, verbose: bool = False, deloop_max_run: int | None = None) -> dict | None:
    """
    Evaluate a single transcription job directory, returning aggregate
    WER stats and the job config, or None if the job cannot be evaluated.

    References and hypotheses both flow through the alt-eval lyrics
    tokenizer and normalizer (Cífka et al. 2024) before WER / I / D / S
    are computed, so differences in casing, punctuation, and per-language
    contraction handling are not counted as errors.

    Every registered song with a reference is scored: a failed song (no
    transcription) or one never run counts as an empty hypothesis, so systems
    are always compared on the same songs. ``complete`` is False when any
    song was failed or missing.

    :param job_dir: Transcription job directory.
    :param verbose: Log why songs or the job are skipped.
    :param deloop_max_run: If set, collapse loops in each normalised
        hypothesis (:func:`~lyricscribe.ensemble.deloop.deloop`) before scoring.
    :return: ``n_songs``, ``wer``, ``insertions``, ``deletions``,
        ``substitutions``, ``hits``, ``n_failed``, ``n_missing``,
        ``complete``, ``n_delooped``, ``by_language`` and ``config``, or ``None``.
    """
    pairs = job_song_pairs(job_dir, verbose=verbose, deloop_max_run=deloop_max_run)
    if pairs is None:
        return None
    config = pairs["config"]
    n_failed, n_missing, n_delooped = pairs["n_failed"], pairs["n_missing"], pairs["n_delooped"]
    refs_list = [s[1] for s in pairs["songs"]]
    hyps_list = [s[2] for s in pairs["songs"]]
    langs_list = [s[3] for s in pairs["songs"]]

    by_language = {}
    for lang in sorted(set(langs_list)):
        idx = [i for i, x in enumerate(langs_list) if x == lang]
        by_language[lang] = _counts([refs_list[i] for i in idx], [hyps_list[i] for i in idx])
    return {
        **_counts(refs_list, hyps_list),
        "n_failed": n_failed,
        "n_missing": n_missing,
        "complete": n_failed == 0 and n_missing == 0,
        "deloop_max_run": deloop_max_run,
        "n_delooped": n_delooped,
        "by_language": by_language,
        "config": config,
    }


def collect_evaluation_data(jobs_dir: Path) -> list[dict]:
    """
    Walk all job subdirectories under *jobs_dir*, evaluate each one, and
    return a list of flat summary dicts ready to be turned into a DataFrame.

    :param jobs_dir: Root searched recursively for ``config.json``.
    :return: One summary dict per job that could be evaluated.
    """
    all_stats: list[dict] = []

    for config_path in jobs_dir.glob("**/config.json"):
        job_dir = config_path.parent
        stats = evaluate_job(job_dir, verbose=False)
        if stats is None:
            continue
        config = stats["config"]
        vad_on = config.get("vad", False)
        all_stats.append(
            {
                "job_dir": str(job_dir.relative_to(jobs_dir)),
                "model": config.get("model", "unknown"),
                "dataset": ", ".join(
                    Path(d).name for d in config.get("directories", [])
                ),
                "filename": config.get("filename", ""),
                "vad": vad_on,
                "vad_method": (
                    config.get("vad_method", "silero") if vad_on else ""
                ),
                "vad_source": config.get("vad_filename") or "",
                "chunked": config.get("chunked", False),
                "repetition_penalty": (
                    float(config.get("repetition_penalty", 1.0) or 1.0)
                    if "whisper" in config.get("model", "").lower()
                    else "-"
                ),
                "no_repeat_ngram_size": (
                    int(config.get("no_repeat_ngram_size", 0) or 0)
                    if "whisper" in config.get("model", "").lower()
                    else "-"
                ),
                "wer": stats["wer"],
                "n_songs": stats["n_songs"],
                "insertions": stats["insertions"],
                "deletions": stats["deletions"],
                "substitutions": stats["substitutions"],
                "hits": stats["hits"],
            }
        )

    all_stats.sort(key=lambda x: x["wer"])
    return all_stats
