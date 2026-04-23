import json
import logging
from pathlib import Path

import jiwer
from alt_eval.tokenizer import WORD, LyricsTokenizer, tokens_as_words

logger = logging.getLogger(__name__)

_TOKENIZER = LyricsTokenizer()

# Cache normalized references per dataset directory. Keyed on resolved Path.
# Jobs that share a dataset (all Whisper musdb_alt configs, etc.) walk the
# directory + tokenize every reference exactly once instead of once per job.
_DATASET_CACHE: dict[Path, dict] = {}


def _language_for(lyrics_data: dict) -> str:
    lang = lyrics_data.get("detected_language") or "en"
    return lang.split("-")[0].split("_")[0].lower() or "en"


def _normalized(text: str, language: str) -> str:
    tokens = _TOKENIZER(text, language=language)
    return " ".join(t.text.lower() for t in tokens_as_words(tokens) if WORD in t.tags)


def _load_dataset(directory: Path) -> dict:
    """
    Walk *directory* once, load every song's ``lyrics.json``, and
    pre-normalize the reference text. Cached by resolved path so repeated
    calls for the same dataset are free.
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


def evaluate_job(job_dir: Path, verbose: bool = False) -> dict | None:
    """
    Evaluate a single transcription job directory, returning aggregate
    WER stats and the job config, or None if the job cannot be evaluated.

    References and hypotheses both flow through the alt-eval lyrics
    tokenizer and normalizer (Cífka et al. 2024) before WER / I / D / S
    are computed, so differences in casing, punctuation, and per-language
    contraction handling are not counted as errors.
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

    # Collect paired refs/hyps for a single corpus-level jiwer call.
    refs_list: list[str] = []
    hyps_list: list[str] = []
    for song_id, r in results_map.items():
        hypothesis = r.get("transcription")
        if hypothesis is None:
            if verbose:
                logger.warning(f"{song_id}: skipped (no transcription)")
            continue
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
        hyp_norm = _normalized(hypothesis, lang)
        refs_list.append(ref_norm)
        hyps_list.append(hyp_norm)

    n = len(refs_list)
    if n == 0:
        if verbose:
            logger.warning("No songs could be evaluated")
        return None

    measures = jiwer.process_words(refs_list, hyps_list)

    ref_length = measures.substitutions + measures.deletions + measures.hits
    errors = measures.substitutions + measures.deletions + measures.insertions
    return {
        "n_songs": n,
        "wer": errors / ref_length if ref_length else 0.0,
        "insertions": measures.insertions,
        "deletions": measures.deletions,
        "substitutions": measures.substitutions,
        "hits": measures.hits,
        "config": config,
    }


def collect_evaluation_data(jobs_dir: Path) -> list[dict]:
    """
    Walk all job subdirectories under *jobs_dir*, evaluate each one, and
    return a list of flat summary dicts ready to be turned into a DataFrame.
    """
    all_stats: list[dict] = []

    for config_path in jobs_dir.glob("**/config.json"):
        job_dir = config_path.parent
        stats = evaluate_job(job_dir, verbose=False)
        if stats is None:
            continue
        config = stats["config"]
        all_stats.append(
            {
                "job_dir": str(job_dir.relative_to(jobs_dir)),
                "model": config.get("model", "unknown"),
                "dataset": ", ".join(
                    Path(d).name for d in config.get("directories", [])
                ),
                "filename": config.get("filename", ""),
                "vad": config.get("vad", False),
                "chunked": config.get("chunked", False),
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
