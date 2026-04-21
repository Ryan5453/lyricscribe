import json
import logging
from pathlib import Path

import jiwer
from alt_eval.tokenizer import WORD, LyricsTokenizer, tokens_as_words

logger = logging.getLogger(__name__)

_TOKENIZER = LyricsTokenizer()


def _language_for(lyrics_data: dict) -> str:
    lang = lyrics_data.get("detected_language") or "en"
    return lang.split("-")[0].split("_")[0].lower() or "en"


def _normalized(text: str, language: str) -> str:
    tokens = _TOKENIZER(text, language=language)
    return " ".join(t.text.lower() for t in tokens_as_words(tokens) if WORD in t.tags)


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

    references: dict[str, str] = {}
    languages: dict[str, str] = {}
    for directory in directories:
        for song_dir in directory.iterdir():
            if not song_dir.is_dir():
                continue
            lyrics_path = song_dir / "lyrics.json"
            if lyrics_path.exists():
                with open(lyrics_path) as f:
                    lyrics_data = json.load(f)
                references[song_dir.name] = lyrics_data["unsynced"]["data"]
                languages[song_dir.name] = _language_for(lyrics_data)

    if not references:
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

    results = list(results_map.values())

    if not results:
        if verbose:
            logger.warning("No results found in job directory")
        return None

    n = 0
    totals = {"insertions": 0, "deletions": 0, "substitutions": 0, "hits": 0}
    for r in results:
        song_id = r["song_id"]
        hypothesis = r["transcription"]

        if hypothesis is None:
            if verbose:
                logger.warning(f"{song_id}: skipped (no transcription)")
            continue

        if song_id not in references:
            if verbose:
                logger.warning(f"{song_id}: skipped (no ground truth)")
            continue

        lang = languages[song_id]
        ref_norm = _normalized(references[song_id], lang)
        hyp_norm = _normalized(hypothesis, lang)
        if not ref_norm:
            if verbose:
                logger.warning(f"{song_id}: skipped (empty reference after normalization)")
            continue

        measures = jiwer.process_words([ref_norm], [hyp_norm])

        if verbose:
            ref_len = measures.substitutions + measures.deletions + measures.hits
            wer = (measures.substitutions + measures.deletions + measures.insertions) / ref_len if ref_len else 0.0
            logger.debug(
                f"{song_id} [{lang}]: WER={wer:.2%}  "
                f"I={measures.insertions}  "
                f"D={measures.deletions}  "
                f"S={measures.substitutions}"
            )

        totals["insertions"] += measures.insertions
        totals["deletions"] += measures.deletions
        totals["substitutions"] += measures.substitutions
        totals["hits"] += measures.hits
        n += 1

    if n == 0:
        if verbose:
            logger.warning("No songs could be evaluated")
        return None

    ref_length = totals["substitutions"] + totals["deletions"] + totals["hits"]
    errors = totals["substitutions"] + totals["deletions"] + totals["insertions"]
    return {
        "n_songs": n,
        "wer": errors / ref_length if ref_length else 0.0,
        "insertions": totals["insertions"],
        "deletions": totals["deletions"],
        "substitutions": totals["substitutions"],
        "hits": totals["hits"],
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
