import json
import logging
from pathlib import Path

import jiwer

logger = logging.getLogger(__name__)


def evaluate_job(job_dir: Path, verbose: bool = False) -> dict | None:
    """
    Evaluate a single transcription job directory, returning aggregate
    WER stats and the job config, or None if the job cannot be evaluated.
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
    for directory in directories:
        for song_dir in directory.iterdir():
            if not song_dir.is_dir():
                continue
            lyrics_path = song_dir / "lyrics.json"
            if lyrics_path.exists():
                with open(lyrics_path) as f:
                    lyrics_data = json.load(f)
                references[song_dir.name] = lyrics_data["unsynced"]["data"]

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

    totals: dict[str, list] = {
        "wer": [],
        "insertions": [],
        "deletions": [],
        "substitutions": [],
    }
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

        measures = jiwer.compute_measures(references[song_id], hypothesis)

        if verbose:
            logger.debug(
                f"{song_id}: WER={measures['wer']:.2%}  "
                f"I={measures['insertions']}  "
                f"D={measures['deletions']}  "
                f"S={measures['substitutions']}"
            )

        totals["wer"].append(measures["wer"])
        totals["insertions"].append(measures["insertions"])
        totals["deletions"].append(measures["deletions"])
        totals["substitutions"].append(measures["substitutions"])

    n = len(totals["wer"])
    if n == 0:
        if verbose:
            logger.warning("No songs could be evaluated")
        return None

    return {
        "n_songs": n,
        "mean_wer": sum(totals["wer"]) / n,
        "insertions": sum(totals["insertions"]),
        "deletions": sum(totals["deletions"]),
        "substitutions": sum(totals["substitutions"]),
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
                "mean_wer": stats["mean_wer"],
                "n_songs": stats["n_songs"],
                "insertions": stats["insertions"],
                "deletions": stats["deletions"],
                "substitutions": stats["substitutions"],
            }
        )

    all_stats.sort(key=lambda x: x["mean_wer"])
    return all_stats
