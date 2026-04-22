import csv
import json
import logging
from pathlib import Path
import jiwer
import numpy as np
from alt_eval.tokenizer import WORD, LyricsTokenizer, tokens_as_words

from lyricscribe.schemas import Lyrics

logger = logging.getLogger(__name__)

_TOKENIZER = LyricsTokenizer()


def _normalized_words(text: str, language: str = "en") -> list[str]:
    tokens = _TOKENIZER(text, language=language)
    return [t.text.lower() for t in tokens_as_words(tokens) if WORD in t.tags]

fields = [
        "song_id", "model_name", "word", "word_idx", "start", "end",
        "error_type", "insertion_count",
        "artifact_rms", "vocal_rms", "artifact_to_signal_ratio",
        "spectral_centroid", "spectral_flatness",
    ]

# artifact_rms, root mean squared of the artifact, bascially its loudness
# vocal_rms, root mean squared of the vocal stems
# artifact_to_signal_ratio, ratio of artifact_rms to vocal_rms (0.1 would mean the artifact is 10% as loud as the voice)
# spectral_centroid, the average frequence of the artifact 
# spectral_flatness, how noisy vs. tonal the artifact is


def _load_alignments(dataset_dir: Path) -> dict[str, list[dict]]:
    """
    Load MFA word-level alignments from each song's lyrics.json.

    Walks ``dataset_dir`` subdirectories; for each song that has a
    non-null ``alignment`` field in its ``lyrics.json``, converts word
    times from milliseconds (schema) to seconds (what the downstream
    window-indexing expects).

    :param dataset_dir: Root dataset directory (one subdirectory per song).
    :returns: Dictionary mapping song_id to a list of word dicts, each
        with ``word``, ``start``, and ``end`` keys (start/end in seconds).
    """
    alignments = {}
    for song_dir in sorted(d for d in dataset_dir.iterdir() if d.is_dir()):
        lyrics_path = song_dir / "lyrics.json"
        if not lyrics_path.exists():
            continue
        try:
            lyrics = Lyrics.model_validate_json(lyrics_path.read_text())
        except Exception as e:
            logger.warning(f"Skipping {song_dir.name}: invalid lyrics.json ({e})")
            continue
        if lyrics.alignment is None:
            continue
        alignments[song_dir.name] = [
            {
                "word": w.word,
                "start": w.start / 1000.0,
                "end": (w.start + w.duration) / 1000.0,
            }
            for w in lyrics.alignment.words
        ]
    logger.info(f"Loaded alignments for {len(alignments)} songs")
    return alignments


def _load_artifact_features(features_dir: Path) -> dict[str, dict]:
    """
    Load precomputed artifact features for each song.

    :param features_dir: absolute path to the directory containing one .json features file per song, as produced by the artifacts extract command.
    :returns: Dictionary mapping song_id to its full features dict, containing per-frame arrays for artifact_rms, vocal_rms, artifact_to_signal_ratio, spectral_centroid, and spectral_flatness.
    """
    features = {}
    for path in (sorted(features_dir.glob("*.json"))):
        with open(path) as f:
            data = json.load(f)
        features[data["song_id"]] = data
    logger.info(f"Loaded alignment features for {len(features)} songs")
    return features

def _load_results(result_files: list[Path]) -> dict[tuple[str, str], str]:
    """
    Load model transcription results from one or more .jsonl results files.

    :param result_files: absolute paths to .jsonl files where each line is a
        JSON object containing song_id, model_name, transcription, and error fields.
    :returns: Dictionary mapping (song_id, model_name) tuples to transcription strings. Entries with errors or missing transcriptions are skipped.
    """
    results = {}
    for result_file in result_files:
        with open(result_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("transcription") and not r.get("error"):
                    results[(r["song_id"], r["model_name"])] = r["transcription"]
    logger.info(
        "Loaded %s transcription results from %s result file(s)",
        len(results),
        len(result_files),
    )
    return results


def _load_ground_truth(musedb_dir: Path) -> tuple[dict[str, str], dict[str, str]]:
    """
    Load ground truth lyrics and language for each MUSDB song from its lyrics.json file.

    :param musdb_dir: root directory of the MUSDB dataset, containing one subdirectory per song, each with a lyrics.json file.
    :returns: Tuple of (ground_truth, languages) dicts keyed by song_id.
    """
    ground_truth = {}
    languages = {}
    for song_path in musedb_dir.iterdir():
        if not song_path or not song_path.is_dir():
            continue
        lyric_path = song_path / "lyrics.json"

        if (lyric_path.exists()):
            with open(lyric_path) as f:
                data = json.load(f)

            ground_truth[song_path.name] = data["unsynced"]["data"]
            lang = (data.get("detected_language") or "en").split("-")[0].split("_")[0].lower()
            languages[song_path.name] = lang or "en"

    logger.info(f"Loaded ground truth for {len(ground_truth)} songs")
    return ground_truth, languages
                     


def _get_artifact_features_for_window(features: dict, start_s: float, end_s: float) -> dict[str]:
    """
    Average artifact features over a given time window.

    Converts start and end times in seconds to frame indices using the hop length
    and sample rate stored in the features dict, then averages each feature
    across all frames in that window.

    :param features: features dict for a single song as returned by _load_artifact_features.
    :param start_s: start of the time window in seconds.
    :param end_s: end of the time window in seconds.
    :returns: dictionary containing the mean value of each artifact feature (artifact_rms, vocal_rms, artifact_to_signal_ratio, spectral_centroid, spectral_flatness) over the given window.
    """
    hop = features["hop_length"]
    sample_rate = features["sample_rate"]
    n_frames = features["n_frames"]

    start_frame = max(0, min(int(start_s * sample_rate / hop), n_frames - 1))
    end_frame   = max(start_frame + 1, min(int(end_s * sample_rate / hop) + 1, n_frames))

    result = {}
    for key in (
    "artifact_rms", "vocal_rms", "artifact_to_signal_ratio",
    "spectral_centroid", "spectral_flatness"):
        values = features[key][start_frame:end_frame]
        result[key] = float(np.mean(values)) if values else 0.0

    return result

def _get_word_error(
    reference: str, hypothesis: str, language: str = "en"
) -> tuple[dict[int, str], dict[int, int]]:
    """
    Compute the error type for each reference word using jiwer word alignment.

    Both sides are first tokenized with the alt-eval lyrics tokenizer
    (Cífka et al. 2024), lowercased, and stripped of non-word tokens so
    that casing and punctuation differences are not counted as errors.
    The returned indices are positions in the normalized word list, which
    is what the MFA per-word alignment also uses (MFA tokens are
    lowercase and punctuation-stripped), so downstream positional lookups
    stay valid.

    :param reference: Ground truth lyrics string.
    :param hypothesis: Model transcription string.
    :param language: ISO 639-1 language code for the reference.
    :returns: tuple of (word_errors, insertion_counts) where word_errors maps
        reference word index to error type, and insertion_counts maps each
        reference word index to the number of hypothesis words inserted
        adjacent to it.
    """
    ref_words = _normalized_words(reference, language)
    hyp_words = _normalized_words(hypothesis, language)
    output = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words))
    word_errors: dict[int, str] = {}
    insertion_counts: dict[int, int] = {}

    for chunk in output.alignments:
        for op in chunk:
            if op.type == "equal":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    word_errors[i] = "correct"
            elif op.type == "insert":
                n_inserted = op.hyp_end_idx - op.hyp_start_idx
                target_idx = max(0, op.ref_start_idx - 1)
                insertion_counts[target_idx] = insertion_counts.get(target_idx, 0) + n_inserted
            elif op.type == "delete":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    word_errors[i] = "deletion"
            elif op.type == "substitute":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    word_errors[i] = "substitution"

    return word_errors, insertion_counts


def build_dataset(
    features_dir: Path,
    results_files: list[Path],
    musdb_dir: Path,
    *,
    csv_output: Path | None = None,
) -> list[dict]:
    """
    Build the word-level dataset combining MFA alignments, artifact features,
    and model transcription results.

    For each song, each model, and each aligned word, looks up the artifact
    features during that word's time window and the model's error type for
    that word.  Returns one dict per (word, model) pair.

    :param features_dir: directory containing artifact feature .json files.
    :param results_files: paths to one or more .jsonl transcription results files.
    :param musdb_dir: root MUSDB directory. Alignments are read from the
        ``alignment`` field of each song's ``lyrics.json``; ground-truth
        lyrics from the same file's ``synced`` field.
    :param csv_output: optional path to also write the dataset as CSV.
    :returns: list of row dicts.
    """
    alignments = _load_alignments(musdb_dir)
    features = _load_artifact_features(features_dir)
    results = _load_results(results_files)
    ground_truth, languages = _load_ground_truth(musdb_dir)

    models = sorted(set(model for _, model in results.keys()))

    result_songs = set(song_id for song_id, _ in results.keys())
    common_songs = set(alignments) & set(features) & result_songs

    rows: list[dict] = []
    songs_processed = 0

    for song_id in sorted(common_songs):
        words = alignments[song_id]
        logger.info(f"{song_id} has {len(words)} aligned words")
        song_features = features[song_id]
        reference = ground_truth[song_id]

        for model_name in models:
            if (song_id, model_name) not in results:
                logger.warning(f"No results for {song_id} / {model_name}")
                continue

            hypothesis = results[(song_id, model_name)]

            try:
                word_errors, insertion_counts = _get_word_error(
                    reference, hypothesis, language=languages.get(song_id, "en")
                )
                logger.info(f"word_errors sample: {list(word_errors.items())[:3]}")
            except Exception as e:
                logger.warning(f"JiWER failed for {song_id}: {e}")
                continue

            for word_index, word_info in enumerate(words):
                artifact_features = _get_artifact_features_for_window(
                    song_features, word_info["start"], word_info["end"]
                )

                rows.append({
                    "song_id":    song_id,
                    "model_name": model_name,
                    "word":       word_info["word"],
                    "word_idx":   word_index,
                    "start":      round(word_info["start"], 4),
                    "end":        round(word_info["end"], 4),
                    "error_type": word_errors.get(word_index, "unknown"),
                    "insertion_count": insertion_counts.get(word_index, 0),
                    **artifact_features,
                })
            songs_processed += 1
            logger.info(f"Processed {song_id} ({songs_processed}/{len(common_songs)})")

    if csv_output is not None:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Wrote {len(rows)} rows to {csv_output}")

    return rows



_QUARTILE_FIELDS = [
    "model", "quartile", "n_words", "error_rate",
    "deletion_rate", "substitution_rate", "mean_insertions_per_word",
    "mean_artifact_to_signal",
]


def analyse(
    rows: list[dict],
    *,
    csv_output: Path | None = None,
) -> list[dict]:
    """
    Run correlation analysis on the word-level dataset.

    Splits words into artifact energy quartiles and computes error rate,
    deletion rate, substitution rate, and mean insertions per word per
    quartile per model.

    :param rows: word-level dataset as returned by :func:`build_dataset`.
    :param csv_output: optional path to write quartile summary CSV.
    :returns: list of quartile summary dicts.
    """
    rows = [r for r in rows if r["artifact_to_signal_ratio"] != float("inf")]

    models = sorted(set(r["model_name"] for r in rows))
    logger.info(f"Models found: {models}")

    asr_values = [r["artifact_to_signal_ratio"] for r in rows]
    q25, q50, q75 = np.percentile(asr_values, [25, 50, 75])

    def get_quartile(val):
        if val <= q25: return "Q1"
        if val <= q50: return "Q2"
        if val <= q75: return "Q3"
        return "Q4"

    quartile_rows = []
    for model_name in models:
        model_rows = [r for r in rows if r["model_name"] == model_name]
        for q_label in ["Q1", "Q2", "Q3", "Q4"]:
            q_rows = [r for r in model_rows if get_quartile(r["artifact_to_signal_ratio"]) == q_label]
            if not q_rows:
                continue
            n = len(q_rows)
            sub_del = sum(1 for r in q_rows if r["error_type"] != "correct")
            insertions = sum(r["insertion_count"] for r in q_rows)
            quartile_rows.append({
                "model":    model_name,
                "quartile": q_label,
                "n_words":  n,
                "error_rate":         round((sub_del + insertions) / n, 4),
                "deletion_rate":      round(sum(1 for r in q_rows if r["error_type"] == "deletion") / n, 4),
                "substitution_rate":  round(sum(1 for r in q_rows if r["error_type"] == "substitution") / n, 4),
                "mean_insertions_per_word": round(insertions / n, 4),
                "mean_artifact_to_signal": round(float(np.mean([r["artifact_to_signal_ratio"] for r in q_rows])), 4),
            })

    if csv_output is not None:
        csv_output.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_QUARTILE_FIELDS)
            writer.writeheader()
            writer.writerows(quartile_rows)
        logger.info(f"Wrote quartile analysis to {csv_output}")

    return quartile_rows
