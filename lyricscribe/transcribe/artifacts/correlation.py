import csv
import json
import logging
from pathlib import Path


import jiwer
import numpy as np


logger = logging.getLogger(__name__)


fields = [
        "song_id", "model_name", "word", "word_idx", "start", "end",
        "error_type",
        "artifact_rms", "vocal_rms", "artifact_to_signal_ratio",
        "spectral_centroid", "spectral_flatness",
    ]

def _load_alignments(alignments_dir: Path) -> dict[str, list[dict]]:
    alignments = {}
    for path in  (sorted(alignments_dir.glob("*.json"))):
        with open(path) as f:
            data = json.load(f)
        alignments[data["song_id"]] = data["words"]
    logger.info(f"Loaded alignments for {len(alignments)} songs")
    return alignments


def _load_artifact_features(features_dir: Path) -> dict[str, dict]:
    features = {}
    for path in (sorted(features_dir.glob("*.json"))):
        with open(path) as f:
            data = json.load(path)
        features[data["song_id"]] = data
    logger.info(f"Loaded alignment features for {len(features)} songs")
    return features

def _load_results(result_file: Path) -> dict[tuple[str, str], str]:
    results = {}
    with open(result_file) as f:
        for line in f:
            line = line.strip()
            
            if not line:
                continue
            
            r = json.load(line)

            if r.get("transcription") and not r.get("error"):
                results[(r["song_id"], r["model"])] = r["transcription"]
    logger.info(f"Loaded {len(results)} transcription results")
    return results


def _load_ground_truth(musedb_dir: Path) -> dict[str, str]:
    ground_truth = {}
    for song_path in musedb_dir.iterdir():
        if not song_path or not song_path.is_dir():
            continue
        lyric_path = song_path / "lyrics.json"

        if (lyric_path.exists()):
            with open(lyric_path) as f:
                data = json.load(f)

            ground_truth[song_path.name] = data["unsynced"]["data"]

    logger.info(f"Loaded ground truth for {len(ground_truth)} songs")
    return ground_truth
                     


def _get_artifact_features_for_window(features: dict, start_s: float, end_s: float) -> dict[str]:
    hop = features["hop"]
    sample_rate = features["sample_rate"]
    n_frames = features["n_frames"]

    start_frame = max(0, min(int(start_s * sample_rate / hop), n_frames - 1))
    end_frame   = max(start_frame + 1, min(int(end_s * sample_rate / hop) + 1, n_frames))

    result = {}
    for key in (
        "artifact_rms", "vocal_rms", "artifact_to_signal_ratio",
        "spectral_centroid", "spectral_flatness",
    ):
        values = features[key][start_frame:end_frame]
        result[key] = float(np.mean(values)) if values else 0.0

    return result

def _get_word_error(reference: str, hypothesis: str) -> dict:
    ouput = jiwer.process_words(reference, hypothesis)
    word_errors = {}

    for chunk in ouput.alignments:
        for op in chunk:
            if op.type == "equal":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    word_errors[i] = 'correct'
            if op.type == "insert":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    word_errors[i] = "insertion"
            if op.type == "delete":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    word_errors[i] = "deletion"
            if op.type == "substitute":
                for i in range(op.ref_start_idx, op.ref_end_idx):
                    word_errors[i] = "substitution"

        return word_errors


def build_dataset(alignment_dir: Path,features_dir: Path, results_file: Path, musbd_dir: Path,   output_dir: Path) -> None:
    alignments = _load_alignments(alignment_dir)
    features = _load_artifact_features(features_dir)
    results = _load_results(results_file)
    ground_truth = _load_ground_truth(ground_truth)

    models = sorted(set(model for _, model in results.keys()))

    common_songs = set(alignments) & set(features) & set(results)


    rows = []
    songs_processed = 0

    for song_id in sorted(common_songs):
        words = alignments[song_id]
        song_features = features[song_id]
        reference = ground_truth[song_id]

        for model_name in models:
            if (song_id, model_name) not in results:
                continue

            hypothesis = results[(song_id, model_name)]

            try:
                word_errors = _get_word_error(reference, hypothesis)
            except Exception as e:
                logger.warning(f"JiWER failed: {e}")
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
                    **artifact_features,
                })
            songs_processed += 1
            logger.info(f"Processed {song_id} ({songs_processed}/{len(common_songs)})")

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        with open(output_dir, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"Wrote {len(rows)} rows to {output_dir}")



def analyse(dataset_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(dataset_path) as f:
        for row in csv.DictReader(f):
            for field in fields:
                row[field] = float(row[field])
        rows.append(row)

    models = sorted(r["model_name"] for r in rows)

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
            q_rows = [r for r in rows if get_quartile(r["artifact_to_signal_ratio"]) == q_label]
            if not q_rows:
                continue
            n = len(q_rows)
            quartile_rows.append({
                "model":    model_name,
                "quartile": q_label,
                "n_words":  n,
                "error_rate":         round(sum(1 for r in q_rows if r["error_type"] != "correct") / n, 4),
                "deletion_rate":      round(sum(1 for r in q_rows if r["error_type"] == "deletion") / n, 4),
                "substitution_rate":  round(sum(1 for r in q_rows if r["error_type"] == "substitution") / n, 4),
                "mean_artifact_to_signal": round(float(np.mean([r["artifact_to_signal_ratio"] for r in q_rows])), 4),
            })

    with open(output_dir / "error_rates_by_artifact_quartile.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "quartile", "n_words", "error_rate",
            "deletion_rate", "substitution_rate", "mean_artifact_to_signal"
        ])
        writer.writeheader()
        writer.writerows(quartile_rows)
