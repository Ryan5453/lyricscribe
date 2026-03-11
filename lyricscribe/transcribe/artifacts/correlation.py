import csv
import json
import logging
from pathlib import Path


import jiwer
import numpy as np


logger = logging.getLogger(__name__)


def _load_attributes(alignments_dir: Path) -> dict[str, list[dict]]:
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
                     


def _get_artifacts_features_for_window(features: dict, start_s: float, end_s: float) -> dict[str]:
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

            
