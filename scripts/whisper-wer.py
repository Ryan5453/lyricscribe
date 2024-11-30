"""
This script calculates Word Error Rate (WER) scores for different Whisper model variants.
It expects a directory structure like this:

/root
    /<ISRC>
        /lyrics.json  # Contains reference lyrics
        /large-v1_results.json  # Whisper transcription results
        /large-v1_orig_vad_results.json
        /large-v1_demucs_novad_results.json
        /large-v1_demucs_vad_results.json
        # ... and similar files for other model variants

The lyrics.json file should have the following structure:
{
    "unsynced": {
        "data": "lyrics"
    }
}

And the results.json files should have the following structure:
{
    "segments": [
        {
            "text": "segment text",
            "start": 0.0,
            "end": 0.0
        }
    ]
}
"""

import os
import json
from jiwer import wer
import numpy as np
from typing import Dict, List
import argparse
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg


def load_reference_lyrics(folder_path: str) -> str:
    """
    Load reference lyrics from lyrics.json

    :param folder_path: Path to the folder containing lyrics.json
    :return: The reference lyrics text
    """
    lyrics_path = os.path.join(folder_path, 'lyrics.json')
    with open(lyrics_path, 'r', encoding='utf-8') as f:
        lyrics_data = json.load(f)
    return lyrics_data['unsynced']['data']


def load_hypothesis(file_path: str) -> str:
    """
    Load hypothesis text from JSON result file

    :param file_path: Path to the Whisper results JSON file
    :return: The concatenated transcription text
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return '\n'.join(segment['text'].strip() for segment in data['segments'])


def remove_outliers(scores: List[float]) -> List[float]:
    """
    Remove outliers using IQR method

    :param scores: List of WER scores
    :return: List of scores with outliers removed
    """
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return [x for x in scores if lower_bound <= x <= upper_bound]


def calculate_wer_scores(root_path: str) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Calculate WER scores for all model variants, grouped by language

    :param root_path: Path to the root directory containing ISRC folders
    :return: Nested dictionary: model -> language -> score_type -> scores
    """
    results = {}
    
    # Define model variants
    models = {
        'large-v1': 'Whisper Large v1',
        'large-v2': 'Whisper Large v2',
        'large-v3': 'Whisper Large v3',
        'faster-whisper-large-v3-turbo-ct2': 'Whisper Large v3 Turbo'
    }
    
    variants = [
        ('results.json', ''),
        ('orig_vad_results.json', 'with VAD'),
        ('demucs_novad_results.json', 'with Demucs'),
        ('demucs_vad_results.json', 'with Demucs + VAD')
    ]

    # Iterate through ISRC folders
    for isrc_folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, isrc_folder)
        if not os.path.isdir(folder_path):
            continue

        try:
            # Load reference lyrics
            lyrics_path = os.path.join(folder_path, 'lyrics.json')
            with open(lyrics_path, 'r', encoding='utf-8') as f:
                lyrics_data = json.load(f)
            reference_text = lyrics_data['unsynced']['data']
            
            # Process each model variant
            for model_key, model_name in models.items():
                for variant_file, variant_name in variants:
                    full_pattern = f"{model_key}_{variant_file}"
                    result_path = os.path.join(folder_path, full_pattern)
                    
                    if os.path.exists(result_path):
                        # Load results and get language
                        with open(result_path, 'r', encoding='utf-8') as f:
                            results_data = json.load(f)
                        language = results_data.get('language', 'unknown').lower()
                        if language not in ['english', 'spanish']:
                            language = 'other'
                            
                        # Get hypothesis text
                        hypothesis = '\n'.join(segment['text'].strip() for segment in results_data['segments'])
                        score = wer(reference_text, hypothesis)
                        
                        model_variant = f"{model_name}"
                        if variant_name:
                            model_variant = f"└─ {variant_name}"
                        
                        # Initialize nested dictionaries if they don't exist
                        if model_variant not in results:
                            results[model_variant] = {'english': [], 'spanish': [], 'other': []}
                        
                        results[model_variant][language].append(score)
                        
        except Exception as e:
            print(f"Error processing {isrc_folder}: {str(e)}")
    
    return results


def print_results(results: Dict[str, Dict[str, List[float]]]):
    """
    Print WER scores in table format
    """
    print("| Model | WER Type | Average | English | Spanish |")
    print("| ---------------------- | --------- | --------- | --------- | --------- |")
    
    for model in sorted(results.keys()):
        # Calculate raw scores
        all_scores = []
        eng_scores = results[model]['english']
        spa_scores = results[model]['spanish']
        
        all_scores.extend(eng_scores + spa_scores)
        
        # Calculate averages
        raw_avg = np.mean(all_scores) if all_scores else 0
        raw_eng = np.mean(eng_scores) if eng_scores else 0
        raw_spa = np.mean(spa_scores) if spa_scores else 0
        
        # Calculate filtered scores
        filtered_all = remove_outliers(all_scores)
        filtered_eng = remove_outliers(eng_scores)
        filtered_spa = remove_outliers(spa_scores)
        
        filtered_avg = np.mean(filtered_all) if filtered_all else 0
        filtered_eng = np.mean(filtered_eng) if filtered_eng else 0
        filtered_spa = np.mean(filtered_spa) if filtered_spa else 0
        
        # Print results
        print(f"| {model} | Raw | {raw_avg:.1f} | {raw_eng:.1f} | {raw_spa:.1f} |")
        print(f"| | Filtered‡ | {filtered_avg:.1f} | {filtered_eng:.1f} | {filtered_spa:.1f} |")


def main():
    """
    Calculate WER scores and print results
    """
    parser = argparse.ArgumentParser(description="Calculate WER scores for Whisper variants")
    parser.add_argument(
        "--directory", type=str, required=True,
        help="Directory containing ISRC folders with transcriptions"
    )
    
    args = parser.parse_args()
    
    # Calculate WER scores
    results = calculate_wer_scores(args.directory)
    
    # Print results
    print_results(results)


if __name__ == "__main__":
    main()