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

import argparse
import json
import os
from collections import Counter
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from jiwer import wer
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def load_reference_lyrics(folder_path: str) -> str:
    """
    Load reference lyrics from lyrics.json

    :param folder_path: Path to the folder containing lyrics.json
    :return: The reference lyrics text
    """
    lyrics_path = os.path.join(folder_path, "lyrics.json")
    with open(lyrics_path, "r", encoding="utf-8") as f:
        lyrics_data = json.load(f)
    return lyrics_data["unsynced"]["data"]


def load_hypothesis(file_path: str) -> Tuple[str, str]:
    """
    Load hypothesis text from JSON result file

    :param file_path: Path to the Whisper results JSON file
    :return: The concatenated transcription text and the predicted language
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return (
        "\n".join(segment["text"].strip() for segment in data["segments"]),
        data["language"],
    )


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


def calculate_wer_scores(
    root_path: str,
) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    """
    Calculate WER scores for all model variants, grouped by language
    """
    results = {}
    language_counts = Counter()

    # Define model variants
    models = {
        "large-v1": "Whisper Large v1",
        "large-v2": "Whisper Large v2",
        "large-v3": "Whisper Large v3",
        "faster-whisper-large-v3-turbo-ct2": "Whisper Large v3 Turbo",
    }

    variants = [
        ("orig_novad_results.json", "Base"),
        ("orig_vad_results.json", "with VAD"),
        ("demucs_novad_results.json", "with Demucs"),
        ("demucs_vad_results.json", "with Demucs + VAD"),
    ]

    # Iterate through ISRC folders
    for isrc_folder in os.listdir(root_path):
        folder_path = os.path.join(root_path, isrc_folder)
        if not os.path.isdir(folder_path):
            continue

        # Load reference lyrics
        try:
            lyrics_path = os.path.join(folder_path, "lyrics.json")
            with open(lyrics_path, "r", encoding="utf-8") as f:
                lyrics_data = json.load(f)
            reference_text = lyrics_data["unsynced"]["data"]

            # Process each model variant
            for model_key, model_name in models.items():
                for variant_file, variant_name in variants:
                    full_pattern = f"{model_key}_{variant_file}"
                    result_path = os.path.join(folder_path, full_pattern)

                    if os.path.exists(result_path):
                        hypothesis, language = load_hypothesis(result_path)
                        score = wer(reference_text, hypothesis) * 100

                        # Use raw language code directly
                        language_counts[language] += 1

                        model_variant = f"{model_name}"
                        if variant_name != "Base":
                            model_variant = f"{model_name} └─ {variant_name}"

                        if model_variant not in results:
                            results[model_variant] = {}
                        if language not in results[model_variant]:
                            results[model_variant][language] = []

                        results[model_variant][language].append(score)

        except Exception as e:
            print(f"Error processing {isrc_folder}: {str(e)}")

    return results, language_counts


def print_results(results: Dict[str, Dict[str, List[float]]], language_counts: Counter):
    """
    Print WER scores in table format for top 4 languages
    """
    # Get top 4 languages by count
    sorted_languages = [lang for lang, _ in language_counts.most_common(4)]

    # Create header
    header = (
        "| Model | WER Type |"
        + " Average |"
        + "".join(f" {lang} |" for lang in sorted_languages)
    )
    print(header)
    print("| " + "-" * (len(header) - 3) + " |")

    # Group models by their base name
    model_groups = {}
    for model in results.keys():
        base_name = model.split("└─")[0].strip()
        if base_name not in model_groups:
            model_groups[base_name] = []
        model_groups[base_name].append(model)

    # Sort base models to ensure consistent order
    sorted_base_models = sorted(
        model_groups.keys(),
        key=lambda x: (
            "Turbo" in x,  # Put Turbo models last
            x,  # Then sort alphabetically
        ),
    )

    for base_model in sorted_base_models:
        # First print the base model
        base_results = [m for m in model_groups[base_model] if "└─" not in m]
        if base_results:
            # Print base model results
            model = base_results[0]
            # ... print logic for base model ...

            # Calculate scores for base model
            lang_scores = {
                lang: results[model].get(lang, []) for lang in sorted_languages
            }

            all_scores = []
            for scores in lang_scores.values():
                all_scores.extend(scores)

            raw_avg = np.mean(all_scores) if all_scores else 0
            raw_langs = {
                lang: np.mean(scores) if scores else 0
                for lang, scores in lang_scores.items()
            }

            filtered_all = remove_outliers(all_scores)
            filtered_avg = np.mean(filtered_all) if filtered_all else 0
            filtered_langs = {
                lang: np.mean(remove_outliers(scores)) if scores else 0
                for lang, scores in lang_scores.items()
            }

            raw_line = f"| {model:<24} | Raw       | {raw_avg:.2f} |" + "".join(
                f" {raw_langs[lang]:.2f} |" for lang in sorted_languages
            )
            filtered_line = f"| {' '*24} | Filtered‡ | {filtered_avg:.2f} |" + "".join(
                f" {filtered_langs[lang]:.2f} |" for lang in sorted_languages
            )

            print(raw_line)
            print(filtered_line)

        # Then print variants in specific order
        variants = [m for m in model_groups[base_model] if "└─" in m]
        variants.sort(
            key=lambda x: (
                "VAD" in x and "Demucs" not in x,  # VAD only first
                "Demucs" in x and "VAD" not in x,  # Demucs only second
                "Demucs + VAD" in x,  # Demucs + VAD last
            )
        )

        for model in variants:
            # ... existing printing logic for variants ...
            lang_scores = {
                lang: results[model].get(lang, []) for lang in sorted_languages
            }

            all_scores = []
            for scores in lang_scores.values():
                all_scores.extend(scores)

            raw_avg = np.mean(all_scores) if all_scores else 0
            raw_langs = {
                lang: np.mean(scores) if scores else 0
                for lang, scores in lang_scores.items()
            }

            filtered_all = remove_outliers(all_scores)
            filtered_avg = np.mean(filtered_all) if filtered_all else 0
            filtered_langs = {
                lang: np.mean(remove_outliers(scores)) if scores else 0
                for lang, scores in lang_scores.items()
            }

            model_name = "└─" + model.split("└─")[1].strip()
            raw_line = f"| {model_name:<24} | Raw       | {raw_avg:.2f} |" + "".join(
                f" {raw_langs[lang]:.2f} |" for lang in sorted_languages
            )
            filtered_line = f"| {' '*24} | Filtered‡ | {filtered_avg:.2f} |" + "".join(
                f" {filtered_langs[lang]:.2f} |" for lang in sorted_languages
            )

            print(raw_line)
            print(filtered_line)


def plot_wer_graphs(
    results: Dict[str, Dict[str, List[float]]],
    language_counts: Counter,
    output_file: str,
):
    """
    Generate and save WER comparison graphs

    :param results: Dictionary of WER results
    :param language_counts: Counter of language frequencies
    :param output_file: Path where the output chart should be saved
    """
    # Get top 4 languages
    top_languages = [lang for lang, _ in language_counts.most_common(4)]

    # Prepare data for plotting
    versions = ["v1", "v2", "v3", "v3 Turbo"]
    variants = ["Base", "with VAD", "with Demucs", "with Demucs + VAD"]

    # Create figure with 2x5 subplots (raw and filtered for average + top 4 languages)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))

    def get_data_for_language(lang, filtered=False):
        data = {variant: [] for variant in variants}
        for version in versions:
            for variant in variants:
                model_name = f"Whisper Large {version}"
                if version == "v3 Turbo":
                    model_name = "Whisper Large v3 Turbo"

                full_model = model_name
                if variant != "Base":
                    full_model = f"{model_name} └─ {variant}"

                scores = results.get(full_model, {}).get(lang, [])
                if filtered and scores:
                    scores = remove_outliers(scores)
                data[variant].append(np.mean(scores) if scores else 0)
        return data

    def create_subplot(ax, data, title, row_label=""):
        for label, values in data.items():
            ax.plot(versions, values, marker="o", label=label, linewidth=2)
        ax.set_title(f"{title}\n{row_label} WER Comparison", fontsize=12)
        ax.set_xlabel("Model Version", fontsize=10)
        ax.set_ylabel("Word Error Rate (WER) %", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.7)

        # Calculate y-axis limits with some padding
        all_values = [v for values in data.values() for v in values]
        min_val = min(all_values)
        max_val = max(all_values)
        padding = (max_val - min_val) * 0.1  # 10% padding
        ax.set_ylim(max(0, min_val - padding), max_val + padding)

        # Only show legend for the rightmost plot in the first row
        if ax.get_position().x0 >= 0.75 and ax.get_position().y0 >= 0.5:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
        else:
            ax.legend().set_visible(False)

    # Calculate and plot average scores (raw and filtered)
    for filtered in [False, True]:
        row = 1 if filtered else 0
        row_label = "Filtered" if filtered else "Raw"
        
        # Calculate average scores
        all_scores_data = {variant: [] for variant in variants}
        for version in versions:
            for variant in variants:
                model_name = f"Whisper Large {version}"
                if version == "v3 Turbo":
                    model_name = "Whisper Large v3 Turbo"

                full_model = model_name
                if variant != "Base":
                    full_model = f"{model_name} └─ {variant}"

                all_scores = []
                if full_model in results:
                    for lang_scores in results[full_model].values():
                        if filtered:
                            lang_scores = remove_outliers(lang_scores)
                        all_scores.extend(lang_scores)

                all_scores_data[variant].append(np.mean(all_scores) if all_scores else 0)

        # Plot average scores
        create_subplot(axes[row, 0], all_scores_data, "Average (All Languages)", row_label)

        # Plot top 4 languages
        for i, lang in enumerate(top_languages, 1):
            lang_data = get_data_for_language(lang, filtered)
            create_subplot(axes[row, i], lang_data, f"Language: {lang}", row_label)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    """
    Calculate WER scores, print results, and generate graphs
    """
    parser = argparse.ArgumentParser(
        description="Calculate WER scores for Whisper variants"
    )
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory containing ISRC folders with transcriptions",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="whisper_comparison.png",
        help="Output file path for the comparison chart (default: whisper_comparison.png)",
    )

    args = parser.parse_args()

    # Calculate WER scores
    results, language_counts = calculate_wer_scores(args.directory)

    # Print results
    print_results(results, language_counts)

    # Generate and save graphs
    plot_wer_graphs(results, language_counts, args.output)


if __name__ == "__main__":
    main()
