"""
This assumes you have a directory structure like this:

/root
    /<ISRC>
        /vocals.wav  # Demucs processed audio
        /audio.mp3  # Original audio
        /lyrics.json

The lyrics.json file should have the following structure:
{
    "unsynced": {
        "data": "lyrics"
    }
}
"""
import argparse
import json
import os
from typing import Tuple

import whisperx
from whisperx.asr import FasterWhisperPipeline
from jiwer import wer


def split_lyrics(segments: list[dict]) -> str:
    """
    Splits lyrics into phrases based on capitalization and punctuation rules.
    Takes in WhisperX segments and preprocesses them before splitting.
    
    :param segments: List of segment dictionaries from WhisperX transcription
    :return: Reformatted text with one phrase per line
    """
    # First preprocess segments into text
    text = "\n".join(segment["text"].strip() for segment in segments)
    
    ending_punct = '.!?'
    MIN_WORDS_FOR_SPLIT = 3  # Minimum words needed before I'll to consider splitting
    
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    reformatted_lines = []
    
    for line in lines:
        sentences = line.replace('. ', '.|').split('|')
        
        for sentence in sentences:
            cleaned = sentence.strip()
            
            if not cleaned:
                continue
                
            while cleaned and cleaned[-1] in ending_punct:
                cleaned = cleaned[:-1].strip()
            
            phrases = []
            last_idx = 0
            
            for i in range(1, len(cleaned)):
                if cleaned[i-1] == ' ' and cleaned[i].isupper():
                    # For I'll/I'm, check if there's a complete phrase before it
                    if cleaned[i] == 'I' and i + 1 < len(cleaned) and cleaned[i + 1] == "'":
                        prev_text = cleaned[last_idx:i].strip()
                        if len(prev_text.split()) >= MIN_WORDS_FOR_SPLIT:
                            phrases.append(prev_text)
                            last_idx = i
                    # Don't split on standalone 'I'
                    elif cleaned[i] == 'I' and (i + 1 == len(cleaned) or cleaned[i + 1] == ' '):
                        continue
                    # Split on all other capital letters
                    else:
                        phrases.append(cleaned[last_idx:i].strip())
                        last_idx = i
            
            final_phrase = cleaned[last_idx:].strip()
            if final_phrase:
                phrases.append(final_phrase)
            
            if not phrases:
                phrases = [cleaned]
            
            reformatted_lines.extend(phrase for phrase in phrases if phrase)
    
    return '\n'.join(reformatted_lines)


def transcribe_audio(model: FasterWhisperPipeline, audio_path: str) -> Tuple[str, str]:
    """
    Transcribes an audio file using WhisperX and returns the formatted text.

    :param model: The WhisperX model to use for transcription.
    :param audio_path: The path to the audio file to transcribe.
    :return: The transcribed and formatted text.
    """
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16)
    return split_lyrics(result["segments"]), result["language"]


def calculate_wer(model: FasterWhisperPipeline, args: argparse.Namespace) -> dict:
    """
    Calculates the WER for all audio files in the given directory, grouped by language.

    :param model: The WhisperX model to use for transcription.
    :param args: The parsed arguments from argparse.
    :return: A dictionary mapping languages to lists of WER scores.
    """
    language_stats = {}  # Format: {'en': [], 'es': [], ...}
    file_name = "vocals.wav" if args.use_demucs else "audio.mp3"

    for root, dirs, files in os.walk(args.directory):
        if root == args.directory:
            continue

        isrc = os.path.basename(root)

        if file_name in files and "lyrics.json" in files:
            try:
                # Get reference text from lyrics.json
                with open(os.path.join(root, "lyrics.json"), "r") as f:
                    lyrics_data = json.load(f)
                reference_text = lyrics_data["unsynced"]["data"]

                # Transcribe audio
                audio_path = os.path.join(root, file_name)
                hypothesis_text, language = transcribe_audio(model, audio_path)

                # Initialize language entry if not exists
                if language not in language_stats:
                    language_stats[language] = []

                model_fs = args.model.split("/")[-1]

                # Save hypothesis and language in a single JSON file
                with open(os.path.join(root, model_fs + "_results.json"), "w") as f:
                    json.dump({
                        "hypothesis": hypothesis_text,
                        "language": language
                    }, f, indent=2)

                # Calculate WER
                removed_double_newlines = reference_text.replace("\n\n", "\n")
                calculated_wer = wer(removed_double_newlines, hypothesis_text)
                
                # Store WER score
                language_stats[language].append(calculated_wer)

                print(f"Processed {isrc} - Language: {language} - WER: {calculated_wer:.4f}")

            except Exception as e:
                print(f"Error processing {isrc}: {str(e)}")
                continue

    return language_stats


def disable_vad(model: FasterWhisperPipeline):
    """
    Disables VAD by setting minimal onset/offset thresholds.
    
    :param model: The WhisperX model to modify
    """
    model._vad_params["vad_onset"] = 0.001
    model._vad_params["vad_offset"] = 0.001


def calculate_wer_both_modes(model: FasterWhisperPipeline, args: argparse.Namespace) -> tuple[dict, dict]:
    """
    Calculates the WER for all audio files with and without VAD.

    :param model: The WhisperX model to use for transcription.
    :param args: The parsed arguments from argparse.
    :return: Two dictionaries of language statistics (with VAD, without VAD).
    """
    print("\nCalculating WER with VAD enabled...")
    stats_vad = calculate_wer(model, args)
    
    print("\nCalculating WER with VAD disabled...")
    disable_vad(model)
    stats_no_vad = calculate_wer(model, args)
    
    return stats_vad, stats_no_vad


def calculate_wer_without_outliers(wer_scores: list[float]) -> tuple[list[float], float]:
    """
    Removes outliers from WER scores using the IQR method and calculates the average.
    
    :param wer_scores: List of WER scores
    :return: Tuple of (filtered scores, average WER without outliers)
    """
    if not wer_scores:
        return [], 0.0
        
    # Calculate Q1, Q3, and IQR
    sorted_scores = sorted(wer_scores)
    n = len(sorted_scores)
    q1_idx = int(n * 0.25)
    q3_idx = int(n * 0.75)
    
    q1 = sorted_scores[q1_idx]
    q3 = sorted_scores[q3_idx]
    iqr = q3 - q1
    
    # Define bounds for outliers (1.5 * IQR)
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    
    # Filter out outliers
    filtered_scores = [score for score in wer_scores if lower_bound <= score <= upper_bound]
    avg_wer = sum(filtered_scores) / len(filtered_scores) if filtered_scores else 0
    
    return filtered_scores, avg_wer


def print_language_statistics(language_stats: dict, mode: str):
    """
    Prints WER statistics for each language, starting with overall averages.
    
    :param language_stats: Dictionary mapping languages to lists of WER scores
    :param mode: String indicating the mode (VAD or No VAD)
    """
    print(f"\n{mode} Results:")
    
    # Calculate overall statistics
    all_scores = []
    total_samples = 0
    for scores in language_stats.values():
        all_scores.extend(scores)
        total_samples += len(scores)
    
    # Print overall statistics
    print("\nOverall Statistics:")
    avg_wer = sum(all_scores) / len(all_scores) if all_scores else 0
    filtered_scores, avg_wer_no_outliers = calculate_wer_without_outliers(all_scores)
    print(f"Total samples: {total_samples}")
    print(f"Average WER (with outliers): {avg_wer:.4f}")
    print(f"Average WER (without outliers): {avg_wer_no_outliers:.4f}")
    print(f"Removed {len(all_scores) - len(filtered_scores)} outliers")
    
    # Print per-language statistics
    print("\nPer-Language Statistics:")
    for language, wer_scores in sorted(language_stats.items()):
        count = len(wer_scores)
        percentage = (count / total_samples) * 100
        
        # Calculate statistics with and without outliers
        avg_wer = sum(wer_scores) / count if wer_scores else 0
        filtered_scores, avg_wer_no_outliers = calculate_wer_without_outliers(wer_scores)
        
        print(f"\nLanguage: {language}")
        print(f"Sample count: {count} ({percentage:.1f}% of total)")
        print(f"Average WER (with outliers): {avg_wer:.4f}")
        print(f"Average WER (without outliers): {avg_wer_no_outliers:.4f}")
        print(f"Individual WERs: {wer_scores}")
        print(f"Filtered WERs: {filtered_scores}")
        print(f"Removed {len(wer_scores) - len(filtered_scores)} outliers")


def main():
    """
    Calculate average WER for a directory of audio files using WhisperX.
    """
    parser = argparse.ArgumentParser(
        description="Calculate average WER using WhisperX"
    )
    parser.add_argument(
        "--directory", type=str, required=True, help="Directory containing audio files"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Whisper model name (e.g., 'large-v2')"
    )
    parser.add_argument(
        "--use_demucs",
        action="store_true",
        help="Use demucs processed files instead of original MP3s",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default="float16",
        choices=["float16", "int8"],
        help="Compute type for the model",
    )

    args = parser.parse_args()

    # Initialize the WhisperX model
    print(f"Initializing WhisperX model: {args.model}")
    model = whisperx.load_model(
        args.model,
        device="cuda",
        compute_type=args.compute_type
    )

    # Calculate WER for both VAD modes
    print(f"\nProcessing {'demucs' if args.use_demucs else 'original'} files...")
    stats_vad, stats_no_vad = calculate_wer_both_modes(model, args)

    # Print results
    print("\nResults:")
    print(f"Model: {args.model}")
    print(f"Processing type: {'Demucs' if args.use_demucs else 'Original'}")
    
    print_language_statistics(stats_vad, "With VAD")
    print_language_statistics(stats_no_vad, "Without VAD")


if __name__ == "__main__":
    main()
