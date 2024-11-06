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


def calculate_wer(model: FasterWhisperPipeline, args: argparse.Namespace) -> list[float]:
    """
    Calculates the WER for all audio files in the given directory. Optionally uses Demucs pre-processed audio.

    :param model: The WhisperX model to use for transcription.
    :param args: The parsed arguments from argparse.
    :return: A list of WER scores.
    """
    wer_scores = []
    file_name = "vocals.wav" if args.use_demucs else "audio.mp3"

    for root, dirs, files in os.walk(args.directory):
        if root == args.directory:
            continue

        isrc = os.path.basename(root)

        # Check if we have both the audio file and lyrics
        if file_name in files and "lyrics.json" in files:
            try:
                # Get reference text from lyrics.json
                with open(os.path.join(root, "lyrics.json"), "r") as f:
                    lyrics_data = json.load(f)
                reference_text = lyrics_data["unsynced"]["data"]

                # Transcribe audio
                audio_path = os.path.join(root, file_name)
                hypothesis_text, language = transcribe_audio(model, audio_path)

                # Save hypothesis text to file
                with open(os.path.join(root, args.model + "_hypothesis.txt"), "w") as f:
                    f.write(hypothesis_text)

                # Save language to file
                with open(os.path.join(root, args.model + "_language.txt"), "w") as f:
                    f.write(language)

                # Calculate WER
                removed_double_newlines = reference_text.replace("\n\n", "\n")
                calculated_wer = wer(removed_double_newlines, hypothesis_text)
                wer_scores.append(calculated_wer)

                print(f"Processed {isrc} - WER: {calculated_wer:.4f}")

            except Exception as e:
                print(f"Error processing {isrc}: {str(e)}")
                continue

    return wer_scores


def disable_vad(model: FasterWhisperPipeline):
    """
    Disables VAD by setting minimal onset/offset thresholds.
    
    :param model: The WhisperX model to modify
    """
    model._vad_params["vad_onset"] = 0.001
    model._vad_params["vad_offset"] = 0.001


def calculate_wer_both_modes(model: FasterWhisperPipeline, args: argparse.Namespace) -> tuple[list[float], list[float]]:
    """
    Calculates the WER for all audio files with and without VAD.

    :param model: The WhisperX model to use for transcription.
    :param args: The parsed arguments from argparse.
    :return: Two lists of WER scores (with VAD, without VAD).
    """
    # First calculate with VAD (default)
    print("\nCalculating WER with VAD enabled...")
    wer_scores_vad = calculate_wer(model, args)
    
    # Disable VAD and calculate again
    print("\nCalculating WER with VAD disabled...")
    disable_vad(model)
    wer_scores_no_vad = calculate_wer(model, args)
    
    return wer_scores_vad, wer_scores_no_vad


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
    wer_scores_vad, wer_scores_no_vad = calculate_wer_both_modes(model, args)

    # Calculate averages and remove outliers
    avg_wer_vad = sum(wer_scores_vad) / len(wer_scores_vad) if wer_scores_vad else 0
    avg_wer_no_vad = sum(wer_scores_no_vad) / len(wer_scores_no_vad) if wer_scores_no_vad else 0
    
    # Calculate WER without outliers
    filtered_vad, avg_wer_vad_no_outliers = calculate_wer_without_outliers(wer_scores_vad)
    filtered_no_vad, avg_wer_no_vad_no_outliers = calculate_wer_without_outliers(wer_scores_no_vad)

    # Print results
    print("\nResults:")
    print(f"Model: {args.model}")
    print(f"Processing type: {'Demucs' if args.use_demucs else 'Original'}")
    
    print("\nWith VAD:")
    print(f"Average WER (with outliers): {avg_wer_vad:.4f}")
    print(f"Average WER (without outliers): {avg_wer_vad_no_outliers:.4f}")
    print("Individual WERs:", wer_scores_vad)
    print(f"Filtered WERs (outliers removed): {filtered_vad}")
    print(f"Removed {len(wer_scores_vad) - len(filtered_vad)} outliers")
    
    print("\nWithout VAD:")
    print(f"Average WER (with outliers): {avg_wer_no_vad:.4f}")
    print(f"Average WER (without outliers): {avg_wer_no_vad_no_outliers:.4f}")
    print("Individual WERs:", wer_scores_no_vad)
    print(f"Filtered WERs (outliers removed): {filtered_no_vad}")
    print(f"Removed {len(wer_scores_no_vad) - len(filtered_no_vad)} outliers")


if __name__ == "__main__":
    main()
