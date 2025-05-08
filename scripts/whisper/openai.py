"""
This assumes you have a directory structure like this:

/root
    /<ISRC>
        /audio.mp3  # Original audio
        ... # Other processed versions
"""

import argparse
import glob
import json
import time
import os

import whisper


def main():
    """
    Transcribe audio files using OpenAI's Whisper Implementation.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe audio files using OpenAI Whisper"
    )
    parser.add_argument("--model", type=str, required=True, help="Whisper model to use")
    parser.add_argument(
        "--directory", type=str, required=True, help="Directory containing ISRC folders"
    )

    args = parser.parse_args()

    available_models = whisper.available_models()
    if args.model not in available_models:
        raise ValueError(
            f"Model {args.model} not found. Available models: {available_models}"
        )

    model = whisper.load_model(args.model)

    # Find all ISRC folders in the directory
    isrc_folders = [
        f
        for f in os.listdir(args.directory)
        if os.path.isdir(os.path.join(args.directory, f))
    ]

    for isrc_folder in isrc_folders:
        isrc_path = os.path.join(args.directory, isrc_folder)

        # Find all versions of the audio file in the ISRC folder (either .wav or .mp3)
        audio_files = glob.glob(os.path.join(isrc_path, "*.wav"))
        audio_files.extend(glob.glob(os.path.join(isrc_path, "*.mp3")))

        if not audio_files:
            print(f"No audio files found in {isrc_path}")
            continue

        for audio_file in audio_files:
            print(f"Transcribing {audio_file}...")
            start_time = time.time()
            results = model.transcribe(audio_file)
            end_time = time.time()
            print(f"Transcribed {audio_file} in {end_time - start_time:.2f}s")

            # Save results to a file in the ISRC folder
            results_file = os.path.join(isrc_path, "transcription_results.jsonl")
            with open(results_file, "a") as f:
                data = {
                    "file": audio_file,
                    "model": args.model,
                    "whisper_implementation": "openai",
                    "transcription_time": end_time - start_time,
                    "transcription": results,
                }
                json.dump(data, f)
                f.write("\n")

            print(f"Transcribed {audio_file} and saved results to {results_file}")


if __name__ == "__main__":
    main()
