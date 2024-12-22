"""
This assumes you have a directory structure like this:

/root
    /<ISRC>
        /audio.mp3  # Original audio
"""

import argparse
import os
import platform
import time
from typing import Tuple

import tensorflow as tf
from spleeter.audio.adapter import AudioAdapter
from spleeter.separator import Separator


def load_and_separate_audio(
    separator: Separator, audio_adapter: AudioAdapter, input_path: str
) -> Tuple[dict, int]:
    """
    Loads and separates an audio file using Spleeter.

    :param separator: The Spleeter separator instance
    :param audio_adapter: The audio adapter for loading/saving files
    :param input_path: Path to the input audio file
    :return: Tuple of (separated sources, sample rate)
    """
    waveform, sample_rate = audio_adapter.load(
        input_path, sample_rate=separator._sample_rate
    )
    start_time = time.time()
    sources = separator.separate(waveform)
    separation_time = time.time() - start_time
    return sources, sample_rate, separation_time


def extract_vocals(
    separator: Separator,
    audio_adapter: AudioAdapter,
    root_path: str,
    isrc: str,
    output_filename: str,
) -> float:
    """
    Extracts vocals from an audio file using Spleeter.

    :param separator: The Spleeter separator instance
    :param audio_adapter: The audio adapter for loading/saving files
    :param root_path: Root directory containing ISRC folders
    :param isrc: ISRC identifier for the current folder
    :param output_filename: Name of the output file
    :return: Time taken for separation in seconds
    """
    input_path = os.path.join(root_path, isrc, "audio.mp3")
    output_path = os.path.join(root_path, isrc, output_filename)

    sources, sample_rate, separation_time = load_and_separate_audio(
        separator, audio_adapter, input_path
    )

    audio_adapter.save(output_path, sources["vocals"], sample_rate, "wav", "128k")

    return separation_time


def process_files(root_path: str, model_name: str, output_filename: str):
    """
    Processes all audio files in the given directory.

    :param root_path: Root directory containing ISRC folders
    :param model_name: Name of the Spleeter model to use
    :param output_filename: Name of the output file
    """
    print("Processing files with Spleeter...")
    print("-----------------------------------------------------")

    # Initialize model once
    separator = Separator(model_name)
    audio_adapter = AudioAdapter.default()

    total_start = time.time()
    processed = 0
    total_separation_time = 0

    for root, dirs, files in os.walk(root_path):
        if root == root_path:
            continue

        isrc = os.path.basename(root)

        try:
            print(f"Processing {isrc}...")
            separation_time = extract_vocals(
                separator, audio_adapter, root_path, isrc, output_filename
            )
            processed += 1
            total_separation_time += separation_time
            print(f"Successfully processed {isrc}")
        except Exception as e:
            print(f"Error processing {isrc}: {str(e)}")

    total_time = time.time() - total_start
    print(f"\nProcessed {processed} files in {total_time:.2f}s")
    print(f"Average separation time per file: {total_separation_time/processed:.2f}s\n")

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        gpu_details = tf.config.experimental.get_device_details(gpus[0])
        print(f"Using GPU: {gpu_details.get('device_name', 'Unknown GPU')}")

    if platform.system() == "Linux":
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    print(f"Using CPU: {line.split(':')[1].strip()}")
                    break


def main():
    """
    Extract vocals from audio files using Spleeter.
    """
    parser = argparse.ArgumentParser(
        description="Extract vocals from audio files using Spleeter"
    )
    parser.add_argument(
        "--directory", type=str, required=True, help="Directory containing ISRC folders"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="spleeter:2stems",
        help="Spleeter model to use (default: spleeter:2stems)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="spleeter.wav",
        help="Output filename (default: spleeter.wav)",
    )

    args = parser.parse_args()
    process_files(args.directory, args.model, args.output)


if __name__ == "__main__":
    main()
