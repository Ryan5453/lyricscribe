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

import torch
from demucs.api import Separator
from demucs.audio import AudioFile, save_audio
from demucs.pretrained import get_model


def load_and_separate_audio(
    separator: Separator, input_path: str
) -> Tuple[dict, float]:
    """
    Loads and separates an audio file using Demucs.

    :param separator: The Demucs separator instance
    :param input_path: Path to the input audio file
    :return: Tuple of (separated sources, separation time)
    """
    # Load audio
    wav = AudioFile(input_path).read()

    # Convert to torch tensor
    wav = torch.tensor(wav, dtype=torch.float32)
    if wav.dim() == 1:
        wav = wav[None]  # Add channel dimension

    start_time = time.time()
    sources = separator.separate_tensor(wav)
    separation_time = time.time() - start_time

    return sources, separation_time


def extract_vocals(
    separator: Separator,
    root_path: str,
    isrc: str,
    output_filename: str,
) -> float:
    """
    Extracts vocals from an audio file using Demucs.

    :param separator: The Demucs separator instance
    :param root_path: Root directory containing ISRC folders
    :param isrc: ISRC identifier for the current folder
    :param output_filename: Name of the output file
    :return: Time taken for separation in seconds
    """
    input_path = os.path.join(root_path, isrc, "audio.mp3")
    output_path = os.path.join(root_path, isrc, output_filename)

    sources, separation_time = load_and_separate_audio(separator, input_path)

    # Save only the vocals
    save_audio(sources["vocals"], output_path, separator.samplerate)

    return separation_time


def process_files(root_path: str, model_name: str, output_filename: str):
    """
    Processes all audio files in the given directory.

    :param root_path: Root directory containing ISRC folders
    :param model_name: Name of the Demucs model to use
    :param output_filename: Name of the output file
    """
    print("Processing files with Demucs...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {gpu_name}")
    else:
        print("Using GPU: None detected")

    cpu_info = "Unknown"
    if platform.system() == "Linux":
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    cpu_info = line.split(':')[1].strip()
                    break
    print(f"Using CPU: {cpu_info}")
    print("-----------------------------------------------------")

    # Initialize model once
    model = get_model(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    separator = Separator(model, device=device)

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
                separator, root_path, isrc, output_filename
            )
            processed += 1
            total_separation_time += separation_time
            print(f"Successfully processed {isrc}")
        except Exception as e:
            print(f"Error processing {isrc}: {str(e)}")

    total_time = time.time() - total_start
    print(f"\nProcessed {processed} files in {total_time:.2f}s")
    print(
        f"Average separation time per file: {total_separation_time/processed:.2f}s]"
    )


def main():
    """
    Extract vocals from audio files using Demucs.
    """
    parser = argparse.ArgumentParser(
        description="Extract vocals from audio files using Demucs"
    )
    parser.add_argument(
        "--directory", type=str, required=True, help="Directory containing ISRC folders"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="htdemucs",
        help="Demucs model to use (default: htdemucs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demucs.wav",
        help="Output filename (default: demucs.wav)",
    )

    args = parser.parse_args()
    process_files(args.directory, args.model, args.output)


if __name__ == "__main__":
    main()
