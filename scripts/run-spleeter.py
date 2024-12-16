"""
This assumes you have a directory structure like this:

/root
    /<ISRC>
        /audio.mp3  # Original audio
"""

import argparse
import os
from typing import Tuple

from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter


def load_and_separate_audio(
    separator: Separator,
    audio_adapter: AudioAdapter,
    input_path: str
) -> Tuple[dict, int]:
    """
    Loads and separates an audio file using Spleeter.

    :param separator: The Spleeter separator instance
    :param audio_adapter: The audio adapter for loading/saving files
    :param input_path: Path to the input audio file
    :return: Tuple of (separated sources, sample rate)
    """
    waveform, sample_rate = audio_adapter.load(
        input_path,
        sample_rate=separator._sample_rate
    )
    sources = separator.separate(waveform)
    return sources, sample_rate


def extract_vocals(root_path: str, isrc: str) -> None:
    """
    Extracts vocals from an audio file using Spleeter.

    :param root_path: Root directory containing ISRC folders
    :param isrc: ISRC identifier for the current folder
    """
    input_path = os.path.join(root_path, isrc, 'audio.mp3')
    output_path = os.path.join(root_path, isrc, 'spleeter.wav')
    
    separator = Separator('spleeter:2stems')
    audio_adapter = AudioAdapter.default()
    
    sources, sample_rate = load_and_separate_audio(
        separator,
        audio_adapter,
        input_path
    )
    
    audio_adapter.save(
        output_path,
        sources['vocals'],
        sample_rate,
        'wav',
        '128k'
    )

def process_files(root_path: str):
    """
    Processes all audio files in the given directory.

    :param root_path: Root directory containing ISRC folders
    """
    print("\nProcessing files with Spleeter...")
    print("-----------------------------------------------------")
    
    for root, dirs, files in os.walk(root_path):
        if root == root_path:
            continue
            
        isrc = os.path.basename(root)
        
        try:
            extract_vocals(root_path, isrc)
            print(f"Successfully processed {isrc}")
        except Exception as e:
            print(f"Error processing {isrc}: {str(e)}")


def main():
    """
    Extract vocals from audio files using Spleeter.
    """
    parser = argparse.ArgumentParser(
        description='Extract vocals from audio files using Spleeter'
    )
    parser.add_argument(
        '--directory',
        type=str,
        required=True,
        help='Directory containing ISRC folders'
    )
    
    args = parser.parse_args()
    process_files(args.directory)


if __name__ == "__main__":
    main()
