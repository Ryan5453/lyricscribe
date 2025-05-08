"""
Unified CLI for audio transcription using selectable Whisper backends (OpenAI or HuggingFace Transformers).

This script assumes a directory structure like this:

/root
    /<ISRC>
        /audio.mp3  # Original audio
        ... # Other processed versions (e.g., .wav)
"""

import argparse
import glob
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple
import pathlib

from .schemas import TranscriptionResult, AudioProcessingType
from .openai import OpenAITranscriber
from .transformers import (
    TransformersSequentialTranscriber,
    TransformersChunkedTranscriber,
)


class BaseTranscriber(ABC):
    """
    Abstract base class for audio transcription.
    """

    def __init__(self, model_name: str, directory: str):
        self.model_name = model_name
        self.directory = directory
        self.model = self._load_model(model_name)
        self.whisper_implementation_name = self._get_implementation_name()

    @abstractmethod
    def _load_model(self, model_name: str) -> Any:
        """
        Attempts to load the transcription model.

        :param model_name: The name of the model to load.
        :return: The loaded model.
        """
        pass

    @abstractmethod
    def _transcribe_file(
        self, audio_file_path: str
    ) -> Tuple[Optional[TranscriptionResult], float]:
        """
        Transcribes a single audio file.

        :param audio_file_path: The path to the audio file to transcribe.
        :return: A tuple containing the transcription result and the transcription time in seconds.
        """
        pass

    @abstractmethod
    def _get_implementation_name(self) -> str:
        """
        Returns the name of the Whisper implementation (e.g., 'openai', 'transformers').
        """
        pass

    def _determine_audio_type(self, file_path: str) -> AudioProcessingType:
        """
        Determines the type of audio processing based on the file path.

        :param file_path: Path to the audio file
        :return: AudioProcessingType enum value
        """
        path = pathlib.Path(file_path)
        stem = path.stem.lower()

        if stem == "audio":
            return AudioProcessingType.REGULAR
        elif stem == "demucs_base":
            return AudioProcessingType.DEMUCS_BASE
        elif stem == "demucs_ft":
            return AudioProcessingType.DEMUCS_FT
        elif stem == "spleeter_11":
            return AudioProcessingType.SPLEETER_11
        elif stem == "spleeter_16":
            return AudioProcessingType.SPLEETER_16

        # Default to REGULAR if no match (shouldn't happen with our file naming)
        return AudioProcessingType.REGULAR

    def process_directory(self):
        """
        Processes all audio files in the specified directory structure.
        It looks for ISRC subfolders, then for .wav or .mp3 files within them.
        Transcription results are saved to 'transcription_results.jsonl' in each ISRC folder.
        """
        isrc_folders = [
            f
            for f in os.listdir(self.directory)
            if os.path.isdir(os.path.join(self.directory, f))
        ]

        if not isrc_folders:
            print(f"No ISRC subdirectories found in {self.directory}")
            return

        for isrc_folder in isrc_folders:
            isrc_path = os.path.join(self.directory, isrc_folder)

            audio_files = glob.glob(os.path.join(isrc_path, "*.wav"))
            audio_files.extend(glob.glob(os.path.join(isrc_path, "*.mp3")))

            if not audio_files:
                print(f"No audio files found in {isrc_path}")
                continue

            for audio_file in audio_files:
                print(
                    f"Transcribing {audio_file} using {self.whisper_implementation_name} with model {self.model_name}..."
                )

                transcription_result, transcription_time = self._transcribe_file(
                    audio_file
                )

                if transcription_result is None:
                    print(
                        f"Skipping {audio_file} due to transcription error or explicit skip."
                    )
                    continue

                print(f"Transcribed {audio_file} in {transcription_time:.2f}s")

                results_file = os.path.join(isrc_path, "transcription_results.jsonl")
                with open(results_file, "a") as f:
                    f.write(json.dumps(transcription_result.model_dump()) + "\n")

                print(f"Saved results to {results_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Whisper model to use (e.g., 'tiny', 'base' for OpenAI; 'openai/whisper-large-v3' for Transformers).",
    )
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory containing ISRC folders, each with audio files.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=["openai", "hf-sequential", "hf-chunked"],
        help="The transcription backend to use: 'openai', 'hf-sequential', or 'hf-chunked'.",
    )

    args = parser.parse_args()
    transcriber = None

    if args.backend == "openai":
        transcriber = OpenAITranscriber(model_name=args.model, directory=args.directory)
    elif args.backend == "hf-sequential":
        transcriber = TransformersSequentialTranscriber(
            model_name=args.model, directory=args.directory
        )
    else:  # hf-chunked
        transcriber = TransformersChunkedTranscriber(
            model_name=args.model, directory=args.directory
        )

    transcriber.process_directory()


if __name__ == "__main__":
    main()
