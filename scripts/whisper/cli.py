"""
Unified CLI for audio transcription using selectable Whisper backends.

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
from typing import Any, Optional
import pathlib
from datetime import timedelta

from .schemas import TranscriptionResult, AudioProcessingType
from .openai import OpenAITranscriber
from .transformers import (
    TransformersSequentialTranscriber,
    TransformersChunkedTranscriber,
)
from .fasterwhisper import (
    FasterWhisperSequentialTranscriber,
    FasterWhisperChunkedTranscriber,
)
from .whisperx import WhisperXTranscriber


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
    def _transcribe_file(self, audio_file_path: str) -> TranscriptionResult:
        """
        Transcribes a single audio file.

        :param audio_file_path: The path to the audio file to transcribe.
        :return: The transcription result containing timing information.
        """
        pass

    @abstractmethod
    def _get_implementation_name(self) -> str:
        """
        Returns the name of the Whisper implementation.
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

    def process_single_file(self, audio_file: str, output_file: Optional[str] = None):
        """
        Transcribes a single audio file and displays the results in the terminal.

        :param audio_file: Path to the audio file to transcribe
        :param output_file: Unused parameter, kept for backwards compatibility
        """
        if not os.path.exists(audio_file):
            print(f"Audio file not found: {audio_file}")
            return

        print(
            f"\nTranscribing {audio_file} using {self.whisper_implementation_name} with model {self.model_name}..."
        )

        try:
            transcription_result = self._transcribe_file(audio_file)
        except Exception as e:
            print(f"Failed to transcribe {audio_file}: {e}")
            return

        print(
            f"\nTranscription completed in {transcription_result.transcription_time:.2f}s"
        )
        if transcription_result.alignment_model_load_time is not None:
            print(
                f"Alignment model load time: {transcription_result.alignment_model_load_time:.2f}s ({(transcription_result.alignment_model_load_time / transcription_result.transcription_time) * 100:.2f}% of total transcription time)"
            )
            print(
                f"Transcription time without alignment loading: {(transcription_result.transcription_time - transcription_result.alignment_model_load_time):.2f}s"
            )
        print("=" * 80)
        print(f"Model: {self.model_name}")
        print(f"Implementation: {self.whisper_implementation_name}")
        print(f"Language detected: {transcription_result.detected_language}")
        print(f"Audio type: {transcription_result.audio_type}")
        print("=" * 80)

        print("\nFull transcription:")
        print("-" * 80)
        print(transcription_result.full_text)
        print("-" * 80)

        print("\nSegments:")
        print("-" * 80)
        for segment in transcription_result.segments:
            start_time = str(timedelta(seconds=int(segment.start)))
            end_time = str(timedelta(seconds=int(segment.end)))
            print(f"[{start_time} -> {end_time}]")
            print(f"{segment.text}\n")

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

                try:
                    transcription_result = self._transcribe_file(audio_file)
                except Exception as e:
                    print(f"Skipping {audio_file} due to error: {e}")
                    continue

                print(
                    f"Transcribed {audio_file} in {transcription_result.transcription_time:.2f}s"
                )

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

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--directory",
        type=str,
        help="Directory containing ISRC folders, each with audio files.",
    )
    input_group.add_argument(
        "--file",
        type=str,
        help="Single audio file to transcribe.",
    )

    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        choices=[
            "openai",
            "hf-sequential",
            "hf-chunked",
            "faster-whisper-sequential",
            "faster-whisper-chunked",
            "whisperx",
        ],
        help="The transcription backend to use.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for beam search decoding (default: 5). Not supported for all backends.",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        help="Enable Voice Activity Detection (VAD) filtering. Not supported for all backends.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for chunked processing (default: 64). Not supported for all backends.",
    )

    args = parser.parse_args()

    # Initialize transcriber with directory (can be None for single file mode)
    directory = args.directory if args.directory else os.path.dirname(args.file)
    transcriber = None

    if args.backend == "openai":
        transcriber = OpenAITranscriber(model_name=args.model, directory=directory)
    elif args.backend == "hf-sequential":
        transcriber = TransformersSequentialTranscriber(
            model_name=args.model, directory=directory
        )
    elif args.backend == "hf-chunked":
        transcriber = TransformersChunkedTranscriber(
            model_name=args.model,
            directory=directory,
            batch_size=args.batch_size,
        )
    elif args.backend == "faster-whisper-sequential":
        transcriber = FasterWhisperSequentialTranscriber(
            model_name=args.model,
            directory=directory,
            beam_size=args.beam_size,
            vad_filter=args.vad,
        )
    elif args.backend == "faster-whisper-chunked":
        transcriber = FasterWhisperChunkedTranscriber(
            model_name=args.model,
            directory=directory,
            beam_size=args.beam_size,
            vad_filter=args.vad,
            batch_size=args.batch_size,
        )
    else:  # whisperx
        transcriber = WhisperXTranscriber(
            model_name=args.model,
            directory=directory,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
        )

    if args.file:
        transcriber.process_single_file(args.file)
    else:
        transcriber.process_directory()


if __name__ == "__main__":
    main()
