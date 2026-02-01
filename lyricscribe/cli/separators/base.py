import os
import time
from abc import ABC, abstractmethod
from typing import Any

class BaseSeparator(ABC):
    """
    Abstract base class for audio separation.

    Assumes a directory structure like:
    /
        /<isrc>  (e.g., ISRC subfolder)
            /audio.mp3 (or other input filename)
    """

    def __init__(
        self,
        model_name: str,
        output_filename_suffix: str,
        directory: str,
        input_filename: str = "audio.mp3",
    ):
        self.model_name = model_name
        self.output_filename_suffix = output_filename_suffix
        self.directory = directory
        self.input_filename = input_filename
        self.implementation_name = self._get_implementation_name()
        print(
            f"Processing files with {self.implementation_name} using model {self.model_name}..."
        )
        self.model = self._load_model(model_name)

    @abstractmethod
    def _load_model(self, model_name: str) -> Any:
        """
        Loads the separation model.

        :param model_name: The name of the model to load.
        :return: The loaded model.
        """
        pass

    @abstractmethod
    def _separate_file(self, input_path: str, output_path: str) -> float:
        """
        Separates a single audio file and saves the vocal stem.

        :param input_path: The path to the input audio file.
        :param output_path: The path to save the separated vocal stem.
        :return: The time taken for separation in seconds.
        """
        pass

    @abstractmethod
    def _get_implementation_name(self) -> str:
        """
        Get the name of the separator implementation (e.g., "Demucs", "Spleeter").

        :return: The name of the separator implementation.
        """
        pass

    def _get_output_filename(self, original_filename_stem: str) -> str:
        """
        Generates the output filename using the stem of the input file and the suffix.
        Example: if input is audio.mp3 and suffix is _demucs_vocals.wav, output is audio_demucs_vocals.wav
        The user-provided output in main.py (e.g. demucs.wav) becomes the suffix here.
        """
        # This method seems unused now as output_filename_suffix is the full name.
        # If self.output_filename_suffix is just a suffix like "_vocals.wav", then this is useful.
        # Given current usage, self.output_filename_suffix is the complete target filename (e.g. "demucs_vocals.wav")
        # So this method might not be strictly needed if output_filename_suffix is always the full name.
        # However, keeping it for potential future flexibility where output_filename_suffix is a true suffix.
        return f"{original_filename_stem}{self.output_filename_suffix}"

    def process_directory(self):
        total_start_time = time.time()
        processed_files_count = 0
        total_separation_time = 0

        isrc_folders = [
            f
            for f in os.listdir(self.directory)
            if os.path.isdir(os.path.join(self.directory, f))
        ]

        if not isrc_folders:
            print(f"No ISRC subdirectories found in {self.directory}")
            return

        print(
            f"Target output filename pattern for separated files: {self.output_filename_suffix}"
        )
        print("-----------------------------------------------------")

        for isrc_folder_name in isrc_folders:
            isrc_path = os.path.join(self.directory, isrc_folder_name)
            input_audio_path = os.path.join(isrc_path, self.input_filename)

            if not os.path.exists(input_audio_path):
                print(
                    f"Input audio {self.input_filename} not found in {isrc_path}. Skipping."
                )
                continue

            output_audio_filename = self.output_filename_suffix
            output_audio_path = os.path.join(isrc_path, output_audio_filename)

            try:
                print(f"Processing {input_audio_path}...")
                separation_time = self._separate_file(
                    input_audio_path, output_audio_path
                )
                processed_files_count += 1
                total_separation_time += separation_time
                print(
                    f"Successfully processed {isrc_folder_name} -> {output_audio_filename} in {separation_time:.2f}s"
                )
            except Exception as e:
                print(f"Error processing {input_audio_path}: {e}")

        total_run_time = time.time() - total_start_time
        print(f"\nFinished processing for {self.implementation_name}.")
        print(f"Processed {processed_files_count} files in {total_run_time:.2f}s.")
        if processed_files_count > 0:
            avg_time = total_separation_time / processed_files_count
            print(f"Average separation time per file: {avg_time:.2f}s")
        print("=====================================================")
