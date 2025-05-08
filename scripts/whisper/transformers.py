import time
import torch
from typing import Optional, Tuple
from transformers import pipeline
from .cli import BaseTranscriber
from .schemas import TranscriptionResult, Segment


class TransformersSequentialTranscriber(BaseTranscriber):
    """
    Transcribes audio files using HuggingFace Transformers Whisper implementation
    with sequential processing for long audio files.
    """

    def _load_model(self, model_name: str):
        """
        Loads the HuggingFace Whisper pipeline with sequential processing.

        :param model_name: The name of the model to load (e.g., 'openai/whisper-large-v3').
        :return: The loaded pipeline.
        """
        print(f"Loading HuggingFace Whisper model (sequential): {model_name}...")
        try:
            return pipeline(
                task="automatic-speech-recognition",
                model=model_name,
                torch_dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                return_timestamps=True,  # Enable timestamp generation
            )
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    def _transcribe_file(
        self, audio_file_path: str
    ) -> Tuple[Optional[TranscriptionResult], float]:
        """
        Transcribes a single audio file using sequential processing.
        The pipeline will automatically handle long-form audio using
        the sequential algorithm (default behavior).

        :param audio_file_path: The path to the audio file to transcribe.
        :return: A tuple containing the transcription result and the transcription time in seconds.
        """
        start_time = time.time()
        try:
            # The pipeline will automatically use sequential processing for long audio
            result = self.model(audio_file_path)
            end_time = time.time()
            transcription_time = end_time - start_time

            # Convert HuggingFace chunks to our schema
            segments = (
                [
                    Segment(
                        text=chunk["text"],
                        start=chunk["timestamp"][0],
                        end=chunk["timestamp"][1],
                    )
                    for chunk in result["chunks"]
                ]
                if "chunks" in result
                else []
            )

            # Create our standardized result
            transcription_result = TranscriptionResult(
                full_text=result["text"],
                segments=segments,
                model_name=self.model_name,
                whisper_implementation=self._get_implementation_name(),
                audio_type=self._determine_audio_type(audio_file_path),
                transcription_time=transcription_time,
                language=result.get(
                    "language", "en"
                ),  # Default to English if not provided
                source_file=audio_file_path,
            )

            return transcription_result, transcription_time
        except Exception as e:
            print(f"Error transcribing {audio_file_path} with HF Sequential: {e}")
            return None, 0

    def _get_implementation_name(self) -> str:
        """
        Returns the name of the implementation.

        :return: The name of the implementation.
        """
        return "HuggingFace-Sequential"


class TransformersChunkedTranscriber(BaseTranscriber):
    """
    Transcribes audio files using HuggingFace Transformers Whisper implementation
    with chunked processing for long audio files. Optimized for H200 GPUs with 140GB VRAM.
    """

    def __init__(self, model_name: str, directory: str):
        # Using aggressive batch size for H200 GPUs (140GB VRAM)
        # Each 30s chunk is ~480K tokens in features
        # Whisper large-v3 is ~1.5GB in FP16
        # This leaves plenty of room for attention matrices
        self.chunk_length_s = 30  # Standard Whisper chunk size
        self.batch_size = 64  # Aggressive batch size for H200
        super().__init__(model_name, directory)

    def _load_model(self, model_name: str):
        """
        Loads the HuggingFace Whisper pipeline with chunked processing configuration.

        :param model_name: The name of the model to load (e.g., 'openai/whisper-large-v3').
        :return: The loaded pipeline.
        """
        print(f"Loading HuggingFace Whisper model (chunked): {model_name}...")
        try:
            return pipeline(
                task="automatic-speech-recognition",
                model=model_name,
                torch_dtype=torch.float16,
                device="cuda" if torch.cuda.is_available() else "cpu",
                return_timestamps=True,  # Enable timestamp generation
            )
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    def _transcribe_file(
        self, audio_file_path: str
    ) -> Tuple[Optional[TranscriptionResult], float]:
        """
        Transcribes a single audio file using chunked processing.
        Explicitly configures the pipeline to use chunked processing
        with specified chunk length and batch size.

        :param audio_file_path: The path to the audio file to transcribe.
        :return: A tuple containing the transcription result and the transcription time in seconds.
        """
        start_time = time.time()
        try:
            # Explicitly use chunked processing with specified parameters
            result = self.model(
                audio_file_path,
                chunk_length_s=self.chunk_length_s,
                batch_size=self.batch_size,
            )
            end_time = time.time()
            transcription_time = end_time - start_time

            # Convert HuggingFace chunks to our schema
            segments = (
                [
                    Segment(
                        text=chunk["text"],
                        start=chunk["timestamp"][0],
                        end=chunk["timestamp"][1],
                    )
                    for chunk in result["chunks"]
                ]
                if "chunks" in result
                else []
            )

            # Create our standardized result
            transcription_result = TranscriptionResult(
                full_text=result["text"],
                segments=segments,
                model_name=self.model_name,
                whisper_implementation=self._get_implementation_name(),
                audio_type=self._determine_audio_type(audio_file_path),
                transcription_time=transcription_time,
                language=result.get(
                    "language", "en"
                ),  # Default to English if not provided
                source_file=audio_file_path,
            )

            return transcription_result, transcription_time
        except Exception as e:
            print(f"Error transcribing {audio_file_path} with HF Chunked: {e}")
            return None, 0

    def _get_implementation_name(self) -> str:
        """
        Returns the name of the implementation.

        :return: The name of the implementation.
        """
        return "HuggingFace-Chunked"
