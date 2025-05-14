import time
import torch
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
                return_timestamps=True,
            )
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    def _transcribe_file(self, audio_file_path: str) -> TranscriptionResult:
        """
        Transcribes a single audio file using sequential processing.
        The pipeline will automatically handle long-form audio using
        the sequential algorithm (default behavior).

        :param audio_file_path: The path to the audio file to transcribe.
        :return: The transcription result.
        """
        start_time = time.time()
        result = self.model(audio_file_path)
        end_time = time.time()
        transcription_time = end_time - start_time

        segments = [
            Segment(
                text=chunk["text"],
                start=chunk["timestamp"][0],
                end=chunk["timestamp"][1],
            )
            for chunk in result["chunks"]
        ]

        return TranscriptionResult(
            full_text=result["text"],
            segments=segments,
            model_name=self.model_name,
            whisper_implementation=self._get_implementation_name(),
            audio_type=self._determine_audio_type(audio_file_path),
            transcription_time=transcription_time,
            detected_language=result["language"],
        )

    def _get_implementation_name(self) -> str:
        """
        Returns the name of the implementation.

        :return: The name of the implementation.
        """
        return "hf-sequential"


class TransformersChunkedTranscriber(BaseTranscriber):
    """
    Transcribes audio files using HuggingFace Transformers Whisper implementation
    with chunked processing for long audio files. Optimized for H200 GPUs with 140GB VRAM.
    """

    def __init__(self, model_name: str, directory: str, batch_size: int = 64):
        """
        Initialize the chunked transcriber.

        :param model_name: Name of the model to load
        :param directory: Working directory
        :param batch_size: Batch size for processing (default: 64 for H200 GPUs)
        """
        self.batch_size = batch_size
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
                return_timestamps=True,
            )
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    def _transcribe_file(self, audio_file_path: str) -> TranscriptionResult:
        """
        Transcribes a single audio file using chunked processing.
        Explicitly configures the pipeline to use chunked processing
        with specified batch size.

        :param audio_file_path: The path to the audio file to transcribe.
        :return: The transcription result.
        """
        start_time = time.time()
        result = self.model(
            audio_file_path,
            chunk_length_s=30,
            batch_size=self.batch_size,
        )
        end_time = time.time()
        transcription_time = end_time - start_time

        segments = [
            Segment(
                text=chunk["text"],
                start=chunk["timestamp"][0],
                end=chunk["timestamp"][1],
            )
            for chunk in result["chunks"]
        ]

        return TranscriptionResult(
            full_text=result["text"],
            segments=segments,
            model_name=self.model_name,
            whisper_implementation=self._get_implementation_name(),
            audio_type=self._determine_audio_type(audio_file_path),
            transcription_time=transcription_time,
            detected_language=result["language"],
        )

    def _get_implementation_name(self) -> str:
        """
        Returns the name of the implementation.

        :return: The name of the implementation.
        """
        return "hf-chunked"
