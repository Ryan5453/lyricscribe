import time
import torch
import whisperx
import gc
from typing import Any
from .cli import BaseTranscriber
from .schemas import TranscriptionResult, Segment


class WhisperXTranscriber(BaseTranscriber):
    """
    Transcribes audio files using WhisperX implementation.
    """

    def __init__(
        self,
        model_name: str,
        directory: str,
        batch_size: int = 16,
        beam_size: int = 5,
    ):
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        super().__init__(model_name, directory)

    def _cleanup_models(self, *models: Any) -> None:
        """
        Cleanup GPU memory by deleting models and running garbage collection.

        :param models: Variable number of models to delete
        """
        for model in models:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self, model_name: str):
        """
        Loads the WhisperX model.

        :param model_name: The name of the model to load.
        :return: The loaded model.
        """
        print(f"Loading WhisperX model: {model_name}...")
        try:
            return whisperx.load_model(
                model_name,
                self.device,
                compute_type="float16",
                asr_options={"beam_size": self.beam_size},
            )
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    def _transcribe_file(self, audio_file_path: str) -> TranscriptionResult:
        """
        Transcribes a single audio file using WhisperX's VAD Cut & Merge
        strategy for efficient batched processing, followed by forced
        alignment for accurate word-level timestamps.

        :param audio_file_path: The path to the audio file to transcribe.
        :return: The transcription result.
        """
        start_time = time.time()

        # 1. Load audio and apply VAD Cut & Merge
        audio = whisperx.load_audio(audio_file_path)
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        language = result["language"]

        # 2. Align whisper output - track alignment model loading time
        alignment_load_start = time.time()
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=self.device
        )
        alignment_load_time = time.time() - alignment_load_start

        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        # Cleanup alignment models
        self._cleanup_models(align_model)

        end_time = time.time()
        total_time = end_time - start_time

        segments = [
            Segment(
                text=segment["text"],
                start=segment["start"],
                end=segment["end"],
            )
            for segment in result["segments"]
        ]

        return TranscriptionResult(
            full_text=" ".join(segment["text"] for segment in result["segments"]),
            segments=segments,
            model_name=self.model_name,
            whisper_implementation=self._get_implementation_name(),
            audio_type=self._determine_audio_type(audio_file_path),
            transcription_time=total_time,
            alignment_model_load_time=alignment_load_time,
            detected_language=language,
        )

    def _get_implementation_name(self) -> str:
        """
        Returns the name of the implementation.

        :return: The name of the implementation.
        """
        return "whisperx"
