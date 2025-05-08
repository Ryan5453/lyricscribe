import whisper
import time
from typing import Optional, Tuple
from .cli import BaseTranscriber
from .schemas import TranscriptionResult, Segment


class OpenAITranscriber(BaseTranscriber):
    """
    Transcribes audio files using OpenAI's Whisper implementation.
    """

    def _load_model(self, model_name: str):
        """
        Loads the OpenAI Whisper model.

        :param model_name: The name of the model to load.
        :return: The loaded model.
        """
        available_models = whisper.available_models()
        if model_name not in available_models:
            print(f"Model {model_name} not found. Available models: {available_models}")
            raise ValueError(f"Model {model_name} not found.")
        print(f"Loading OpenAI Whisper model: {model_name}...")
        return whisper.load_model(model_name)

    def _transcribe_file(
        self, audio_file_path: str
    ) -> Tuple[Optional[TranscriptionResult], float]:
        """
        Transcribes a single audio file using the loaded OpenAI model.

        :param audio_file_path: The path to the audio file to transcribe.
        :return: A tuple containing the transcription result and the transcription time in seconds.
        """
        start_time = time.time()
        try:
            result = self.model.transcribe(audio_file_path)
            end_time = time.time()
            transcription_time = end_time - start_time

            # Convert OpenAI Whisper segments to our schema
            segments = [
                Segment(
                    text=segment["text"], start=segment["start"], end=segment["end"]
                )
                for segment in result["segments"]
            ]

            # Create our standardized result
            transcription_result = TranscriptionResult(
                full_text=result["text"],
                segments=segments,
                model_name=self.model_name,
                whisper_implementation=self._get_implementation_name(),
                audio_type=self._determine_audio_type(audio_file_path),
                transcription_time=transcription_time,
                language=result["language"],
                source_file=audio_file_path,
            )

            return transcription_result, transcription_time
        except Exception as e:
            print(f"Error transcribing {audio_file_path} with OpenAI Whisper: {e}")
            return None, 0

    def _get_implementation_name(self) -> str:
        """
        Returns the name of the implementation.

        :return: The name of the implementation.
        """
        return "OpenAI"
