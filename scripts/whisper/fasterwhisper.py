import time
import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline
from .cli import BaseTranscriber
from .schemas import TranscriptionResult, Segment


class FasterWhisperBaseTranscriber(BaseTranscriber):
    """
    Base class for faster-whisper implementations with common configuration options.
    """

    def __init__(
        self,
        model_name: str,
        directory: str,
        beam_size: int = 5,
        vad_filter: bool = False,
    ):
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        super().__init__(model_name, directory)

    def _create_base_model(self, model_name: str) -> WhisperModel:
        """
        Creates a base WhisperModel with common configuration.

        :param model_name: The name of the model to load.
        :return: The configured WhisperModel instance.
        """
        compute_type = "float16" if torch.cuda.is_available() else "float32"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        return WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=None,
        )


class FasterWhisperSequentialTranscriber(FasterWhisperBaseTranscriber):
    """
    Transcribes audio files using the faster-whisper implementation with sequential processing.
    This implementation uses the basic configuration without batching for baseline comparison.
    """

    def _load_model(self, model_name: str):
        """
        Loads the faster-whisper model with basic configuration.

        :param model_name: The name of the model to load.
        :return: The loaded model.
        """
        print(f"Loading faster-whisper model (sequential): {model_name}...")
        try:
            return self._create_base_model(model_name)
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    def _transcribe_file(self, audio_file_path: str) -> TranscriptionResult:
        """
        Transcribes a single audio file using sequential processing.
        This implementation uses basic settings without batching for baseline comparison.

        :param audio_file_path: The path to the audio file to transcribe.
        :return: The transcription result.
        """
        start_time = time.time()
        segments, info = self.model.transcribe(
            audio_file_path,
            beam_size=self.beam_size,
            word_timestamps=False,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,
            initial_prompt=None,
        )

        segments_list = list(segments)
        end_time = time.time()
        transcription_time = end_time - start_time

        segments = [
            Segment(
                text=segment.text,
                start=segment.start,
                end=segment.end,
            )
            for segment in segments_list
        ]

        return TranscriptionResult(
            full_text=" ".join(segment.text for segment in segments_list),
            segments=segments,
            model_name=self.model_name,
            whisper_implementation=self._get_implementation_name(),
            audio_type=self._determine_audio_type(audio_file_path),
            transcription_time=transcription_time,
            detected_language=info.language,
        )

    def _get_implementation_name(self) -> str:
        """
        Returns the name of the implementation.

        :return: The name of the implementation.
        """
        return "faster-whisper-sequential"


class FasterWhisperChunkedTranscriber(FasterWhisperBaseTranscriber):
    """
    Transcribes audio files using the faster-whisper implementation with chunked processing
    and batching via BatchedInferencePipeline. This implementation uses batching for
    improved performance.
    """

    def __init__(
        self,
        model_name: str,
        directory: str,
        beam_size: int = 5,
        vad_filter: bool = False,
        batch_size: int = 64,
    ):
        self.batch_size = batch_size
        super().__init__(model_name, directory, beam_size, vad_filter)

    def _load_model(self, model_name: str):
        """
        Loads the faster-whisper model with optimized configuration and wraps it in
        BatchedInferencePipeline for efficient batched processing.

        :param model_name: The name of the model to load.
        :return: The loaded model wrapped in BatchedInferencePipeline.
        """
        print(f"Loading faster-whisper model (chunked): {model_name}...")
        try:
            base_model = self._create_base_model(model_name)
            return BatchedInferencePipeline(
                model=base_model,
            )
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            raise

    def _transcribe_file(self, audio_file_path: str) -> TranscriptionResult:
        """
        Transcribes a single audio file using chunked processing with batching.
        This implementation uses BatchedInferencePipeline for efficient processing.

        :param audio_file_path: The path to the audio file to transcribe.
        :return: The transcription result.
        """
        start_time = time.time()
        segments, info = self.model.transcribe(
            audio_file_path,
            batch_size=self.batch_size,
            beam_size=self.beam_size,
            word_timestamps=False,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,
            initial_prompt=None,
        )

        segments_list = list(segments)
        end_time = time.time()
        transcription_time = end_time - start_time

        segments = [
            Segment(
                text=segment.text,
                start=segment.start,
                end=segment.end,
            )
            for segment in segments_list
        ]

        return TranscriptionResult(
            full_text=" ".join(segment.text for segment in segments_list),
            segments=segments,
            model_name=self.model_name,
            whisper_implementation=self._get_implementation_name(),
            audio_type=self._determine_audio_type(audio_file_path),
            transcription_time=transcription_time,
            detected_language=info.language,
        )

    def _get_implementation_name(self) -> str:
        """
        Returns the name of the implementation.

        :return: The name of the implementation.
        """
        return "faster-whisper-chunked"
