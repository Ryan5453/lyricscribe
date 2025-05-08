from pydantic import BaseModel
from typing import List
from enum import Enum
from datetime import timedelta


class AudioProcessingType(str, Enum):
    """Enum for different types of audio processing"""

    REGULAR = "regular"  # Original audio.mp3
    DEMUCS_BASE = "demucs_base"  # Demucs base model
    DEMUCS_FT = "demucs_ft"  # Demucs finetuned model
    SPLEETER_11 = "spleeter_11"  # Spleeter 11kHz model
    SPLEETER_16 = "spleeter_16"  # Spleeter 16kHz model


class Segment(BaseModel):
    """A segment of transcribed text with timing information"""

    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds


class TranscriptionResult(BaseModel):
    """Complete transcription result including metadata and segmented text"""

    full_text: str
    segments: List[Segment]
    model_name: str
    whisper_implementation: str
    audio_type: AudioProcessingType
    transcription_time: float  # in seconds
    language: str
    source_file: str

    def get_transcription_duration(self) -> timedelta:
        """Returns the transcription time as a timedelta object"""
        return timedelta(seconds=self.transcription_time)
