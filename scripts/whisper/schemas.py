from pydantic import BaseModel
from typing import List, Optional
from enum import Enum


class AudioProcessingType(str, Enum):
    """
    Enum for different types of audio processing
    """

    REGULAR = "regular"  # Original audio.mp3
    DEMUCS_BASE = "demucs_base"  # Demucs base model
    DEMUCS_FT = "demucs_ft"  # Demucs finetuned model
    SPLEETER_11 = "spleeter_11"  # Spleeter 11kHz model
    SPLEETER_16 = "spleeter_16"  # Spleeter 16kHz model


class Segment(BaseModel):
    """
    A segment of transcribed text with timing information
    """

    text: str
    start: float  # Start time in seconds
    end: float  # End time in seconds


class TranscriptionResult(BaseModel):
    """
    Complete transcription result including metadata and segmented text
    """

    full_text: str
    segments: List[Segment]
    model_name: str
    whisper_implementation: str
    audio_type: AudioProcessingType
    alignment_model_load_time: Optional[float] = (
        None  # Time spent loading alignment models in seconds
    )
    transcription_time: float  # in seconds
    detected_language: str
