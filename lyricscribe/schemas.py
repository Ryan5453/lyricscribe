from pydantic import BaseModel


class SyncedLine(BaseModel):
    """
    A single line of time-synced lyrics.
    """

    text: str
    start: int
    duration: int


class UnsyncedLyrics(BaseModel):
    """
    Full unsynced lyrics text with provider attribution.
    """

    data: str
    provider: str


class SyncedLyrics(BaseModel):
    """
    Line-level synced lyrics with provider attribution.
    """

    data: list[SyncedLine]
    provider: str


class Lyrics(BaseModel):
    """
    Top-level lyrics schema matching the private dataset format.
    """

    unsynced: UnsyncedLyrics
    synced: SyncedLyrics
    detected_language: str
    language_confidence: float
