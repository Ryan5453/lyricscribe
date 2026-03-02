from pydantic import BaseModel


class SyncedLine(BaseModel):
    """
    A single line of time-synced lyrics.

    :param text: Lyrics text for this line.
    :param start: Start time in milliseconds.
    :param duration: Duration in milliseconds.
    """

    text: str
    start: int
    duration: int


class UnsyncedLyrics(BaseModel):
    """
    Full unsynced lyrics text with provider attribution.

    :param data: Complete lyrics as a single string.
    :param provider: Name of the provider that supplied the lyrics.
    """

    data: str
    provider: str


class SyncedLyrics(BaseModel):
    """
    Line-level synced lyrics with provider attribution.

    :param data: List of time-synced lyric lines.
    :param provider: Name of the provider that supplied the lyrics.
    """

    data: list[SyncedLine]
    provider: str


class Lyrics(BaseModel):
    """
    Top-level lyrics schema matching the private dataset format.

    :param unsynced: Unsynced lyrics with provider information.
    :param synced: Line-level synced lyrics with provider information.
    :param detected_language: Language code for lyrics.
    :param language_confidence: Confidence score for the detected language (0.0-1.0).
        Always 1.0 for curated/ground-truth datasets.
    """

    unsynced: UnsyncedLyrics
    synced: SyncedLyrics
    detected_language: str
    language_confidence: float
