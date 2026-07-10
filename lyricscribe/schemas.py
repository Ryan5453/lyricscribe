from datetime import datetime

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


class AlignedWord(BaseModel):
    """
    A single word with MFA-derived time alignment.

    Uses the same ``start`` / ``duration`` ms convention as :class:`SyncedLine`
    for consistency — callers that want word ``end`` can compute
    ``start + duration``.

    :param word: Word text in its original cased/punctuated form,
        recovered from the synced line at alignment write-back. Consumers
        that need a normalized form should lowercase/clean at the use site.
    :param start: Start time in milliseconds.
    :param duration: Duration in milliseconds.
    """

    word: str
    start: int
    duration: int


class Alignment(BaseModel):
    """
    Word-level forced alignment for a song, with metadata about how it was
    produced. Lives alongside :class:`SyncedLyrics` on a :class:`Lyrics` so
    downstream tools only need to load one file.

    :param words: Word-level alignments, sorted by ``start``.
    :param source_audio: Filename of the audio used for alignment (e.g.
        ``"vocals.wav"``, ``"htdemucs_ft_vocals.wav"``) — different vocal
        sources can give meaningfully different alignments.
    :param mfa_model: Name of the MFA acoustic model + dictionary used
        (e.g. ``"english_mfa"``).
    :param generated_at: When the alignment was produced, UTC ISO-8601.
    """

    words: list[AlignedWord]
    source_audio: str
    mfa_model: str
    generated_at: datetime


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
    :param alignment: Optional word-level MFA alignment. ``None`` until
        ``lyricscribe dataset align`` has been run on this song. Re-running
        the align command overwrites any existing alignment.
    """

    unsynced: UnsyncedLyrics
    synced: SyncedLyrics
    detected_language: str
    language_confidence: float
    alignment: Alignment | None = None
