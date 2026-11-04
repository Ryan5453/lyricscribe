import json
import logging
from pathlib import Path

import jiwer

from lyricscribe.evaluate import _normalized

logger = logging.getLogger(__name__)


def build_selfwer_artifact(dataset_dir: Path, output_path: Path) -> int:
    """
    Write the per-song unsynced-vs-synced self-WER artifact (JSONL).

    Rows: ``song_id``, ``self_wer`` (``None`` when either text is empty),
    ``lang``, ``unsynced_provider``, ``synced_provider``, ``ref_words``. A
    song whose lyrics.json can't be read gets an ``error`` row instead.

    :param dataset_dir: Dataset directory.
    :param output_path: Destination JSONL.
    :return: Number of songs written.
    """
    n = 0
    with open(output_path, "w") as f:
        for song_dir in Path(dataset_dir).iterdir():
            path = song_dir / "lyrics.json"
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text())
                # Not evaluate._language_for: changing the parse changes existing artifacts.
                lang = (data.get("detected_language") or "en").split("-")[0]
                unsynced = data.get("unsynced") or {}
                synced = data.get("synced") or {}
                ref_u = _normalized_or_empty(unsynced.get("data"), lang)
                synced_data = synced.get("data") or []
                synced_text = (
                    " ".join(x.get("text", "") for x in synced_data)
                    if isinstance(synced_data, list)
                    else str(synced_data or "")
                )
                ref_s = _normalized_or_empty(synced_text, lang)
                self_wer = None
                if ref_u and ref_s:
                    m = jiwer.process_words([ref_u], [ref_s])
                    ref_len = m.substitutions + m.deletions + m.hits
                    if ref_len:
                        self_wer = round(
                            (m.insertions + m.deletions + m.substitutions) / ref_len, 4
                        )
                row = {
                    "song_id": song_dir.name,
                    "self_wer": self_wer,
                    "lang": lang,
                    "unsynced_provider": unsynced.get("provider"),
                    "synced_provider": synced.get("provider"),
                    "ref_words": len(ref_u.split()),
                }
            except Exception as e:  # noqa: BLE001
                row = {"song_id": song_dir.name, "error": str(e)[:80]}
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def _normalized_or_empty(text, language: str) -> str:
    """
    Normalise *text*, treating missing or non-string text as empty.

    :param text: Raw transcript, possibly ``None``.
    :param language: Language code.
    :return: Normalised text.
    """
    if not isinstance(text, str) or not text:
        return ""
    return _normalized(text, language)
