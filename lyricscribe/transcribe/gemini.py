"""
Gemini (Google Generative Language API) as a transcriber, under the same
RMS-VAD chunked protocol as the open audio-LLMs.

The API key is read from the ``GEMINI_API_KEY`` environment variable only and
is never written anywhere. Each chunk is one request; 429/500/503/504
responses and network errors are retried with backoff.
"""

import base64
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request

import numpy as np
from torch.jit import ScriptModule

from lyricscribe.transcribe.base import Transcriber
from lyricscribe.transcribe.chunking import SAMPLE_RATE, language_name, song_chunks

logger = logging.getLogger(__name__)

ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
PROMPT = (
    "Transcribe the sung lyrics in this audio. Output only the lyrics text, "
    "with no commentary, no headings, and no section labels. "
    "The language is {language}."
)


class GeminiTranscriber(Transcriber):
    """
    Chunk-by-chunk transcription through the Gemini API (temperature 0).
    """

    def __init__(self, model_name: str, batch_size: int = 1, max_retries: int = 12) -> None:
        """
        :param model_name: Gemini model id, e.g. ``gemini-3.5-flash-lite``.
        :param batch_size: Unused.
        :param max_retries: Attempts per chunk on 429/5xx/network errors.
        """
        super().__init__(model_name, batch_size)
        self.max_retries = max_retries
        self.key = None

    def load(self) -> None:
        self.key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not self.key:
            raise RuntimeError("GEMINI_API_KEY is not set")

    def _call(self, chunk: np.ndarray, language: str) -> str:
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, chunk, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        body = json.dumps({
            "contents": [{"parts": [
                {"text": PROMPT.format(language=language)},
                {"inline_data": {
                    "mime_type": "audio/wav",
                    "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                }},
            ]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048},
        }).encode()
        url = ENDPOINT.format(model=self.model_name.removeprefix("gemini/"))
        delay = 4.0
        for attempt in range(self.max_retries):
            request = urllib.request.Request(
                url, data=body,
                headers={"Content-Type": "application/json", "X-goog-api-key": self.key},
            )
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    payload = json.loads(response.read())
                candidates = payload.get("candidates") or []
                if not candidates:
                    return ""
                parts = candidates[0].get("content", {}).get("parts") or []
                texts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
                return " ".join(t.strip() for t in texts if t.strip())
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 503, 504) or attempt == self.max_retries - 1:
                    detail = exc.read()[:300].decode(errors="replace")
                    raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc
                time.sleep(delay)
                delay = min(delay * 2, 90)
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay = min(delay * 2, 120)
        return ""

    def transcribe(
        self,
        audio_path: str,
        use_vad: bool = False,
        vad_model: ScriptModule | None = None,
        use_chunked: bool = False,
        language: str | None = None,
        vad_source: str | None = None,
        vad_method: str = "silero",
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        num_beams: int = 5,
        condition_on_prev: bool = True,
        temperature_fallback: bool = False,
        compression_ratio_threshold: float | None = None,
        logprob_threshold: float | None = None,
        initial_prompt: str | None = None,
    ) -> str:
        """
        Transcribe one song. Decoding knobs are accepted and ignored.
        """
        del vad_model, use_chunked, repetition_penalty, no_repeat_ngram_size
        del num_beams, condition_on_prev, temperature_fallback
        del compression_ratio_threshold, logprob_threshold, initial_prompt
        name = language_name(language) or "English"
        chunks = song_chunks(audio_path, use_vad, vad_method, vad_source)
        texts = [self._call(c, name) for c in chunks]
        return " ".join(t for t in texts if t)
