import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.jit import ScriptModule

from lyricscribe.transcribe.base import Transcriber
from lyricscribe.transcribe.chunking import SAMPLE_RATE, language_name, song_chunks

logger = logging.getLogger(__name__)

PROMPT = (
    "Transcribe the sung lyrics. Output only the lyrics, no commentary. "
    "The language is {language}."
)

FAMILIES = ("voxtral", "qwen3_omni", "musicflamingo")


class AudioLLMTranscriber(Transcriber):
    """
    Chunk-by-chunk transcription with an instruction-following audio-LLM.

    ``no_repeat_ngram_size`` is applied at decode time; ``0`` disables it.
    """

    def __init__(
        self,
        model_name: str,
        family: str,
        batch_size: int = 1,
        max_new_tokens: int = 1024,
        adapter: str | None = None,
    ) -> None:
        """
        :param model_name: Hub id or local dir.
        :param family: One of :data:`FAMILIES`.
        :param batch_size: Unused (chunks are decoded one at a time).
        :param max_new_tokens: Generation cap per chunk.
        :param adapter: PEFT LoRA adapter dir merged into the Qwen3-Omni
            thinker (Qwen3-Omni only).
        """
        super().__init__(model_name, batch_size)
        if family not in FAMILIES:
            raise ValueError(f"Unknown audio-LLM family {family!r}")
        if adapter and family != "qwen3_omni":
            raise ValueError("LoRA adapters are only wired for Qwen3-Omni")
        self.family = family
        self.max_new_tokens = max_new_tokens
        self.adapter = adapter
        self.model = None
        self.processor = None

    def load(self) -> None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {self.family}: {self.model_name} on {device}")
        # cuDNN SDPA rebuilds its graph per sequence shape, which stalls decoding.
        torch.backends.cuda.enable_cudnn_sdp(False)
        if self.family == "voxtral":
            from transformers import AutoProcessor, VoxtralForConditionalGeneration

            try:
                import mistral_common  # noqa: F401
            except ImportError as e:
                # Not in pyproject: its numpy <2.4 cap would downgrade the whole env.
                raise ImportError(
                    "Voxtral needs mistral-common, installed outside the lock: "
                    "uv pip install --no-deps mistral-common jsonschema "
                    "jsonschema-specifications referencing rpds-py tiktoken "
                    "pycountry pydantic-extra-types"
                ) from e

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_name, dtype=torch.bfloat16, device_map=device
            )
        elif self.family == "qwen3_omni":
            from transformers import (
                Qwen3OmniMoeForConditionalGeneration,
                Qwen3OmniMoeProcessor,
            )

            self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_name)
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                self.model_name,
                dtype=torch.bfloat16,
                device_map=device,
                attn_implementation="sdpa",
            )
            # Without the talker, generate() returns the thinker's ids directly.
            self.model.disable_talker()
            if self.adapter:
                from peft import PeftModel

                # Merged, not wrapped, so the model's generate() path is untouched.
                wrapped = PeftModel.from_pretrained(self.model.thinker, self.adapter)
                self.model.thinker = wrapped.merge_and_unload()
                logger.info(f"Merged LoRA adapter {self.adapter}")
        else:
            from transformers import (
                AutoProcessor,
                MusicFlamingoForConditionalGeneration,
            )

            self.processor = AutoProcessor.from_pretrained(self.model_name)
            # bf16 hits a dtype mismatch in a layer_norm on the fp32 audio path.
            self.model = MusicFlamingoForConditionalGeneration.from_pretrained(
                self.model_name, dtype=torch.float32, device_map=device
            )
        self.model.eval()

    @torch.inference_mode()
    def _generate(self, wav_path: Path, prompt: str, no_repeat_ngram_size: int) -> str:
        audio = {"type": "audio", "path": str(wav_path)}
        text = {"type": "text", "text": prompt}
        content = [text, audio] if self.family == "musicflamingo" else [audio, text]
        conversation = [{"role": "user", "content": content}]
        ngram = {"no_repeat_ngram_size": no_repeat_ngram_size} if no_repeat_ngram_size else {}

        if self.family == "voxtral":
            inputs = self.processor.apply_chat_template(conversation)
            inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
            out = self.model.generate(
                **ngram, **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        elif self.family == "qwen3_omni":
            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
            out = self.model.generate(
                **ngram,
                **inputs,
                thinker_max_new_tokens=self.max_new_tokens,
                thinker_do_sample=False,
                return_audio=False,
                use_audio_in_video=False,
            )
            if isinstance(out, (tuple, list)):
                out = out[0]
        else:
            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            ).to(self.model.device)
            inputs["input_features"] = inputs["input_features"].to(self.model.dtype)
            out = self.model.generate(
                **ngram, **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        decoded = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return decoded[0].strip()

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
        Transcribe one song with the fixed :data:`PROMPT`. Only
        ``no_repeat_ngram_size`` is honoured; the other decoding knobs and
        ``initial_prompt`` are accepted and ignored.
        """
        import soundfile as sf

        del vad_model, use_chunked, repetition_penalty, num_beams
        del condition_on_prev, temperature_fallback
        del compression_ratio_threshold, logprob_threshold, initial_prompt
        prompt = PROMPT.format(language=language_name(language) or "English")
        chunks = song_chunks(audio_path, use_vad, vad_method, vad_source)
        scratch_root = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR")
        pieces = []
        with tempfile.TemporaryDirectory(dir=scratch_root) as scratch:
            wav_path = Path(scratch) / "chunk.wav"
            for chunk in chunks:
                sf.write(str(wav_path), np.asarray(chunk), SAMPLE_RATE, subtype="PCM_16")
                pieces.append(self._generate(wav_path, prompt, no_repeat_ngram_size))
        return " ".join(p.strip() for p in pieces if p.strip())
