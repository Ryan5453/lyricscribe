import logging

import numpy as np
import torch
from peft import PeftModel
from transformers import (
    AutoProcessor,
    MusicFlamingoForConditionalGeneration,
    Qwen3OmniMoeForConditionalGeneration,
    Qwen3OmniMoeProcessor,
)

from lyricscribe.transcribe.adapter import check_train_record, load_adapter
from lyricscribe.transcribe.base import (
    DECODE_KNOBS,
    MAX_NEW_TOKENS,
    ChunkTranscriber,
    Decode,
    describe_kwargs,
    generate_kwargs,
    shared_prompt,
)
from lyricscribe.transcribe.chunking import language_name

logger = logging.getLogger(__name__)

# The instruction every audio-LLM got before the vendor prompts; kept to reproduce earlier runs.
LEGACY_PROMPT = (
    "Transcribe the sung lyrics. Output only the lyrics, no commentary. "
    "The language is {language}."
)
PROMPT = LEGACY_PROMPT

# Prompts documented by each vendor, by task: "asr" (speech recognition) or "lyrics".
VENDOR_PROMPTS = {
    "qwen3_omni": {
        "asr": "Transcribe the {language} audio into text.",
        "lyrics": (
            "Transcribe the song lyrics into text without any punctuation, separate lines with "
            "line breaks, and output only the lyrics without additional explanations."
        ),
    },
    "musicflamingo": {
        # MF documents no ASR prompt; this is its base model Audio Flamingo 3's default_transcription_prompt.
        "asr": "Transcribe the input speech.",
        "lyrics": "Generate a structured lyric sheet from the input music.",
    },
}

PROMPT_MODES = ("legacy", "asr", "lyrics", "shared")

FAMILIES = ("qwen3_omni", "musicflamingo")

# Adapters saved before transformers 5.22 put the thinker's towers beside ``model``, not inside it.
OMNI_ADAPTER_KEY_MAPPING = {
    r"^audio_tower\.": "model.audio_tower.",
    r"^visual\.": "model.visual.",
    r"^model\.(?!language_model\.|audio_tower\.|visual\.)": "model.language_model.",
}


def load_omni_adapter(thinker, adapter: str, is_trainable: bool = False) -> PeftModel:
    """
    Wrap the Qwen3-Omni thinker with a saved LoRA adapter, mapping adapters
    saved under the pre-5.22 module layout onto the current one.

    PEFT's own remapping covers the text layers but skips the audio tower,
    silently leaving those LoRA modules at their zero init, hence the checks in
    :func:`~lyricscribe.transcribe.adapter.load_adapter`.

    :param thinker: ``Qwen3OmniMoeThinkerForConditionalGeneration``.
    :param adapter: Adapter dir.
    :param is_trainable: Keep the adapter trainable.
    :return: The wrapped thinker.
    :raises RuntimeError: If any LoRA module or ``modules_to_save`` copy did not load.
    """
    return load_adapter(thinker, adapter, is_trainable=is_trainable, key_mapping=OMNI_ADAPTER_KEY_MAPPING)


def audio_llm_prompt(family: str, mode: str, language: str | None) -> str:
    """
    The instruction sent with each chunk.

    :param family: One of :data:`FAMILIES`.
    :param mode: One of :data:`PROMPT_MODES`.
    :param language: ``detected_language`` from ``lyrics.json``, or ``None``.
    :return: Prompt text.
    :raises ValueError: If the vendor documents no prompt for that task.
    """
    name = language_name(language) or "English"
    if mode == "legacy":
        return LEGACY_PROMPT.format(language=name)
    if mode == "shared":
        return shared_prompt(language)
    template = VENDOR_PROMPTS.get(family, {}).get(mode)
    if template is None:
        raise ValueError(f"{family} documents no {mode!r} prompt")
    return template.format(language=name)


def audio_llm_inputs(processor, family: str, prompt: str, wav: np.ndarray):
    """
    Processor inputs for one chunk in the family's chat format; training
    renders its prompts through this too, so the two can't drift apart.

    :param processor: The family's processor.
    :param family: ``"qwen3_omni"`` or ``"musicflamingo"``.
    :param prompt: Instruction text.
    :param wav: 16 kHz mono audio.
    :return: The processor's ``BatchFeature`` for a batch of one.
    """
    audio = {"type": "audio"}
    text = {"type": "text", "text": prompt}
    content = [text, audio] if family == "musicflamingo" else [audio, text]
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
    # No padding override: Music Flamingo's encoder needs the processor's default 30 s windows.
    return processor(text=[rendered], audio=[np.asarray(wav, dtype=np.float32)], return_tensors="pt")


class AudioLLMTranscriber(ChunkTranscriber):
    """
    Chunk-by-chunk transcription with an instruction-following audio-LLM.

    Chunks are passed to the processor as float arrays and decoded one at a
    time; the instruction is chosen by ``prompt_mode`` (see :func:`audio_llm_prompt`).
    """

    supports = frozenset(DECODE_KNOBS)
    prompt_modes = PROMPT_MODES

    def __init__(
        self,
        model_name: str,
        family: str,
        batch_size: int = 1,
        max_new_tokens: int = MAX_NEW_TOKENS,
        adapter: str | None = None,
        prompt_mode: str = "legacy",
        dtype: str | None = None,
    ) -> None:
        """
        :param model_name: Hub id or local dir.
        :param family: One of :data:`FAMILIES`.
        :param batch_size: Ignored; chunks are decoded one at a time.
        :param max_new_tokens: Generation cap per chunk.
        :param adapter: PEFT adapter dir merged into the model (the Qwen3-Omni
            thinker, or the whole Music Flamingo model).
        :param prompt_mode: One of :data:`PROMPT_MODES`.
        :param dtype: Music Flamingo only: ``bfloat16`` (default, its checkpoint dtype) or
            ``float32``; the other families run bf16.
        """
        super().__init__(model_name, 1, prompt_mode)
        if family not in FAMILIES:
            raise ValueError(f"Unknown audio-LLM family {family!r}")
        if dtype and family != "musicflamingo" and dtype != "bfloat16":
            raise ValueError(f"{family} runs bfloat16 only, got dtype {dtype!r}")
        if batch_size != 1:
            logger.info(f"{family}: batch_size {batch_size} -> 1 (chunks decode one at a time)")
        self.family = family
        self.max_new_tokens = max_new_tokens
        self.adapter = adapter
        self.dtype = dtype
        self.model = None
        self.processor = None

    def load(self) -> None:
        """
        Load the model and processor for this family onto the GPU (CPU without CUDA).

        :raises RuntimeError: If the adapter's training record has another ``prompt_mode``.
        """
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading {self.family}: {self.model_name} on {device}")
        self.train_record = check_train_record(
            self.adapter or self.model_name, self.prompt_mode, self.model_name if self.adapter else None
        )
        # cuDNN SDPA rebuilds its graph per sequence shape, which stalls decoding.
        torch.backends.cuda.enable_cudnn_sdp(False)
        if self.family == "qwen3_omni":
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
                # Merged, not wrapped, so the model's generate() path is untouched.
                wrapped = load_omni_adapter(self.model.thinker, self.adapter)
                self.model.thinker = wrapped.merge_and_unload()
                logger.info(f"Merged LoRA adapter {self.adapter}")
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = MusicFlamingoForConditionalGeneration.from_pretrained(
                self.model_name, dtype=getattr(torch, self.dtype or "bfloat16"), device_map=device
            )
            if self.adapter:
                self.model = load_adapter(self.model, self.adapter).merge_and_unload()
                logger.info(f"Merged adapter {self.adapter}")
        self.model.eval()

    @torch.inference_mode()
    def _generate(self, wav: np.ndarray, prompt: str, decode: Decode) -> str:
        """
        Transcribe one chunk with the family's chat format.

        :param wav: 16 kHz mono chunk.
        :param prompt: Instruction text sent with the audio.
        :param decode: Decoding settings.
        :return: The generated text.
        """
        kwargs = generate_kwargs(self.max_new_tokens, decode)
        self.last_generate_kwargs = {**describe_kwargs(kwargs), "prompt": prompt}
        inputs = audio_llm_inputs(self.processor, self.family, prompt, wav).to(self.model.device)
        if self.family == "qwen3_omni":
            inputs = inputs.to(self.model.device, dtype=torch.bfloat16)
            # Omni's generate() routes thinker_* kwargs to the thinker; the talker is disabled.
            out = self.model.generate(
                **inputs,
                **{f"thinker_{k}": v for k, v in kwargs.items()},
                return_audio=False,
                use_audio_in_video=False,
            )
            if isinstance(out, (tuple, list)):
                out = out[0]
        else:
            inputs["input_features"] = inputs["input_features"].to(self.model.dtype)
            out = self.model.generate(**inputs, **kwargs)
        decoded = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        return decoded[0].strip()

    def _decode(self, wavs: list[np.ndarray], language: str | None, decode: Decode) -> list[str]:
        """
        Decode each chunk on its own.

        :param wavs: 16 kHz mono chunks.
        :param language: ``detected_language`` from ``lyrics.json``, or ``None``.
        :param decode: Decoding settings.
        :return: One transcription per chunk.
        """
        prompt = audio_llm_prompt(self.family, self.prompt_mode, language)
        return [self._generate(w, prompt, decode) for w in wavs]
