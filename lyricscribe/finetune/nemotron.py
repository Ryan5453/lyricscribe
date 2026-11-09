import importlib.util
import logging

import torch
from transformers import AutoProcessor, Nemotron3_5TranscribeForConditionalGeneration

from lyricscribe.finetune import engine
from lyricscribe.finetune.engine import FeatureLayout, configured_prompt_mode, drop_bad, training_processor
from lyricscribe.finetune.qwen import _concat_prompt_target, _QwenTrainerBase, adapter_summary, nvidia_adapter
from lyricscribe.transcribe.base import shared_prompt
from lyricscribe.transcribe.chunking import language_name
from lyricscribe.transcribe.nemotron import PROMPT, NemotronTranscriber

logger = logging.getLogger(__name__)

# NVIDIA's make_ft_config targets (q,k,v,o,in_proj,out_proj,up_proj,down_proj) matched as modules and
# restricted to the LLM: they reach shared_experts.{up,down}_proj, never the 3-D routed-expert tensors.
NEMOTRON_LORA_MODULES = (
    r"model\.language_model\.layers\.\d+\.mixer\."
    r"(?:q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|shared_experts\.up_proj|shared_experts\.down_proj)"
)
# ``--train-encoder --train-proj``: the ASR branch of the parallel-expert encoder, its norm, and the projection.
NEMOTRON_FULL_RANK = ["asr_encoder", "asr_norm", "multi_modal_projector"]


def require_unfused_mamba() -> None:
    """
    Refuse to train with ``mamba_ssm`` importable: its fused kernel reads ``out_proj.weight``
    directly, silently bypassing the LoRA on ``out_proj``.

    :raises RuntimeError: If ``mamba_ssm`` is installed.
    """
    if importlib.util.find_spec("mamba_ssm") is not None:
        raise RuntimeError("mamba_ssm is installed: its fused Mamba path would bypass the out_proj LoRA")


class NemotronCollator:
    """
    The transcription request inference sends (``apply_transcription_request`` with
    :data:`~lyricscribe.transcribe.nemotron.PROMPT` or the shared prompt, thinking off)
    followed by the transcript and ``<|im_end|>``; the prompt is masked out of the loss.

    :param processor: ``Nemotron3_5TranscribeProcessor``.
    :param prompt_mode: ``asr`` or ``shared``.
    """

    def __init__(self, processor, prompt_mode: str = "asr"):
        """
        Keep the processor and resolve the pad id.

        :param processor: ``Nemotron3_5TranscribeProcessor``.
        :param prompt_mode: ``asr`` (the card's prompt) or ``shared``.
        """
        self.processor = processor
        self.prompt_mode = prompt_mode
        self.tokenizer = processor.tokenizer
        self.pad_id = self.tokenizer.pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        """
        Collate dataset items into a training batch.

        :param features: :class:`~lyricscribe.finetune.engine.ManifestDataset` items.
        :return: Token ids, attention mask, labels, the padded audio features and the
            trainer's ``_feature_lengths`` / ``_placeholder``.
        """
        features = drop_bad(features)
        prompts, targets, feats, masks = [], [], [], []
        for item in features:
            if self.prompt_mode == "shared":
                prompt = shared_prompt(item["language"])
            else:
                prompt = PROMPT.format(language=language_name(item["language"]) or "as spoken")
            enc = self.processor.apply_transcription_request([item["audio"]], prompt=prompt)
            prompts.append(enc["input_ids"][0])
            targets.append(self.tokenizer(
                item["text"] + self.tokenizer.eos_token, add_special_tokens=False, return_tensors="pt",
            )["input_ids"][0])
            feats.append(enc["input_features"][0])
            masks.append(enc["input_features_mask"][0])
        input_ids, attention, labels = _concat_prompt_target(prompts, targets, self.pad_id)
        # (frames, mel bins), each a multiple of the subsampling factor.
        frames = max(f.shape[0] for f in feats)
        mask = torch.stack([torch.nn.functional.pad(m, (0, frames - m.shape[0])) for m in masks])
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": labels,
            "input_features": torch.stack([
                torch.nn.functional.pad(f, (0, 0, 0, frames - f.shape[0])) for f in feats
            ]),
            "input_features_mask": mask,
            "_feature_lengths": mask.sum(-1),
            "_placeholder": any(it.get("placeholder") for it in features),
        }


class NemotronLoraTrainer(_QwenTrainerBase):
    """
    NVIDIA's speechlm2 fine-tuning recipe on Nemotron 3.5 Transcribe: the ASR encoder,
    its norm and the projection at full rank, LoRA r32/α64 on the hybrid Mamba-Transformer
    MoE LLM, routers frozen. Checkpoints are PEFT adapters over the base model.
    """

    adapter = True
    layout = FeatureLayout(key="input_features", time_dim=1, freq_dim=2)

    def _load(self, path: str) -> None:
        """
        Load the base model in bf16 on this rank's GPU and wrap it with the adapter.

        :param path: Hub id or path of the base checkpoint.
        """
        require_unfused_mamba()
        logger.info(f"Loading Nemotron 3.5 Transcribe: {path}")
        self.processor = AutoProcessor.from_pretrained(path)
        model = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, device_map=engine.device_map()
        )
        model.config.use_cache = False
        model.config.text_config.use_cache = False
        self.model = nvidia_adapter(model, NEMOTRON_LORA_MODULES, NEMOTRON_FULL_RANK)

    def collator(self):
        """
        The training collator.

        :return: :class:`NemotronCollator`.
        """
        return NemotronCollator(training_processor(self.processor), self.prompt_mode())

    def prompt_mode(self) -> str:
        """
        The training prompt, as :class:`~lyricscribe.transcribe.nemotron.NemotronTranscriber` names it.

        :return: The config's ``prompt_mode`` (default ``asr``, the card's prompt).
        """
        return configured_prompt_mode(self.config, NemotronTranscriber.prompt_modes)

    def notes(self) -> dict:
        """
        Recipe facts recorded in the training state.

        :return: Notes.
        """
        return {
            "adaptation": "NVIDIA recipe (make_ft_config --train-encoder --train-proj); routers and routed experts frozen",
            **adapter_summary(self.model),
        }
