import json
from pathlib import Path

import torch
from peft import PeftModel
from peft.utils import ModulesToSaveWrapper

# Written by training beside every exported model or adapter.
TRAIN_RECORD = "lyricscribe_train.json"


class TrainRecordMismatch(RuntimeError):
    """
    A decode that contradicts its checkpoint's training record; fatal for the job, not one song.
    """


def _same_model(a: str, b: str) -> bool:
    """
    Whether two model references name the same model (equal strings or the same local dir).

    :param a: Hub id or path.
    :param b: Hub id or path.
    :return: ``True`` if they match.
    """
    if a == b:
        return True
    pa, pb = Path(a), Path(b)
    return pa.exists() and pb.exists() and pa.resolve() == pb.resolve()


def check_train_record(path: str | None, prompt_mode: str, base_model: str | None = None) -> dict | None:
    """
    Read the training record of a fine-tuned checkpoint or adapter and refuse to decode it
    with a different prompt, or an adapter on a different base model, than it was trained with.

    :param path: Adapter or checkpoint dir (a hub id has no record).
    :param prompt_mode: The prompt this job decodes with, named as training records it.
    :param base_model: The job's base model when *path* is an adapter, else ``None``.
    :return: The record, or ``None`` when *path* has none.
    :raises TrainRecordMismatch: If the record's ``prompt_mode`` or ``base_model`` differs.
    """
    record_path = Path(path) / TRAIN_RECORD if path else None
    if record_path is None or not record_path.is_file():
        return None
    record = json.loads(record_path.read_text())
    if record.get("prompt_mode") != prompt_mode:
        raise TrainRecordMismatch(
            f"{path} was trained with prompt_mode {record.get('prompt_mode')!r}; this job decodes with {prompt_mode!r}"
        )
    trained_on = record.get("base_model")
    if base_model is not None and trained_on and not _same_model(trained_on, base_model):
        raise TrainRecordMismatch(f"{path} was trained on {trained_on!r}; this job loads it onto {base_model!r}")
    return record


def load_adapter(
    model, adapter: str, is_trainable: bool = False, key_mapping: dict[str, str] | None = None
) -> PeftModel:
    """
    Wrap a model with a saved PEFT adapter and refuse one whose weights did
    not all land.

    PEFT loads silently when saved keys don't match the module layout, leaving
    LoRA ``B`` matrices at their zero init and ``modules_to_save`` copies (the
    full-rank encoder and projector of the NVIDIA recipe) equal to the base
    modules they replace; both are checked.

    :param model: Base model.
    :param adapter: Adapter dir.
    :param is_trainable: Keep the adapter trainable.
    :param key_mapping: Regex remapping of saved keys, for adapters saved
        under an older module layout.
    :return: The wrapped model.
    :raises RuntimeError: If any LoRA module or ``modules_to_save`` copy did not load.
    """
    kwargs = {"key_mapping": key_mapping} if key_mapping else {}
    wrapped = PeftModel.from_pretrained(model, adapter, is_trainable=is_trainable, **kwargs)
    unloaded = [
        name for name, module in wrapped.named_modules()
        if hasattr(module, "lora_B") and any(not b.weight.any() for b in module.lora_B.values())
    ]
    if unloaded:
        raise RuntimeError(f"{adapter}: {len(unloaded)} LoRA modules did not load, e.g. {unloaded[0]}")
    unchanged = [
        name for name, module in wrapped.named_modules()
        if isinstance(module, ModulesToSaveWrapper)
        and any(
            all(torch.equal(a, b) for a, b in zip(module.original_module.parameters(), copy.parameters()))
            for copy in module.modules_to_save.values()
        )
    ]
    if unchanged:
        raise RuntimeError(
            f"{adapter}: {len(unchanged)} modules_to_save copies equal the base, e.g. {unchanged[0]}"
        )
    return wrapped
