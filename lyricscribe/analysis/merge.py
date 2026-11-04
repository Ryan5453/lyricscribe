import json
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_SKIP_FILES = {
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "trainer_state.json",
    "training_args.bin",
    "model.safetensors.index.json",
}


def load_state(path: Path) -> dict:
    """
    Load a (possibly sharded) safetensors checkpoint dir.

    :param path: Checkpoint dir.
    :return: ``{name: tensor}``.
    """
    from safetensors.torch import load_file

    path = Path(path)
    index = path / "model.safetensors.index.json"
    state = {}
    if index.exists():
        for shard in sorted(set(json.loads(index.read_text())["weight_map"].values())):
            state.update(load_file(str(path / shard)))
    else:
        state.update(load_file(str(path / "model.safetensors")))
    return state


def merge_checkpoints(
    members: list[Path],
    output_dir: Path,
    weights: list[float] | None = None,
    dtype: str = "float16",
) -> dict:
    """
    Weighted average of member checkpoints, accumulated in float32.

    Config/tokenizer/processor files are copied from the FIRST member, so put
    the checkpoint whose config should load the soup first.

    :param members: Checkpoint dirs.
    :param output_dir: Destination dir (created).
    :param weights: Per-member weights (normalised to sum 1); default uniform.
    :param dtype: Saved dtype, as a ``torch`` dtype name.
    :return: ``{"weights", "tensors", "dropped_keys"}``.
    """
    import torch
    from safetensors.torch import save_file

    if len(members) < 2:
        raise ValueError("need at least two members")
    if weights is None:
        weights = [1.0] * len(members)
    if len(weights) != len(members):
        raise ValueError("weights/members length mismatch")
    total = sum(weights)
    weights = [w / total for w in weights]

    merged: dict = {}
    keys = None
    dropped: set[str] = set()
    for path, weight in zip(members, weights):
        state = load_state(path)
        if keys is None:
            keys = set(state)
        elif set(state) != keys:
            gone = keys ^ set(state)
            logger.warning(f"{path}: key mismatch, merging intersection; dropped {sorted(gone)}")
            dropped |= gone
            keys &= set(state)
            for k in list(merged):
                if k not in keys:
                    del merged[k]
        for k, t in state.items():
            if k in keys:
                c = t.to(torch.float32) * weight
                merged[k] = c if k not in merged else merged[k] + c
        del state
        logger.info(f"merged {path} (w={weight:.4f})")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in Path(members[0]).iterdir():
        if item.is_file() and item.suffix != ".safetensors" and item.name not in _SKIP_FILES \
                and not item.name.startswith("rng_state"):
            shutil.copy2(item, output_dir / item.name)
    out_dtype = getattr(torch, dtype)
    save_file(
        {k: v.to(out_dtype).contiguous() for k, v in merged.items()},
        str(output_dir / "model.safetensors"),
        metadata={"format": "pt"},
    )
    (output_dir / "merge_info.json").write_text(json.dumps({
        "members": [str(m) for m in members],
        "weights": weights,
        "dtype": dtype,
        "dropped_keys": sorted(dropped),
    }, indent=2) + "\n")
    return {"weights": weights, "tensors": len(merged), "dropped_keys": sorted(dropped)}
