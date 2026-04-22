import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MODEL_LABELS = {
    "openai/whisper-large-v3": "Whisper",
    "nvidia/canary-1b-v2": "Canary",
    "nvidia/parakeet-tdt-0.6b-v3": "Parakeet TDT",
}

INPUT_FROM_FILENAME = {
    "vocals.wav": "stems",
    "htdemucs_ft_vocals.wav": "sep",
    "mixture.wav": "mix",
    "audio.mp3": "mix",
    "audio.wav": "mix",
}

DATASET_ORDER = ["musdb_alt", "jam-alt", "final_test"]
DATASET_DISPLAY = {
    "musdb_alt": "MUSDB-ALT",
    "jam-alt": "JAM-ALT",
    "final_test": "Private",
}


def _dataset_tex(ds: str) -> str:
    return DATASET_DISPLAY[ds].replace("_", "\\_")


def _derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["input"] = df["filename"].map(INPUT_FROM_FILENAME)
    is_cross = df["job_dir"].str.contains("cross_vad")
    df["vad_kind"] = "none"
    df.loc[df["vad"] & is_cross, "vad_kind"] = "cross"
    df.loc[df["vad"] & ~is_cross, "vad_kind"] = "same"
    df["chunk"] = df["chunked"].map({True: "Y", False: "N"})
    df["ref_words"] = df["hits"] + df["substitutions"] + df["deletions"]
    df["total_err"] = df["insertions"] + df["deletions"] + df["substitutions"]
    df["i_ref"] = df["insertions"] / df["ref_words"]
    df["d_ref"] = df["deletions"] / df["ref_words"]
    df["s_ref"] = df["substitutions"] / df["ref_words"]
    df["ins_pct"] = df["insertions"] / df["total_err"] * 100
    df["del_pct"] = df["deletions"] / df["total_err"] * 100
    df["sub_pct"] = df["substitutions"] / df["total_err"] * 100
    return df


def _row_full(r) -> str:
    return (
        f"{r.model_label} & {r.input} & {r.vad_kind} & {r.chunk} & "
        f"{int(r.n_songs)} & {r.wer * 100:.1f} & "
        f"{r.i_ref:.3f} & {r.d_ref:.3f} & {r.s_ref:.3f} & "
        f"{r.ins_pct:.1f} & {r.del_pct:.1f} & {r.sub_pct:.1f} \\\\"
    )


def _row_headline(r) -> str:
    return (
        f"{r.model_label} & {r.input} & {r.vad_kind} & {r.chunk} & "
        f"{r.wer * 100:.1f} & "
        f"{r.i_ref:.3f} & {r.d_ref:.3f} & {r.s_ref:.3f} & "
        f"{r.ins_pct:.1f} & {r.del_pct:.1f} & {r.sub_pct:.1f} \\\\"
    )


def write_full_results_tex(df: pd.DataFrame, out_path: Path) -> None:
    df = _derive_columns(df)
    lines = [
        "% Auto-generated — do not edit by hand.",
        "\\begin{tabular}{llllrrrrrrrr}",
        "\\hline",
        "Model & Input & VAD & Chunk & n & WER & I/ref & D/ref & S/ref & Ins\\% & Del\\% & Sub\\% \\\\",
        "\\hline",
    ]
    for ds in DATASET_ORDER:
        sub = df[df["dataset"] == ds].sort_values("wer")
        if sub.empty:
            continue
        n = int(sub["n_songs"].max())
        lines.append(
            f"\\multicolumn{{12}}{{l}}{{\\textit{{{_dataset_tex(ds)} ({n} songs)}}}} \\\\"
        )
        for r in sub.itertuples():
            lines.append(_row_full(r))
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {out_path}")


def write_headline_results_tex(df: pd.DataFrame, out_path: Path) -> None:
    df = _derive_columns(df)
    lines = [
        "% Auto-generated — do not edit by hand.",
        "\\begin{tabular}{llllrrrrrrr}",
        "\\hline",
        "Model & Input & VAD & Chunk & WER & I/ref & D/ref & S/ref & Ins\\% & Del\\% & Sub\\% \\\\",
        "\\hline",
    ]
    for ds in DATASET_ORDER:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        best = sub.sort_values("wer").groupby("model_label", as_index=False).first()
        best = best.sort_values("wer")
        lines.append(
            f"\\multicolumn{{11}}{{l}}{{\\textit{{{_dataset_tex(ds)}}}}} \\\\"
        )
        for r in best.itertuples():
            lines.append(_row_headline(r))
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {out_path}")
