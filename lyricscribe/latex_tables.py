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


def _config_label(job_dir: str) -> str:
    """Strip the dataset prefix from the job-dir basename to get a compact
    pipeline config label (e.g. 'stems_vad_chunked', 'mix_cross_vad')."""
    basename = job_dir.split("/")[-1]
    for prefix in ("musdb_alt_", "jam_alt_", "final_test_"):
        if basename.startswith(prefix):
            basename = basename[len(prefix):]
            break
    return basename.replace("_", "\\_")


def _derive_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["input"] = df["filename"].map(INPUT_FROM_FILENAME)
    is_cross = df["job_dir"].str.contains("cross_vad")
    df["vad_kind"] = "none"
    df.loc[df["vad"] & is_cross, "vad_kind"] = "cross"
    df.loc[df["vad"] & ~is_cross, "vad_kind"] = "same"
    df["chunk"] = df["chunked"].map({True: "Y", False: "N"})
    df["config_label"] = df["job_dir"].map(_config_label)
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
        f"{r.model_label} & {r.config_label} & {int(r.n_songs)} & "
        f"{r.wer * 100:.1f} & "
        f"{r.i_ref:.3f} & {r.d_ref:.3f} & {r.s_ref:.3f} \\\\"
    )


def _row_headline(r) -> str:
    return (
        f"{r.model_label} & {r.config_label} & {r.wer * 100:.1f} & "
        f"{int(r.insertions):,} & {int(r.deletions):,} & {int(r.substitutions):,} \\\\"
    )


def write_full_results_tex(df: pd.DataFrame, out_path: Path) -> None:
    """
    Write the full per-config results table as a standalone LaTeX document.
    This is intended to be compiled directly (``pdflatex full_results.tex``)
    as a reference appendix; it is NOT meant to be ``\\input{}``-ed into
    the main paper, since it can run hundreds of rows long.
    """
    df = _derive_columns(df)
    lines = [
        "% Auto-generated — do not edit by hand.",
        "\\documentclass[10pt]{article}",
        "\\usepackage[margin=1in]{geometry}",
        "\\usepackage{longtable}",
        "\\usepackage{booktabs}",
        "\\begin{document}",
        "\\begin{center}",
        "\\textbf{\\large Full ALT evaluation results}",
        "\\end{center}",
        "\\begin{longtable}{llrrrrr}",
        "\\toprule",
        "Model & Config & n & WER & I/ref & D/ref & S/ref \\\\",
        "\\midrule",
        "\\endhead",
    ]
    for ds in DATASET_ORDER:
        sub = df[df["dataset"] == ds].sort_values("wer")
        if sub.empty:
            continue
        n = int(sub["n_songs"].max())
        lines.append(
            f"\\multicolumn{{7}}{{l}}{{\\textit{{{_dataset_tex(ds)} ({n} songs)}}}} \\\\"
        )
        for r in sub.itertuples():
            lines.append(_row_full(r))
        lines.append("\\midrule")
    lines.append("\\bottomrule")
    lines.append("\\end{longtable}")
    lines.append("\\end{document}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {out_path}")


def write_headline_results_tex(df: pd.DataFrame, out_path: Path) -> None:
    df = _derive_columns(df)
    lines = [
        "% Auto-generated — do not edit by hand.",
        "\\begin{tabular}{llrrrr}",
        "\\hline",
        "Model & Config & WER & Ins & Del & Sub \\\\",
        "\\hline",
    ]
    for ds in DATASET_ORDER:
        sub = df[df["dataset"] == ds]
        if sub.empty:
            continue
        best = sub.sort_values("wer").groupby("model_label", as_index=False).first()
        best = best.sort_values("wer")
        lines.append(
            f"\\multicolumn{{6}}{{l}}{{\\textit{{{_dataset_tex(ds)}}}}} \\\\"
        )
        for r in best.itertuples():
            lines.append(_row_headline(r))
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Wrote {out_path}")
