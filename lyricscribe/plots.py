import logging
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lyricscribe.evaluate import collect_evaluation_data
from lyricscribe.latex_tables import DATASET_DISPLAY

logger = logging.getLogger(__name__)

_MODEL_PALETTE = [
    "#006BA4",  # blue
    "#FF800E",  # orange
    "#595959",  # dark gray
    "#5F9ED1",  # light blue
    "#C85200",  # dark orange
    "#ABABAB",  # gray
]
_ERROR_COLORS = {
    "ins": "#FF800E",  # orange
    "del": "#006BA4",  # blue
    "sub": "#595959",  # dark gray
}


def _apply_style() -> None:
    """
    Apply a consistent matplotlib style to all plots.

    Uses the ``tableau-colorblind10`` style recommended by ISMIR for
    accessibility (distinguishable both to colorblind readers and when
    printed in grayscale).
    """
    plt.style.use("tableau-colorblind10")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
        }
    )


_MODEL_LABELS = {
    "openai/whisper-large-v3": "Whisper large-v3",
    "nvidia/canary-1b-v2": "Canary 1B v2",
    "nvidia/parakeet-tdt-0.6b-v3": "Parakeet TDT 0.6B v3",
}


def _model_label(full_name: str) -> str:
    """
    Human-readable label for a HuggingFace model ID.

    Uses the vendor's own capitalization for known models; falls back to
    a hyphen-split, title-cased form otherwise.

    :param full_name: full model identifier, e.g. ``'openai/whisper-large-v3'``.
    :returns: readable label, e.g. ``'Whisper large-v3'``.
    """
    if full_name in _MODEL_LABELS:
        return _MODEL_LABELS[full_name]
    name = full_name.rsplit("/", 1)[-1]
    parts = name.split("-")
    return " ".join(p.upper() if p.isalpha() and len(p) <= 3 else p.title() for p in parts)


def _model_colors(models: list[str]) -> dict[str, str]:
    """
    Assign a colour from :data:`_MODEL_PALETTE` to each model.

    :param models: ordered list of model identifiers.
    :returns: mapping of model identifier to hex colour string.
    """
    return {m: _MODEL_PALETTE[i % len(_MODEL_PALETTE)] for i, m in enumerate(models)}


def _config_label(row: pd.Series) -> str:
    """
    Build a multi-line label describing a pipeline configuration.

    The label contains the dataset name and audio filename stem, with
    ``+vad`` / ``+chunked`` suffixes when those options are enabled.

    :param row: a single row from the evaluation summary DataFrame.
    :returns: newline-separated label string suitable for tick labels.
    """
    stem = Path(row["filename"]).stem
    ds = DATASET_DISPLAY.get(row["dataset"], row["dataset"])
    label = f"{ds}\n({stem})"
    if row.get("vad"):
        label += "\n+vad"
    if row.get("chunked"):
        label += "\n+chunked"
    return label


def plot_baseline_wer(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate a grouped bar chart of WER by dataset configuration and model.

    Each unique (dataset, filename, vad, chunked) combination becomes an
    x-axis category, and each model gets one bar per category with the WER
    value annotated above.

    :param df: evaluation summary DataFrame as returned by
        :func:`~lyricscribe.evaluate.collect_evaluation_data`.
    :param output_path: file path to write the plot image to.
    """
    _apply_style()
    df = df.copy()

    models = sorted(df["model"].unique())
    colors = _model_colors(models)

    df["config"] = df.apply(_config_label, axis=1)
    configs = list(dict.fromkeys(df["config"]))

    x = np.arange(len(configs))
    n_models = len(models)
    w = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(max(9, 2 * len(configs)), 5))

    for i, model in enumerate(models):
        model_df = df[df["model"] == model]
        wer_map = model_df.groupby("config")["wer"].mean()
        vals = [wer_map.get(c, 0) for c in configs]

        offset = (i - (n_models - 1) / 2) * w
        bars = ax.bar(
            x + offset,
            [v * 100 for v in vals],
            w,
            label=_model_label(model),
            color=colors[model],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{v * 100:.0f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=colors[model],
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(configs, fontsize=10)
    ax.set_ylabel("Word Error Rate (%)", fontsize=11)
    ax.set_title(
        "Baseline WER by Dataset & Model", fontsize=13, fontweight="bold", pad=12
    )
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(frameon=False, fontsize=10)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved baseline WER plot -> {output_path}")


def plot_artifact_quartile_error(
    quartile_data: list[dict], output_path: Path
) -> None:
    _apply_style()

    rows = quartile_data
    if not rows:
        logger.warning("No quartile data found, skipping chart")
        return

    models = sorted(set(r["model"] for r in rows))
    colors = _model_colors(models)
    quartile_labels = ["Q1\n(cleanest)", "Q2", "Q3", "Q4\n(noisiest)"]
    quartile_keys = ["Q1", "Q2", "Q3", "Q4"]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(quartile_keys))

    for model in models:
        model_rows = {r["quartile"]: r for r in rows if r["model"] == model}
        vals = [
            float(model_rows[q]["error_rate"]) if q in model_rows else 0
            for q in quartile_keys
        ]

        ax.plot(
            x,
            [v * 100 for v in vals],
            "-o",
            color=colors[model],
            label=_model_label(model),
            linewidth=2.5,
            markersize=8,
            zorder=3,
        )
        if vals[0] > 0 and vals[-1] > 0:
            delta = (vals[-1] - vals[0]) * 100
            sign = "+" if delta >= 0 else ""
            ax.annotate(
                f"{sign}{delta:.1f}pp",
                xy=(3, vals[-1] * 100),
                xytext=(3.08, vals[-1] * 100),
                va="center",
                fontsize=8.5,
                color=colors[model],
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(quartile_labels, fontsize=10)
    ax.set_ylabel("WER (%)", fontsize=11)
    ax.set_title(
        "Error Rate Across Artifact Noise Quartiles",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.legend(frameon=False, fontsize=10)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved artifact quartile error plot -> {output_path}")


def plot_error_type_shares(df: pd.DataFrame, output_path: Path) -> None:
    _apply_style()
    df = df.copy()

    total_errors = df["substitutions"] + df["insertions"] + df["deletions"]
    df["insertion_share"] = df["insertions"] / total_errors
    df["deletion_share"] = df["deletions"] / total_errors
    df["substitution_share"] = df["substitutions"] / total_errors

    models = sorted(df["model"].unique())
    colors = _model_colors(models)

    share_cols = ["insertion_share", "deletion_share", "substitution_share"]
    share_labels = ["Insertions", "Deletions", "Substitutions"]
    share_colors = [_ERROR_COLORS["ins"], _ERROR_COLORS["del"], _ERROR_COLORS["sub"]]

    agg = df.groupby("model")[share_cols].mean().reindex(models)

    x = np.arange(len(models))
    n_bars = len(share_cols)
    w = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(models)), 5))

    for j, (col, label, color) in enumerate(zip(share_cols, share_labels, share_colors)):
        offset = (j - (n_bars - 1) / 2) * w
        vals = agg[col].values
        bars = ax.bar(
            x + offset,
            vals,
            w,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
                color=color,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([_model_label(m) for m in models], fontsize=10)
    ax.set_ylabel("Share of Total Errors", fontsize=11)
    ax.set_title(
        "Insertion, Deletion & Substitution Shares by Model",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.legend(frameon=False, fontsize=10)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved error type shares plot -> {output_path}")


def plot_wer_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate a heatmap of WER across models and pipeline configurations.

    Models are placed on the y-axis and pipeline configurations on the
    x-axis.  Each cell is colour-coded on a red-yellow-green scale
    (green = low WER, red = high WER) with the numeric WER annotated
    inside the cell.

    :param df: evaluation summary DataFrame as returned by
        :func:`~lyricscribe.evaluate.collect_evaluation_data`.
    :param output_path: file path to write the plot image to.
    """
    _apply_style()
    df = df.copy()

    def _compact_config(row: pd.Series) -> str:
        """
        Build a single-line config label for heatmap columns.

        :param row: a single row from the evaluation summary DataFrame.
        :returns: compact label string, e.g. ``'jam-alt / mix (+vad)'``.
        """
        stem = Path(row["filename"]).stem
        ds = DATASET_DISPLAY.get(row["dataset"], row["dataset"])
        label = f"{ds} / {stem}"
        flags = []
        if row.get("vad"):
            flags.append("vad")
        if row.get("chunked"):
            flags.append("chunked")
        if flags:
            label += f" (+{', '.join(flags)})"
        return label

    df["config"] = df.apply(_compact_config, axis=1)

    models = sorted(df["model"].unique())
    configs = list(dict.fromkeys(df["config"]))

    pivot = df.pivot_table(
        values="wer", index="model", columns="config", aggfunc="mean"
    )
    pivot = pivot.reindex(index=models, columns=configs)
    matrix = pivot.values * 100

    model_labels = [_model_label(m) for m in models]

    fig_width = max(8, 1.4 * len(configs))
    fig_height = max(4, 0.8 * len(models) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")

    for i in range(len(models)):
        for j in range(len(configs)):
            val = matrix[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val > 70 or val < 30 else "black"
            ax.text(
                j,
                i,
                f"{val:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=text_color,
            )

    ax.set_xticks(np.arange(len(configs)))
    ax.set_xticklabels(configs, fontsize=9, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(models)))
    ax.set_yticklabels(model_labels, fontsize=10)
    ax.set_title(
        "WER Across Models & Pipeline Configurations",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("WER (%)", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info(f"Saved WER heatmap -> {output_path}")


def plot_error_type_breakdown(df: pd.DataFrame, output_path: Path) -> None:
    """
    Stacked bar chart of error type percentages per model on the
    MUSDB-ALT clean-stems baseline.

    Restricted to one row per model: ``dataset == musdb_alt``,
    ``filename == vocals.wav``, no VAD, no chunking. Pooling across all
    configurations would let mix+VAD collapse runs (where Silero VAD
    misclassifies instrumental regions and drives WER to 72–85%) swamp
    the deletion counts, hiding the per-architecture profile the paper's
    Section 5 claims.

    :param df: evaluation summary DataFrame as returned by
        :func:`~lyricscribe.evaluate.collect_evaluation_data`.
    :param output_path: file path to write the plot image to.
    """
    _apply_style()

    baseline = df[
        (df["dataset"] == "musdb_alt")
        & (df["filename"] == "vocals.wav")
        & (~df["vad"])
        & (~df["chunked"])
    ]
    agg = baseline.groupby("model")[
        ["insertions", "deletions", "substitutions"]
    ].sum()
    models = agg.index.tolist()

    ins = agg["insertions"].values.astype(float)
    dels = agg["deletions"].values.astype(float)
    subs = agg["substitutions"].values.astype(float)
    totals = ins + dels + subs

    ins_p = ins / totals * 100
    dels_p = dels / totals * 100
    subs_p = subs / totals * 100

    model_labels = [_model_label(m).replace(" ", "\n", 1) for m in models]
    x = np.arange(len(models))
    bar_w = min(0.45, 0.8 / max(len(models), 1))

    fig, ax = plt.subplots(figsize=(max(6.5, 2 * len(models)), 5))
    ax.bar(
        x, subs_p, bar_w, label="Substitutions", color=_ERROR_COLORS["sub"], zorder=3
    )
    ax.bar(
        x,
        dels_p,
        bar_w,
        bottom=subs_p,
        label="Deletions",
        color=_ERROR_COLORS["del"],
        zorder=3,
    )
    ax.bar(
        x,
        ins_p,
        bar_w,
        bottom=subs_p + dels_p,
        label="Insertions",
        color=_ERROR_COLORS["ins"],
        zorder=3,
        linewidth=0,
    )

    for i in range(len(models)):
        mid_sub = subs_p[i] / 2
        mid_del = subs_p[i] + dels_p[i] / 2
        mid_ins = subs_p[i] + dels_p[i] + ins_p[i] / 2
        for mid, val in [
            (mid_sub, subs_p[i]),
            (mid_del, dels_p[i]),
            (mid_ins, ins_p[i]),
        ]:
            if val > 5:
                ax.text(
                    i,
                    mid,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    fontweight="bold",
                    color="white",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=10)
    ax.set_ylabel("% of total errors", fontsize=11)
    ax.set_title(
        "Error Type Breakdown by Model (MUSDB-ALT clean stems)",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_ylim(0, 105)
    ax.legend(
        frameon=False,
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
    )
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved error type breakdown plot -> {output_path}")


_PIPELINE_LABELS: dict[tuple[str, bool, bool], str] = {
    ("vocals.wav", False, False): "Clean Stems",
    ("htdemucs_ft_vocals.wav", False, False): "Demucs",
    ("htdemucs_ft_vocals.wav", True, False): "Demucs + VAD",
    ("mixture.wav", True, False): "VAD-guided Mix",
    ("mixture.wav", False, False): "Raw Mix",
    ("htdemucs_ft_vocals.wav", False, True): "Demucs + Chunked",
    ("htdemucs_ft_vocals.wav", True, True): "Demucs + VAD + Chunked",
    ("mixture.wav", True, True): "VAD-guided Mix + Chunked",
}


def plot_pipeline_shift(df: pd.DataFrame, output_path: Path) -> None:
    """
    Generate a scatter plot of pipeline error-profile shifts.

    Filters to ``musdb_alt`` and the key pipeline configurations, plots
    all models on a single axes colour-coded by model, with human-readable
    pipeline labels.  Axes are changes in insertion and deletion rate
    (errors per reference word) relative to the clean stems baseline
    (vocals.wav, no VAD, no chunking).

    :param df: evaluation summary DataFrame as returned by
        :func:`~lyricscribe.evaluate.collect_evaluation_data`.
    :param output_path: file path to write the plot image to.
    """
    _apply_style()
    df = df.copy()

    # Filter to musdb_alt (only dataset with clean stems baseline)
    df = df[df["dataset"] == "musdb_alt"]
    df = df.drop_duplicates(subset=["model", "filename", "vad", "chunked"])

    ref_words = df["hits"] + df["substitutions"] + df["deletions"]
    df["insertion_rate"] = df["insertions"] / ref_words
    df["deletion_rate"] = df["deletions"] / ref_words

    df["_key"] = list(zip(df["filename"], df["vad"], df["chunked"]))
    df = df[df["_key"].isin(_PIPELINE_LABELS)].copy()
    df["label"] = df["_key"].map(_PIPELINE_LABELS)

    models = sorted(df["model"].unique())
    colors = _model_colors(models)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    for model_id in models:
        m = df[df["model"] == model_id].copy()

        baseline = m[m["label"] == "Clean Stems"].iloc[0]
        m["d_insertion"] = m["insertion_rate"] - baseline["insertion_rate"]
        m["d_deletion"] = m["deletion_rate"] - baseline["deletion_rate"]

        color = colors[model_id]
        ax.scatter(
            m["d_insertion"],
            m["d_deletion"],
            s=90,
            color=color,
            edgecolors="white",
            linewidths=0.8,
            label=_model_label(model_id),
            zorder=3,
        )

        for _, row in m.iterrows():
            if row["label"] == "Clean Stems":
                continue
            ax.annotate(
                row["label"],
                (row["d_insertion"], row["d_deletion"]),
                fontsize=8,
                fontweight="bold",
                color=color,
                textcoords="offset points",
                xytext=(6, 6),
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
            )

    ax.axhline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(0, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_xlabel("Change in Insertion Rate (per reference word)", fontsize=11)
    ax.set_ylabel("Change in Deletion Rate (per reference word)", fontsize=11)
    ax.legend(
        fontsize=10,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(models),
        frameon=False,
    )
    ax.set_title(
        "Pipeline Shift vs. Clean Stems Baseline (MUSDB18)",
        fontsize=12,
        fontweight="bold",
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved pipeline shift plot -> {output_path}")


def generate_all_plots(
    jobs_dir: Path,
    output_dir: Path,
    word_dataset: list[dict] | None = None,
    df: pd.DataFrame | None = None,
) -> None:
    """
    Collect evaluation data from job directories and produce all analysis plots.

    Reads transcription job results from *jobs_dir*, computes evaluation
    metrics, and writes PDF plots into *output_dir*.  If *word_dataset*
    is provided, quartile analysis is computed in-memory and an additional
    artifact quartile error chart is generated.

    :param jobs_dir: root directory containing transcription job
        subdirectories with ``config.json`` and ``results*.jsonl`` files.
    :param output_dir: directory to write the generated PDF plot files
        into.  Created if it does not exist.
    :param word_dataset: optional word-level dataset as returned by
        :func:`~lyricscribe.transcribe.artifacts.correlation.build_dataset`.
        When provided, the artifact quartile error chart is included.
    :param df: pre-computed DataFrame from :func:`collect_evaluation_data`.
        When provided, job evaluation is skipped and this frame is used
        directly — useful when the caller has already paid that cost.
    """
    from lyricscribe.transcribe.artifacts.correlation import analyse

    if df is None:
        all_stats = collect_evaluation_data(jobs_dir)
        if not all_stats:
            logger.error("No evaluation data found in job directories.")
            return
        df = pd.DataFrame(all_stats)

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_error_type_breakdown(df, output_dir / "error_type_breakdown.pdf")
    plot_pipeline_shift(df, output_dir / "pipeline_shift.pdf")

    if word_dataset is not None:
        quartile_data = analyse(word_dataset)
        plot_artifact_quartile_error(
            quartile_data, output_dir / "artifact_quartile_error.pdf"
        )

    logger.info(f"All plots saved to {output_dir}")
