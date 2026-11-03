import json
import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)

analyze_app = typer.Typer(help="Scoring, reference self-WER, significance testing and checkpoint averaging")

# Heavy modules (jiwer/alt_eval, torch, transformers) are imported inside each
# command so ``lyricscribe --help`` stays fast.


def _write_json(path: Path | None, data) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {path}")


def _keep_set(selfwer_file: Path | None, max_self_wer: float):
    if selfwer_file is None:
        return None
    from lyricscribe.analysis.common import load_selfwer_keep

    keep = load_selfwer_keep(selfwer_file, max_self_wer)
    typer.echo(f"reference filter self_wer<={max_self_wer}: {len(keep)} songs kept")
    return keep


def _print_result(name: str, res: dict) -> None:
    from lyricscribe.analysis.scoring import format_stats

    if "error" in res:
        typer.echo(f"SKIP {name}: {res['error']}")
        return
    typer.echo(name)
    typer.echo("  " + format_stats("all", res["all"]))
    if "filtered" in res:
        typer.echo("  " + format_stats("filtered", res["filtered"]))
    if "raw" in res:
        typer.echo("  " + format_stats("raw (un-normalised)", res["raw"]))
    for lang, s in res.get("by_language", {}).items():
        typer.echo("    " + format_stats(lang, s))


_SELFWER_OPT = typer.Option(
    None, "--selfwer-file",
    help="Self-WER artifact (from `analyze selfwer`); adds the filtered slice.",
)
_MAX_SELFWER_OPT = typer.Option(
    0.3, "--max-self-wer", help="Inclusive self-WER cutoff."
)


@analyze_app.command("score")
def score(
    job_dirs: list[Path] = typer.Argument(..., help="Job dirs to score"),
    selfwer_file: Path | None = _SELFWER_OPT,
    max_self_wer: float = _MAX_SELFWER_OPT,
    common: bool = typer.Option(
        False, "--common",
        help="Score only songs common to all job dirs (compares partial runs "
             "without a song-set confound; disables the completeness check).",
    ),
    by_language: bool = typer.Option(False, "--by-language", help="Per-language breakdown."),
    raw: bool = typer.Option(
        False, "--raw",
        help="Also report un-normalised WER (the gap is surface style the "
             "metric discards).",
    ),
    allow_partial: bool = typer.Option(
        False, "--allow-partial",
        help="Score jobs with missing/failed songs or corrupt result lines "
             "(refused by default).",
    ),
    out: Path | None = typer.Option(None, "--out", help="Write results as JSON."),
):
    """
    Score job dirs with the evaluator's normalisation, refusing partial runs.
    """
    from lyricscribe.analysis.scoring import score_many

    keep = _keep_set(selfwer_file, max_self_wer)
    results = score_many(
        job_dirs, keep=keep, common=common, by_language=by_language, raw=raw,
        require_complete=not allow_partial,
    )
    for name, res in results.items():
        _print_result(name, res)
    _write_json(out, results)


@analyze_app.command("score-all")
def score_all(
    root: Path = typer.Argument(..., help="Root to search (e.g. the jobs dir)"),
    pattern: str = typer.Option(
        "**", "--pattern", help="Glob for job dirs relative to ROOT (default: all)."
    ),
    selfwer_file: Path | None = _SELFWER_OPT,
    max_self_wer: float = _MAX_SELFWER_OPT,
    out: Path | None = typer.Option(None, "--out", help="Write results as JSON."),
    top: int = typer.Option(20, "--top", help="Ranked rows to print."),
):
    """
    Score every complete run found under ROOT; partial/corrupt ones are listed.
    """
    from lyricscribe.analysis.scoring import discover_job_dirs, score_many

    keep = _keep_set(selfwer_file, max_self_wer)
    dirs = discover_job_dirs(root, pattern)
    typer.echo(f"candidate runs: {len(dirs)}")
    results = score_many(dirs, keep=keep)
    ok = {k: v for k, v in results.items() if "error" not in v}
    for k, v in results.items():
        if "error" in v:
            typer.echo(f"SKIP {k}: {v['error']}")
    rank_key = "filtered" if keep is not None else "all"
    typer.echo(f"\n=== {len(ok)} complete runs, ranked by {rank_key} WER ===")
    for name, v in sorted(ok.items(), key=lambda kv: kv[1][rank_key]["wer"])[:top]:
        extra = f"  (unfiltered {v['all']['wer']:.4f})" if keep is not None else ""
        typer.echo(f"  {v[rank_key]['wer']:.4f}{extra}  {name}")
    _write_json(out, results)


@analyze_app.command("selfwer")
def selfwer(
    dataset_dir: Path = typer.Argument(..., help="Dataset dir with per-song lyrics.json"),
    out: Path = typer.Option(..., "--out", help="Destination JSONL artifact"),
):
    """
    Build the per-song unsynced-vs-synced reference self-WER artifact.
    """
    from lyricscribe.analysis.refquality import build_selfwer_artifact

    n = build_selfwer_artifact(dataset_dir, out)
    typer.echo(f"wrote {n} songs to {out}")


@analyze_app.command("bootstrap")
def bootstrap(
    job_a: Path = typer.Argument(..., help="System A job dir"),
    job_b: Path = typer.Argument(..., help="System B job dir"),
    n_resamples: int = typer.Option(2000, "--n-resamples"),
    seed: int = typer.Option(0, "--seed"),
    selfwer_file: Path | None = _SELFWER_OPT,
    max_self_wer: float = _MAX_SELFWER_OPT,
):
    """
    Paired song-level bootstrap CI on WER(A) - WER(B).
    """
    from lyricscribe.analysis.stats import paired_bootstrap

    r = paired_bootstrap(job_a, job_b, n_resamples, seed, _keep_set(selfwer_file, max_self_wer))
    lo, hi = r["ci"]
    typer.echo(f"n={r['n']}  A={100 * r['wer_a']:.3f}%  B={100 * r['wer_b']:.3f}%  "
               f"delta={100 * r['delta']:+.3f}pt")
    typer.echo(f"95% CI [{100 * lo:+.3f}, {100 * hi:+.3f}] pt  P(sign flip)={r['p_sign_flip']:.3f}  "
               f"({n_resamples} resamples, seed {seed})")
    typer.echo("CI excludes 0" if r["excludes_zero"] else "CI spans 0: indistinguishable on this data")


@analyze_app.command("merge")
def merge(
    member: list[Path] = typer.Option(
        ..., "--member",
        help="Checkpoint dir (repeat). Config/tokenizer come from the FIRST.",
    ),
    out: Path = typer.Option(..., "--out", help="Output model dir"),
    weights: str | None = typer.Option(None, "--weights", help="Comma-separated; default uniform."),
    dtype: str = typer.Option("float16", "--dtype", help="Saved dtype."),
):
    """
    Weight-average finetunes of one base into a single model ("soup").
    """
    from lyricscribe.analysis.merge import merge_checkpoints

    w = [float(x) for x in weights.split(",")] if weights else None
    r = merge_checkpoints(member, out, w, dtype)
    typer.echo(f"wrote {out}: {r['tensors']} tensors, weights {r['weights']}")
    if r["dropped_keys"]:
        typer.echo(f"dropped (not in every member): {r['dropped_keys']}")
