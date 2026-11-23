import json
import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)

analyze_app = typer.Typer(help="Scoring, reference self-WER, significance testing and checkpoint averaging")


def _write_json(path: Path | None, data) -> None:
    """
    Write *data* as indented JSON, creating parent dirs; no-op without a path.

    :param path: Destination file, or ``None`` to skip.
    :param data: JSON-serialisable object.
    """
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        logger.info(f"Wrote {path}")


def _keep_set(selfwer_file: Path | None, max_self_wer: float) -> set[str] | None:
    """
    Load the reference-quality keep-set and report its size.

    :param selfwer_file: Self-WER artifact, or ``None`` for no filter.
    :param max_self_wer: Inclusive self-WER cutoff.
    :return: Song IDs to keep, or ``None`` when no artifact is given.
    """
    if selfwer_file is None:
        return None
    from lyricscribe.analysis.common import load_selfwer_keep

    keep = load_selfwer_keep(selfwer_file, max_self_wer)
    typer.echo(f"reference filter self_wer<={max_self_wer}: {len(keep)} songs kept")
    return keep


def _format_stats(label: str, s: dict) -> str:
    """
    One-line human summary of a stats dict.

    :param label: Row label.
    :param s: Stats from :func:`~lyricscribe.analysis.common.corpus_stats`.
    :return: The summary line.
    """
    sh = s["share"]
    return (
        f"{label}: n={s['n']} WER={s['wer']:.4f} "
        f"I/D/S={sh['I']:.1f}/{sh['D']:.1f}/{sh['S']:.1f} "
        f"(I={s['I']} D={s['D']} S={s['S']})"
    )


def _print_result(name: str, res: dict) -> None:
    """
    Print one job's scores, or why it was skipped.

    :param name: Job label.
    :param res: Result from :func:`~lyricscribe.analysis.scoring.score_many`.
    """
    if "error" in res:
        typer.echo(f"SKIP {name}: {res['error']}")
        return
    typer.echo(name)
    typer.echo("  " + _format_stats("all", res["all"]))
    if "filtered" in res:
        typer.echo("  " + _format_stats("filtered", res["filtered"]))
    if "raw" in res:
        typer.echo("  " + _format_stats("raw (un-normalised)", res["raw"]))
    for lang, s in res.get("by_language", {}).items():
        typer.echo("    " + _format_stats(lang, s))


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


@analyze_app.command("ortho")
def ortho(
    job_dirs: list[Path] = typer.Argument(..., help="Job dirs to audit for orthographic conventions"),
    deloop: int | None = typer.Option(None, "--deloop", min=1, help="Post-hoc deloop max run."),
    selfwer_file: Path | None = _SELFWER_OPT,
    max_self_wer: float = _MAX_SELFWER_OPT,
    dump_pairs_dir: Path | None = typer.Option(
        None, "--dump-pairs-dir",
        help="Write per-job distinct substitution pairs JSON here (input for `analyze judge-pairs`).",
    ),
    out: Path | None = typer.Option(None, "--out", help="Write full audit summary JSON."),
):
    """
    Decompose scored substitutions into structural orthographic conventions vs. genuine errors (T23).
    """
    from lyricscribe.analysis.ortho import audit_job

    keep = _keep_set(selfwer_file, max_self_wer)
    summaries: dict[str, dict] = {}
    if dump_pairs_dir is not None:
        dump_pairs_dir.mkdir(parents=True, exist_ok=True)

    for d in job_dirs:
        res = audit_job(d, deloop_max_run=deloop, keep=keep)
        pairs = res.pop("pairs")
        summaries[str(d)] = res
        typer.echo(f"\n=== {d} ===")
        typer.echo(
            f"  WER={100 * res['wer']:.3f}%  convention={100 * res['convention_wer_pts']:.3f}pt "
            f"({100 * res['convention_error_share']:.2f}% of errors)  "
            f"WER_excl_conv={100 * res['wer_excl_convention']:.3f}%  (n={res['n_songs']})"
        )
        for cls, info in res["by_class"].items():
            typer.echo(
                f"    {cls:15s} {info['count']:7d}  "
                f"({100 * info['share_of_subs']:5.2f}% of subs, {100 * info['wer_pts']:.3f} WER pts)"
            )
        if dump_pairs_dir is not None:
            safe_name = str(d).strip("/").replace("/", "_")
            pairs_path = dump_pairs_dir / f"ortho_pairs_{safe_name}.json"
            pairs_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=1) + "\n")
            typer.echo(f"  wrote {len(pairs)} distinct pairs -> {pairs_path}")

    _write_json(out, summaries)


@analyze_app.command("judge-pairs")
def judge_pairs_cmd(
    pairs_file: Path = typer.Argument(..., help="Pairs JSON written by `analyze ortho --dump-pairs-dir`"),
    out: Path = typer.Option(..., "--out", help="Destination JSON with per-pair verdicts and summary"),
    model: str = typer.Option(
        "perplexity-ai/pplx-decider-v1.3", "--model",
        help="System-1 pplx-decider HF repo ID or local checkpoint dir.",
    ),
    revision: str | None = typer.Option(
        None, "--revision",
        help="Optional pinned HF revision (auto-recorded in the output JSON if unset).",
    ),
    batch_size: int = typer.Option(64, "--batch-size", min=1),
    min_count: int = typer.Option(1, "--min-count", min=1),
    seed: int = typer.Option(0, "--seed"),
):
    """
    Blind System-1 adjudication of substitution pairs via pplx-decider logits (T23b).
    """
    from lyricscribe.analysis.ortho import judge_pairs

    pairs = json.loads(pairs_file.read_text())
    res = judge_pairs(
        pairs,
        model_name=model,
        revision=revision or None,
        batch_size=batch_size,
        min_count=min_count,
        seed=seed,
    )
    _write_json(out, res)
    cm = res["confusion"]
    typer.echo(
        f"adjudicated {res['n_distinct_pairs']} pairs ({res['total_occurrences']} occurrences) | "
        f"weighted agreement={100 * res['agreement']:.2f}%"
    )
    typer.echo(
        f"  both_conv={cm['both_convention']}  rule_only_fp={cm['rule_only_fp']}  "
        f"judge_only_missed={cm['judge_only_missed']}  both_genuine={cm['both_genuine']}"
    )


@analyze_app.command("third-vote")
def third_vote_cmd(
    dataset_dir: Path = typer.Argument(..., help="Dataset dir with per-song lyrics.json"),
    consensus: Path = typer.Option(..., "--consensus", help="ROVER consensus job dir"),
    member: list[str] = typer.Option(
        ..., "--member", help="Member job as name=job_dir (repeat at least twice)."
    ),
    selfwer_file: Path | None = _SELFWER_OPT,
    deloop: int | None = typer.Option(None, "--deloop", min=1),
    suspect_threshold: float = typer.Option(0.5, "--suspect-threshold"),
    out: Path | None = typer.Option(None, "--out", help="Write per-song audit JSONL"),
    suspects_out: Path | None = typer.Option(None, "--suspects-out", help="Write suspect same-provider JSONL"),
    summary_out: Path | None = typer.Option(None, "--summary-out", help="Write aggregate summary JSON"),
):
    """
    Validate reference self-WER against multi-architecture ASR consensus (T8 third-vote audit).
    """
    from lyricscribe.analysis.refquality import third_vote_audit

    member_jobs: dict[str, Path] = {}
    for arg in member:
        name, sep, path = arg.partition("=")
        if not sep:
            raise typer.BadParameter(f"--member must be name=job_dir, got {arg!r}")
        member_jobs[name] = Path(path)

    res = third_vote_audit(
        dataset_dir,
        consensus,
        member_jobs,
        selfwer_file=selfwer_file,
        deloop_max_run=deloop,
        suspect_wer_threshold=suspect_threshold,
        out_path=out,
        suspects_out=suspects_out,
    )
    typer.echo(
        f"songs={res['n_songs']}  same_provider={res['n_same_provider']}  "
        f"diff_provider={res['n_different_provider']}  "
        f"median_disagreement={res['median_system_disagreement']:.4f}"
    )
    if res["corr_selfwer_ref_vs_consensus_diff_provider"] is not None:
        typer.echo(
            f"corr(self_wer, ref_vs_consensus) = {res['corr_selfwer_ref_vs_consensus_diff_provider']:.4f}"
        )
    typer.echo(f"suspect same-provider refs: {res['n_suspect_same_provider']}")
    _write_json(summary_out, res)


@analyze_app.command("readability")
def readability_cmd(
    job_dirs: list[Path] = typer.Argument(..., help="Job dirs to score with Jam-ALT readability metrics"),
    selfwer_file: Path | None = _SELFWER_OPT,
    max_self_wer: float = _MAX_SELFWER_OPT,
    by_language: bool = typer.Option(False, "--by-language", help="Per-language readability breakdown."),
    allow_partial: bool = typer.Option(False, "--allow-partial", help="Allow incomplete job dirs."),
    out: Path | None = typer.Option(None, "--out", help="Write readability JSON."),
):
    """
    Compute Jam-ALT readability metrics (WER, case-sensitive WER, punctuation/line/section F1).
    """
    from lyricscribe.analysis.scoring import readability_many

    keep = _keep_set(selfwer_file, max_self_wer)
    results = readability_many(
        job_dirs, keep=keep, by_language=by_language, require_complete=not allow_partial
    )
    for name, res in results.items():
        if "error" in res:
            typer.echo(f"SKIP {name}: {res['error']}")
            continue
        m = res["all"]
        parts = [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in sorted(m.items())]
        typer.echo(f"{name} (n={res['n_songs']}): " + "  ".join(parts))
    _write_json(out, results)


@analyze_app.command("examples")
def examples_cmd(
    job_dirs: list[Path] = typer.Argument(..., help="Job dirs to compare side-by-side"),
    song: list[str] | None = typer.Option(None, "--song", help="Explicit song ID (repeatable)."),
    label: list[str] | None = typer.Option(None, "--label", help="Display label matching each job dir."),
    top_diff: int = typer.Option(10, "--top-diff", min=1, help="Top songs by cross-system WER spread."),
    deloop: int | None = typer.Option(None, "--deloop", min=1),
    max_words: int = typer.Option(80, "--max-words", min=10),
    out: Path | None = typer.Option(None, "--out", help="Write side-by-side comparison JSON."),
):
    """
    Export qualitative side-by-side multi-model song excerpts for error analysis.
    """
    from lyricscribe.analysis.scoring import qualitative_examples

    records = qualitative_examples(
        job_dirs,
        songs=song,
        top_diff=top_diff,
        deloop_max_run=deloop,
        labels=label,
        max_words=max_words,
    )
    for rec in records:
        typer.echo(
            f"\n[{rec['song_id']} | {rec['language']} | words={rec['ref_words']} | "
            f"WER spread={100 * rec['wer_spread']:.1f}pt]"
        )
        typer.echo(f"  REF: {rec['ref_excerpt']}")
        for sys_name, info in rec["systems"].items():
            typer.echo(
                f"  {sys_name} (WER={100 * info['wer']:.1f}% I/D/S={info['I']}/{info['D']}/{info['S']}): "
                f"{info['excerpt']}"
            )
    _write_json(out, records)
