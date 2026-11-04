import json
import logging
from pathlib import Path

import typer

logger = logging.getLogger(__name__)

ensemble_app = typer.Typer(help="System combination (ROVER) over transcription job dirs")

_SPEC_HELP = (
    "Ensemble spec (YAML/JSON): datasets, systems -> per-dataset job dirs, "
    "and ordered subsets. Format: see lyricscribe.ensemble.spec.load_spec."
)
_SYSTEM_HELP = (
    "Ad-hoc system: name=job_dir (requires a single --dataset) or "
    "name:dataset=job_dir. Repeatable."
)
_DATASET_DIR_HELP = "Ad-hoc dataset: name=dataset_dir. Repeatable."


def _build_spec(spec_path, systems, dataset_dirs, datasets):
    from lyricscribe.ensemble.spec import load_spec

    spec = load_spec(spec_path)
    for arg in dataset_dirs or []:
        name, sep, path = arg.partition("=")
        if not sep:
            raise typer.BadParameter(f"--dataset-dir must be name=dir, got {arg!r}")
        spec.add_dataset(name, Path(path))
    for arg in systems or []:
        head, sep, path = arg.partition("=")
        if not sep:
            raise typer.BadParameter(f"--system must be name=job_dir, got {arg!r}")
        name, colon, ds = head.partition(":")
        if not colon:
            if len(datasets) != 1:
                raise typer.BadParameter(
                    f"--system {arg!r} has no dataset; use name:dataset=job_dir "
                    "when combining several datasets"
                )
            ds = datasets[0]
        spec.add_system(f"{name}={path}", ds)
    return spec


def _load(spec, dataset, members):
    """
    Reference tokens, languages and normalised hypotheses for *members*.
    """
    from lyricscribe.ensemble import rover

    if dataset not in spec.datasets:
        raise typer.BadParameter(f"unknown dataset {dataset!r}; add it to the spec or pass --dataset-dir")
    ds = spec.datasets[dataset]
    refs, langs = rover.load_references(ds.dir)
    missing = [m for m in members if dataset not in spec.systems.get(m, {})]
    if missing:
        raise typer.BadParameter(f"{dataset}: no hypotheses registered for {sorted(missing)}")
    hyps = {m: rover.load_hypotheses(spec.job_dir(m, dataset), langs, ds.expected) for m in dict.fromkeys(members)}
    return ds, refs, hyps


def _members(spec, subset):
    try:
        return spec.members(subset)
    except KeyError as e:
        raise typer.BadParameter(str(e)) from None


def _pct(x: float) -> str:
    return f"{100 * x:.3f}%"


def _ids(s: dict) -> str:
    e = s["I"] + s["D"] + s["S"]
    if not e:
        return "-"
    return f"{100 * s['I'] / e:.1f}/{100 * s['D'] / e:.1f}/{100 * s['S'] / e:.1f}"


@ensemble_app.command("rover")
def rover_cmd(
    subset: list[str] = typer.Option(
        ..., "--subset",
        help="Subset name from the spec, or comma-separated members (order = "
             "alignment order; ties break toward earlier members). Repeatable.",
    ),
    dataset: list[str] = typer.Option(..., "--dataset", help="Dataset name. Repeatable."),
    out: Path = typer.Option(..., "--out", help="Writes <out>/<tag>/vote/<dataset> job dirs."),
    spec_path: Path | None = typer.Option(None, "--spec", help=_SPEC_HELP),
    system: list[str] = typer.Option(None, "--system", help=_SYSTEM_HELP),
    dataset_dir: list[str] = typer.Option(None, "--dataset-dir", help=_DATASET_DIR_HELP),
    orderings: int = typer.Option(
        1, "--orderings", min=1,
        help="ROVER-of-orderings: vote under N member orderings (the given order "
             "plus N-1 seeded shuffles) and re-vote the N outputs. 1 = plain "
             "ROVER.",
    ),
    seed: int = typer.Option(0, "--seed", help="Seed for --orderings shuffles."),
    null_weight: float = typer.Option(
        1.0, "--null-weight",
        help="Weight of a NULL vote. 1.0 = plain ROVER.",
    ),
    oracle: bool = typer.Option(True, "--oracle/--no-oracle", help="Also report the oracle WER."),
    tag: str | None = typer.Option(
        None, "--tag",
        help="Output dir name (default: subset name, plus _ORDER<N> with --orderings).",
    ),
    workers: int = typer.Option(16, "--workers", min=1, help="Processes for alignment."),
):
    """
    Combine systems by ROVER voting and write the result as a standard job dir.

    Combined outputs are themselves valid members, so a meta-ensemble (vote
    over ensemble outputs) or two-level ROVER is just a spec whose systems
    point at earlier ROVER output dirs.

    Example: lyricscribe ensemble rover --spec ensemble.yaml
    --subset my_subset --dataset jam_alt --out jobs/rover
    """
    import numpy as np

    from lyricscribe.ensemble import rover
    from lyricscribe.evaluate import evaluate_job

    spec = _build_spec(spec_path, system, dataset_dir, dataset)
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for ds_name in dataset:
        members_by_subset = {s: _members(spec, s) for s in subset}
        wanted = [m for ms in members_by_subset.values() for m in ms]
        ds, refs, hyps = _load(spec, ds_name, wanted)
        for name, members in members_by_subset.items():
            if len(members) < 3:
                logger.warning(
                    f"{name}: {len(members)}-system ROVER cannot break ties without "
                    "confidences; it reproduces the first member exactly"
                )
            orders = rover.orderings(members, orderings, seed)
            res = rover.run_orders(orders, hyps, refs, oracle=oracle,
                                   null_weight=null_weight, workers=workers)
            label = tag or (name if orderings == 1 else f"{name}_ORDER{orderings}")
            if "," in label:
                label = label.replace(",", "+")
            entry = {"members": members, "dataset": ds_name, "orderings": orderings,
                     "seed": seed, "null_weight": null_weight}
            ref_words = sum(len(refs[s]) for s in refs)

            if orderings == 1:
                texts = {s: " ".join(v[0]) for s, (v, _) in res.items()}
            else:
                per_order = [rover.score(refs, {s: v[i] for s, (v, _) in res.items()})["wer"]
                             for i in range(len(orders))]
                texts = {s: " ".join(rover.combine_orderings(v)) for s, (v, _) in res.items()}
                entry["per_order_wer"] = per_order
                entry["orders"] = orders
                print(f"{label} {ds_name}: {len(orders)} orderings  "
                      f"min {_pct(min(per_order))} max {_pct(max(per_order))} "
                      f"range {100 * (max(per_order) - min(per_order)):.3f}pt "
                      f"sd {100 * float(np.std(per_order)):.3f}pt  "
                      f"given-order {_pct(per_order[0])}")
            if oracle:
                errs = [sum(o[i] for _, o in res.values()) for i in range(len(orders))]
                entry["oracle_wer"] = errs[0] / ref_words if ref_words else 0.0
                if orderings > 1:
                    entry["oracle_wer_per_order"] = [e / ref_words for e in errs]
                print(f"{label} {ds_name}: ORACLE WER={entry['oracle_wer']:.4f} "
                      f"({errs[0]}/{ref_words})")

            job_dir = out / label / "vote" / ds_name
            rover.write_vote_job(job_dir, ds.dir, members, texts, extra_config={
                "rover_orderings": orderings, "rover_seed": seed,
                "rover_null_weight": null_weight,
            })
            stats = evaluate_job(job_dir, verbose=False)
            if stats is None:
                raise RuntimeError(f"{job_dir}: evaluator returned nothing")
            entry.update(wer=stats["wer"], n_songs=stats["n_songs"], I=stats["insertions"],
                         D=stats["deletions"], S=stats["substitutions"], job_dir=str(job_dir))
            print(f"{label} {ds_name}: VOTE WER={stats['wer']:.5f} "
                  f"I/D/S={_ids(entry)} n={stats['n_songs']} -> {job_dir}", flush=True)
            summary[f"{label}/{ds_name}"] = entry
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")


@ensemble_app.command("deloop")
def deloop_cmd(
    src: Path = typer.Argument(..., help="Source job dir (config.json + results.jsonl)."),
    out: Path = typer.Option(..., "--out", help="Destination job dir."),
    max_run: int = typer.Option(5, "--max-run", min=1, help="Repeats kept of a looping cycle."),
    max_cycle: int = typer.Option(25, "--max-cycle", min=1, help="Longest cycle length, in words."),
):
    """
    Collapse runaway repetition in an existing job's transcriptions.

    Example: lyricscribe ensemble deloop jobs/qwen3omni/jam_alt
    --out jobs/qwen3omni_deloop/jam_alt --max-run 5
    """
    from lyricscribe.ensemble.deloop import deloop_job

    changed, removed = deloop_job(src, out, max_run, max_cycle)
    print(f"de-looped {changed} songs, removed {removed} tokens -> {out}")
