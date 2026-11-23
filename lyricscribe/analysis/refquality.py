import itertools
import json
import logging
import math
from pathlib import Path

import jiwer

from lyricscribe.ensemble.deloop import deloop
from lyricscribe.evaluate import _language_for, _normalized

logger = logging.getLogger(__name__)


def build_selfwer_artifact(dataset_dir: Path, output_path: Path) -> int:
    """
    Write the per-song unsynced-vs-synced self-WER artifact (JSONL).

    Rows: ``song_id``, ``self_wer`` (``None`` when either text is empty),
    ``lang``, ``unsynced_provider``, ``synced_provider``, ``ref_words``. A
    song whose lyrics.json can't be read gets an ``error`` row instead.

    :param dataset_dir: Dataset directory.
    :param output_path: Destination JSONL.
    :return: Number of songs written.
    """
    n = 0
    with open(output_path, "w") as f:
        for song_dir in Path(dataset_dir).iterdir():
            path = song_dir / "lyrics.json"
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text())
                # Not evaluate._language_for: changing the parse changes existing artifacts.
                lang = (data.get("detected_language") or "en").split("-")[0]
                unsynced = data.get("unsynced") or {}
                synced = data.get("synced") or {}
                ref_u = _normalized_or_empty(unsynced.get("data"), lang)
                synced_data = synced.get("data") or []
                synced_text = (
                    " ".join(x.get("text", "") for x in synced_data)
                    if isinstance(synced_data, list)
                    else str(synced_data or "")
                )
                ref_s = _normalized_or_empty(synced_text, lang)
                self_wer = None
                if ref_u and ref_s:
                    m = jiwer.process_words([ref_u], [ref_s])
                    ref_len = m.substitutions + m.deletions + m.hits
                    if ref_len:
                        self_wer = round(
                            (m.insertions + m.deletions + m.substitutions) / ref_len, 4
                        )
                row = {
                    "song_id": song_dir.name,
                    "self_wer": self_wer,
                    "lang": lang,
                    "unsynced_provider": unsynced.get("provider"),
                    "synced_provider": synced.get("provider"),
                    "ref_words": len(ref_u.split()),
                }
            except Exception as e:  # noqa: BLE001
                row = {"song_id": song_dir.name, "error": str(e)[:80]}
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def _normalized_or_empty(text, language: str) -> str:
    """
    Normalise *text*, treating missing or non-string text as empty.

    :param text: Raw transcript, possibly ``None``.
    :param language: Language code.
    :return: Normalised text.
    """
    if not isinstance(text, str) or not text:
        return ""
    return _normalized(text, language)


def _load_job_hyps(job_dir: Path, langs: dict[str, str], deloop_max_run: int | None = None) -> dict[str, str]:
    out: dict[str, str] = {}
    for results_path in sorted(Path(job_dir).glob("results*.jsonl")):
        for line in results_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = row.get("song_id")
            if sid not in langs:
                continue
            hyp = _normalized_or_empty(row.get("transcription"), langs[sid])
            if deloop_max_run and hyp:
                hyp = " ".join(deloop(hyp.split(), max_run=int(deloop_max_run)))
            out[sid] = hyp
    return out


def _pair_wer(ref: str, hyp: str) -> float:
    if not ref.strip():
        return float("nan")
    return float(jiwer.process_words([ref], [hyp]).wer)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0.0 or dy == 0.0:
        return None
    return num / (dx * dy)


def third_vote_audit(
    dataset_dir: Path,
    consensus_job: Path,
    member_jobs: dict[str, Path],
    *,
    selfwer_file: Path | None = None,
    deloop_max_run: int | None = None,
    suspect_wer_threshold: float = 0.5,
    out_path: Path | None = None,
    suspects_out: Path | None = None,
) -> dict:
    """
    Third-vote reference audit: validate ``self_wer`` against multi-architecture ASR
    consensus and flag suspect same-provider references where models agree with each other
    but diverge from the reference.

    :param dataset_dir: Dataset directory with per-song ``lyrics.json``.
    :param consensus_job: ROVER consensus job dir.
    :param member_jobs: ``{name: job_dir}`` for at least two diverse recognisers.
    :param selfwer_file: Optional JSONL artifact from :func:`build_selfwer_artifact`.
    :param deloop_max_run: Optional deloop cap applied to loaded hypotheses.
    :param suspect_wer_threshold: Minimum ``ref_vs_consensus`` WER to flag a consistent song.
    :param out_path: Optional JSONL path for per-song audit rows.
    :param suspects_out: Optional JSONL path for flagged same-provider suspect songs.
    :return: Summary dictionary with group means, Pearson ``r(self_wer, ref_vs_consensus)``,
        median system disagreement, and suspect count.
    """
    if len(member_jobs) < 2:
        raise ValueError("third_vote_audit requires at least two member jobs")

    meta: dict[str, dict] = {}
    if selfwer_file is not None:
        for line in Path(selfwer_file).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                meta[row["song_id"]] = row

    refs: dict[str, str] = {}
    langs: dict[str, str] = {}
    for song_dir in sorted(Path(dataset_dir).iterdir()):
        path = song_dir / "lyrics.json"
        if not song_dir.is_dir() or not path.exists():
            continue
        data = json.loads(path.read_text())
        lang = _language_for(data)
        langs[song_dir.name] = lang
        refs[song_dir.name] = _normalized_or_empty((data.get("unsynced") or {}).get("data"), lang)

    members = {name: _load_job_hyps(path, langs, deloop_max_run) for name, path in member_jobs.items()}
    consensus = _load_job_hyps(consensus_job, langs, deloop_max_run)
    member_names = list(members)

    rows: list[dict] = []
    for sid in sorted(refs):
        if not refs[sid] or sid not in consensus:
            continue
        hyps = [members[n][sid] for n in member_names if sid in members[n]]
        if len(hyps) < len(member_names):
            continue
        pair_vals = [_pair_wer(a, b) for a, b in itertools.combinations(hyps, 2)]
        finite_pairs = [v for v in pair_vals if not math.isnan(v)]
        disagreement = sum(finite_pairs) / len(finite_pairs) if finite_pairs else float("nan")
        info = meta.get(sid, {})
        u_prov = info.get("unsynced_provider")
        s_prov = info.get("synced_provider")
        same_provider = (u_prov is not None and u_prov == s_prov) if selfwer_file is not None else None
        rows.append({
            "song_id": sid,
            "language": langs[sid],
            "self_wer": info.get("self_wer"),
            "unsynced_provider": u_prov,
            "synced_provider": s_prov,
            "same_provider": same_provider,
            "system_disagreement": round(disagreement, 6) if not math.isnan(disagreement) else None,
            "ref_vs_consensus": round(_pair_wer(refs[sid], consensus[sid]), 6),
        })

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    valid_dis = sorted(r["system_disagreement"] for r in rows if r["system_disagreement"] is not None)
    median_dis = valid_dis[len(valid_dis) // 2] if valid_dis else 0.0

    same = [r for r in rows if r["same_provider"] is True]
    diff = [r for r in rows if r["same_provider"] is False]
    suspects = sorted(
        (
            r for r in same
            if r["system_disagreement"] is not None
            and r["system_disagreement"] <= median_dis
            and r["ref_vs_consensus"] >= suspect_wer_threshold
        ),
        key=lambda r: -r["ref_vs_consensus"],
    )

    if suspects_out is not None:
        suspects_out.parent.mkdir(parents=True, exist_ok=True)
        with open(suspects_out, "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in suspects)

    corr_pairs = [
        (r["self_wer"], r["ref_vs_consensus"])
        for r in diff
        if r["self_wer"] is not None and r["ref_vs_consensus"] is not None
    ]
    corr_r = _pearson([x for x, _ in corr_pairs], [y for _, y in corr_pairs])

    def _mean(group: list[dict], key: str) -> float | None:
        vals = [x[key] for x in group if x.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "n_songs": len(rows),
        "n_same_provider": len(same),
        "n_different_provider": len(diff),
        "median_system_disagreement": median_dis,
        "corr_selfwer_ref_vs_consensus_diff_provider": corr_r,
        "same_provider_mean_disagreement": _mean(same, "system_disagreement"),
        "same_provider_mean_ref_vs_consensus": _mean(same, "ref_vs_consensus"),
        "diff_provider_mean_disagreement": _mean(diff, "system_disagreement"),
        "diff_provider_mean_ref_vs_consensus": _mean(diff, "ref_vs_consensus"),
        "n_suspect_same_provider": len(suspects),
        "top_suspects": suspects[:15],
    }
