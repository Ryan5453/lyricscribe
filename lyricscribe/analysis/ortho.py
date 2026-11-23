import logging
import math
import random
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import jiwer

from lyricscribe.evaluate import job_song_pairs

logger = logging.getLogger(__name__)

DEFAULT_JUDGE_MODEL = "perplexity-ai/pplx-decider-v1.3"

_PPLX_INSTRUCTIONS = (
    "Are Form 1 and Form 2 the SAME word spelled differently in the given language, "
    "or DIFFERENT words?"
)
_PPLX_CRITERIA = {
    "true": (
        "SAME word spelled differently: dropped-g and apostrophe elisions "
        "(goin'/going, aquello'/aquellos), accent or diacritic differences "
        "(que/qué, solo/sólo), interjection/onomatopoeia spelling variants "
        "(woah/whoa, ooh/oh), or repeated-letter stretching (doo/do)."
    ),
    "false": (
        "DIFFERENT words: distinct lexical items, even if they sound alike "
        "or differ by one letter (want/what, da/la, van/va)."
    ),
}


def defold(word: str) -> str:
    """
    Strip Unicode combining diacritics from *word*.

    :param word: Input token.
    :return: Diacritic-stripped token.
    """
    return "".join(c for c in unicodedata.normalize("NFKD", word) if not unicodedata.combining(c))


def squash(word: str) -> str:
    """
    Collapse consecutive repeated characters (e.g. ``"ooooh" -> "oh"``).

    :param word: Input token.
    :return: Deduplicated token.
    """
    out: list[str] = []
    for c in word:
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def classify_pair(ref: str, hyp: str) -> str:
    """
    Conservatively classify one aligned substitution pair ``(ref, hyp)`` using purely
    structural rules (no hardcoded word lists).

    :param ref: Normalised reference word.
    :param hyp: Normalised hypothesis word.
    :return: One of ``"identical"``, ``"diacritic"``, ``"elision"``,
        ``"repeat-letters"``, or ``"genuine"``.
    """
    if ref == hyp:
        return "identical"
    if defold(ref) == defold(hyp):
        return "diacritic"
    r, h = ref.strip("'’"), hyp.strip("'’")
    if r == h:
        return "elision"
    if r + "g" == h or h + "g" == r:
        return "elision"
    if ref.endswith(("'", "’")) and h.startswith(r) and len(h) - len(r) <= 1:
        return "elision"
    if hyp.endswith(("'", "’")) and r.startswith(h) and len(r) - len(h) <= 1:
        return "elision"
    if squash(defold(r)) == squash(defold(h)):
        return "repeat-letters"
    return "genuine"


def audit_job(
    job_dir: Path,
    *,
    deloop_max_run: int | None = None,
    keep: set[str] | None = None,
    top_k: int = 15,
) -> dict:
    """
    Decompose every scored substitution of a transcription job into structural
    orthographic-convention classes vs. genuine recognition errors.

    Walks the exact ``jiwer.process_words`` alignment over alt-eval normalised tokens
    (matching :func:`~lyricscribe.evaluate.evaluate_job`).

    :param job_dir: Transcription job directory.
    :param deloop_max_run: Optional post-hoc deloop run cap matching scoring.
    :param keep: Optional reference-quality song ID keep-set.
    :param top_k: Number of most-frequent convention pairs to include in the summary.
    :return: Dict with WER, convention-adjusted WER, per-category counts, top convention
        pairs, and ``pairs`` (every distinct substitution pair including ``genuine``).
    :raises ValueError: If *job_dir* cannot be evaluated.
    """
    loaded = job_song_pairs(Path(job_dir), verbose=False, deloop_max_run=deloop_max_run)
    if loaded is None:
        raise ValueError(f"{job_dir}: no evaluable songs found")

    songs = loaded["songs"]
    if keep is not None:
        songs = [s for s in songs if s[0] in keep]
    if not songs:
        raise ValueError(f"{job_dir}: zero songs left after filtering")

    refs = [s[1] for s in songs]
    hyps = [s[2] for s in songs]
    langs = [s[3] for s in songs]
    aligned = jiwer.process_words(refs, hyps)

    ref_len = aligned.substitutions + aligned.deletions + aligned.hits
    total_err = aligned.substitutions + aligned.deletions + aligned.insertions

    cats: Counter[str] = Counter()
    conv_pairs: Counter[tuple[str, str, str]] = Counter()
    all_pairs: Counter[tuple[str, str, str]] = Counter()
    lang_counts: dict[tuple[str, str], Counter[str]] = {}

    for sent_align, r_str, h_str, lang in zip(aligned.alignments, refs, hyps, langs):
        rw, hw = r_str.split(), h_str.split()
        for chunk in sent_align:
            if chunk.type != "substitute":
                continue
            for i, j in zip(
                range(chunk.ref_start_idx, chunk.ref_end_idx),
                range(chunk.hyp_start_idx, chunk.hyp_end_idx),
            ):
                cls = classify_pair(rw[i], hw[j])
                cats[cls] += 1
                all_pairs[(cls, rw[i], hw[j])] += 1
                key = (rw[i], hw[j])
                if key not in lang_counts:
                    lang_counts[key] = Counter()
                lang_counts[key][lang] += 1
                if cls not in ("genuine", "identical"):
                    conv_pairs[(cls, rw[i], hw[j])] += 1

    conv_total = sum(v for k, v in cats.items() if k not in ("genuine", "identical"))
    wer = total_err / ref_len if ref_len else 0.0
    conv_wer_pts = conv_total / ref_len if ref_len else 0.0
    wer_excl = (total_err - conv_total) / ref_len if ref_len else 0.0

    distinct_pairs = [
        {
            "ref": r,
            "hyp": h,
            "count": n,
            "rule": c,
            "lang": lang_counts[(r, h)].most_common(1)[0][0],
        }
        for (c, r, h), n in all_pairs.items()
    ]
    distinct_pairs.sort(key=lambda d: (-d["count"], d["ref"], d["hyp"]))

    return {
        "job_dir": str(job_dir),
        "n_songs": len(songs),
        "ref_words": ref_len,
        "I": aligned.insertions,
        "D": aligned.deletions,
        "S": aligned.substitutions,
        "H": aligned.hits,
        "wer": wer,
        "convention_subs": conv_total,
        "convention_wer_pts": conv_wer_pts,
        "convention_error_share": conv_total / total_err if total_err else 0.0,
        "wer_excl_convention": wer_excl,
        "by_class": {
            k: {
                "count": v,
                "share_of_subs": v / max(1, aligned.substitutions),
                "wer_pts": v / ref_len if ref_len else 0.0,
            }
            for k, v in cats.most_common()
        },
        "top_convention_pairs": [
            {"rule": c, "ref": r, "hyp": h, "count": n}
            for (c, r, h), n in conv_pairs.most_common(top_k)
        ],
        "pairs": distinct_pairs,
    }


def _resolve_decider_checkpoint(model_name: str, revision: str | None) -> tuple[Path, str | None]:
    """Resolve a local dir or HF repo for a System-1 ``pplx-decider`` checkpoint."""
    from huggingface_hub import snapshot_download

    p = Path(model_name)
    if p.is_dir():
        ckpt, resolved_rev = p, revision
    else:
        ckpt = Path(snapshot_download(model_name, revision=revision))
        resolved_rev = revision or (ckpt.name if ckpt.parent.name == "snapshots" else None)

    if not ((ckpt / "decision_config.json").exists() and (ckpt / "source" / "src" / "pplx_decider").is_dir()):
        raise ValueError(
            f"{model_name}: expected a System-1 pplx-decider checkpoint containing "
            "decision_config.json and source/src/pplx_decider"
        )
    return ckpt, resolved_rev


def judge_pairs(
    pairs: list[dict],
    *,
    model_name: str = DEFAULT_JUDGE_MODEL,
    revision: str | None = None,
    batch_size: int = 64,
    min_count: int = 1,
    seed: int = 0,
) -> dict:
    """
    Blind local adjudication of distinct substitution pairs using a System-1 decision
    model (``perplexity-ai/pplx-decider-v1.3``).

    Pair order is randomly flipped per item under *seed* to prevent positional bias, and
    the judge sees every pair including those labelled ``genuine`` by structural rules.
    Both the raw logit margin (``true - false``) and calibrated ``prob_same`` are saved,
    along with the resolved HF commit hash for reproducibility.

    :param pairs: List of dicts from :func:`audit_job`'s ``"pairs"`` output.
    :param model_name: ``pplx-decider`` HF repo ID or local checkpoint dir.
    :param revision: Optional pinned HF revision (auto-recorded from snapshot if omitted).
    :param batch_size: Requests per batch.
    :param min_count: Skip pairs occurring fewer than *min_count* times.
    :param seed: RNG seed for pair-order randomisation.
    :return: Dict with per-pair verdicts, weighted confusion matrix vs. structural rules,
        agreement rate, and top missed/false-positive pairs.
    """
    import torch

    torch.manual_seed(seed)
    rng = random.Random(seed)
    selected = [p for p in pairs if p.get("count", 1) >= min_count]
    logger.info(f"Adjudicating {len(selected)} distinct pairs with {model_name} (rev={revision})")

    ckpt, resolved_rev = _resolve_decider_checkpoint(model_name, revision)
    src_dir = str(ckpt / "source" / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from pplx_decider import Decider

    device = "cuda" if torch.cuda.is_available() else "cpu"
    decider = Decider.from_pretrained(ckpt, device=device, mode="separate", locate=False)
    temp = float(decider.temperature)

    results: list[dict] = []
    for start in range(0, len(selected), batch_size):
        chunk = selected[start : start + batch_size]
        requests = []
        for p in chunk:
            flip = rng.random() < 0.5
            a, b = (p["hyp"], p["ref"]) if flip else (p["ref"], p["hyp"])
            requests.append({
                "state": f'Language: {p.get("lang") or "unknown"}\nForm 1: "{a}"\nForm 2: "{b}"',
                "questions": {
                    "same_word": {
                        "type": "noul",
                        "instructions": _PPLX_INSTRUCTIONS,
                        "criteria": _PPLX_CRITERIA,
                    }
                },
            })
        # In pplx_decider/protocol.py, noul option order is ["false", "true"].
        raw_batch = decider.logits(requests)
        for p, logit_map in zip(chunk, raw_batch):
            false_lp, true_lp = float(logit_map["same_word"][0]), float(logit_map["same_word"][1])
            margin = true_lp - false_lp
            prob_same = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, margin / temp))))
            results.append({
                **p,
                "judge": "SAME" if true_lp > false_lp else "DIFF",
                "margin": round(margin, 3),
                "prob_same": round(prob_same, 4),
            })

    cm: Counter[tuple[bool, bool]] = Counter()
    for r in results:
        rule_conv = r["rule"] not in ("genuine", "identical")
        judge_conv = r["judge"] == "SAME"
        cm[(rule_conv, judge_conv)] += r["count"]
    total = sum(cm.values())
    agree = cm[(True, True)] + cm[(False, False)]

    missed = sorted(
        (r for r in results if r["rule"] == "genuine" and r["judge"] == "SAME"),
        key=lambda r: -r["count"],
    )
    false_pos = sorted(
        (r for r in results if r["rule"] not in ("genuine", "identical") and r["judge"] == "DIFF"),
        key=lambda r: -r["count"],
    )

    return {
        "model": model_name,
        "revision": resolved_rev,
        "temperature": temp,
        "seed": seed,
        "n_distinct_pairs": len(results),
        "total_occurrences": total,
        "agreement": agree / total if total else 0.0,
        "confusion": {
            "both_convention": cm[(True, True)],
            "rule_only_fp": cm[(True, False)],
            "judge_only_missed": cm[(False, True)],
            "both_genuine": cm[(False, False)],
        },
        "summary_subs": {
            "rules_only": cm[(True, True)] + cm[(True, False)],
            "judge_only": cm[(True, True)] + cm[(False, True)],
            "both_agree": cm[(True, True)],
            "either": cm[(True, True)] + cm[(True, False)] + cm[(False, True)],
        },
        "top_missed": missed[:40],
        "top_false_positives": false_pos[:25],
        "results": results,
    }
