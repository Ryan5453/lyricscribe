"""
Paired song-level bootstrap between two job dirs.

WER is a ratio of sums, so each resample recomputes it from summed errors
and reference words rather than averaging per-song WERs. Pairing on songs
cancels song-difficulty variance.
"""

from pathlib import Path

import numpy as np

from lyricscribe.analysis.common import load_job, per_song_counts

DEFAULT_RESAMPLES = 2000
DEFAULT_SEED = 0


def paired_bootstrap(
    job_a: Path,
    job_b: Path,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    keep: set[str] | None = None,
) -> dict:
    """
    Bootstrap CI on WER(a) - WER(b) over the songs both jobs transcribed.

    :param job_a: First job dir.
    :param job_b: Second job dir.
    :param n_resamples: Bootstrap resamples.
    :param seed: RNG seed (``numpy.random.default_rng``).
    :param keep: Optional song keep-set (e.g. the self-WER<=0.3 slice).
    :return: ``n``, ``wer_a``, ``wer_b``, ``delta``, ``ci`` (2.5/97.5
        percentiles of the delta), ``p_sign_flip`` (fraction of resamples
        whose delta has the opposite sign to the point estimate) and
        ``excludes_zero``.
    """
    a, b = load_job(job_a), load_job(job_b)
    songs = sorted(
        s
        for s in set(a.hyps) & set(b.hyps)
        if a.hyps[s] is not None
        and b.hyps[s] is not None
        and a.refs.get(s)
        and (keep is None or s in keep)
    )
    if not songs:
        raise ValueError("no songs in common")
    refs = [a.refs[s] for s in songs]
    ea, wa = map(np.asarray, per_song_counts(refs, [a.normalized_hyp(s) for s in songs]))
    eb, wb = map(np.asarray, per_song_counts(refs, [b.normalized_hyp(s) for s in songs]))
    wer_a, wer_b = ea.sum() / wa.sum(), eb.sum() / wb.sum()

    rng = np.random.default_rng(seed)
    n = len(songs)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        deltas[i] = ea[idx].sum() / wa[idx].sum() - eb[idx].sum() / wb[idx].sum()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    flip = float((deltas >= 0).mean()) if wer_a < wer_b else float((deltas <= 0).mean())
    return {
        "n": n,
        "wer_a": float(wer_a),
        "wer_b": float(wer_b),
        "delta": float(wer_a - wer_b),
        "ci": [float(lo), float(hi)],
        "p_sign_flip": flip,
        "excludes_zero": not (lo <= 0 <= hi),
        "n_resamples": n_resamples,
        "seed": seed,
    }
