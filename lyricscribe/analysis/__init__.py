"""
Scoring, reference self-WER, significance testing and checkpoint averaging.

Everything here scores through the same alt_eval normalisation as
:func:`lyricscribe.evaluate.evaluate_job`, so numbers are directly comparable
with the canonical evaluator. Exposed on the CLI as ``lyricscribe analyze``.
"""
