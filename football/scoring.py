"""Proper scoring rules for the three-way match outcome.

This is the football counterpart of the lottery's set-based hit counting: the
number that says how good a probabilistic forecast is. The outcome is treated
as ORDERED — home win, draw, away win, in that order — so the default metric
is the ranked probability score, which penalises a forecast that put its mass
on the draw when the away side won less than one that put its mass on the home
win. Brier is offered too but is symmetric across outcomes and does not make
that distinction; log-loss is offered for the confidently-wrong case.

`skill_score(model, baseline, ...)` is the headline "vs the market" figure:
positive means the model scored better than the baseline on the same matches.

Every function takes `probs` as `(n, 3)` (or `(3,)`) in `football.common.OUTCOMES`
order. Getting that order wrong corrupts RPS silently, which is why the order
lives in one place and is imported, never redeclared.
"""

import numpy as np

from football.common import OUTCOMES, outcome_index

METRICS = ("brier", "rps", "log_loss")
_LOG_CLIP = 1e-15


def _as_matrix(probs):
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        probs = probs[None, :]
    if probs.ndim != 2 or probs.shape[1] != len(OUTCOMES):
        raise ValueError(
            f"Expected probabilities shaped (n, {len(OUTCOMES)}) in {OUTCOMES} order, "
            f"got {probs.shape}."
        )
    return probs


def _onehot(outcomes):
    idx = np.array([outcome_index(o) for o in outcomes])
    out = np.zeros((idx.size, len(OUTCOMES)))
    out[np.arange(idx.size), idx] = 1.0
    return out


def _require_finite(probs):
    if np.isnan(probs).any():
        raise ValueError(
            "probs contains NaN. The single-forecast metrics score every row; for a "
            "frame with partial market coverage use skill_score, which drops unmatched rows."
        )


def per_match_scores(probs, outcomes, metric="rps"):
    """The per-match score contribution — the quantity averaged by the aggregate metrics.

    `evaluation.py` needs these one-per-match so it can form a paired difference
    between the model and the market and test its mean against zero.
    """
    p = _as_matrix(probs)
    y = _onehot(outcomes)
    if p.shape[0] != y.shape[0]:
        raise ValueError(f"{p.shape[0]} probability rows but {y.shape[0]} outcomes.")
    if metric == "brier":
        return np.sum((p - y) ** 2, axis=1)
    if metric == "rps":
        cp = np.cumsum(p, axis=1)[:, :-1]
        cy = np.cumsum(y, axis=1)[:, :-1]
        return np.sum((cp - cy) ** 2, axis=1) / (len(OUTCOMES) - 1)
    if metric == "log_loss":
        return -np.sum(y * np.log(np.clip(p, _LOG_CLIP, 1.0)), axis=1)
    raise ValueError(f"Unknown metric {metric!r}. Available: {METRICS}.")


def brier_score(probs, outcomes):
    """Mean squared error between the probability vector and the outcome indicator."""
    _require_finite(_as_matrix(probs))
    return float(per_match_scores(probs, outcomes, "brier").mean())


def ranked_probability_score(probs, outcomes):
    """Mean squared error between the cumulative forecast and cumulative outcome.

    0 is perfect. For a 3-way outcome a uniform forecast scores 1/9. Lower is
    better, and a near miss (mass on the adjacent outcome) costs less than a
    far one — the property Brier lacks.
    """
    _require_finite(_as_matrix(probs))
    return float(per_match_scores(probs, outcomes, "rps").mean())


def log_loss(probs, outcomes):
    """Mean negative log probability assigned to the outcome that happened."""
    _require_finite(_as_matrix(probs))
    return float(per_match_scores(probs, outcomes, "log_loss").mean())


_AGGREGATE = {"brier": brier_score, "rps": ranked_probability_score, "log_loss": log_loss}


def skill_score(model_probs, baseline_probs, outcomes, metric="rps"):
    """`1 - score(model) / score(baseline)` on the matches both can score.

    Positive means the model beat the baseline. Rows where either side has a
    NaN (a match with no market price, say) are dropped from both before
    scoring, so the two are always compared on the same matches.
    """
    if metric not in _AGGREGATE:
        raise ValueError(f"Unknown metric {metric!r}. Available: {METRICS}.")
    model = _as_matrix(model_probs)
    baseline = _as_matrix(baseline_probs)
    outcomes = np.asarray(list(outcomes))
    keep = ~(np.isnan(model).any(axis=1) | np.isnan(baseline).any(axis=1))
    if not keep.any():
        return float("nan")
    fn = _AGGREGATE[metric]
    base = fn(baseline[keep], outcomes[keep])
    if base == 0:
        return float("nan")
    return float(1.0 - fn(model[keep], outcomes[keep]) / base)
