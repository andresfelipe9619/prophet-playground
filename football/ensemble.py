"""Combining a model's probabilities with the market's.

**Why this is the most informative thing in the package.** Asking "does the
model beat the market?" sets a bar almost nothing clears, and a no answers very
little: a model can be genuinely informative and still lose to a price that
already contains everything it knows plus team news, lineups and money. The
sharper question is *does the model know anything the market does not*, and a
blend answers it directly. A blend at weight 0 **is** the market, so it scores
exactly like the market and the paired test returns an effect of 0 — that is
not a degenerate case to guard against, it is the null the comparison is built
on. If putting weight on the model improves the score from there, the model
carries information the price does not, whether or not it could ever stand
alone.

Two pooling rules, because they disagree and the disagreement is informative.
Linear pooling averages the probabilities straight and hedges: the result always
lands between its inputs, component by component, and is never more confident
than the more confident source. Logarithmic pooling takes a weighted geometric
mean and renormalises, which leans harder on whatever both sources favour and is
far harsher on an outcome one of them nearly ruled out — a source saying 2%
drags the pool most of the way down instead of being averaged away. Report which
one you used; if a conclusion flips between them it is about the pooling rule,
not about the model. This mirrors `market.py`'s three de-margining methods
exactly.

Nothing here is fitted. The weight is a parameter the caller chooses and the
backtest measures — fitting it on the same matches you then score would be the
purest form of the leak this package exists to avoid.
"""

import numpy as np

from football.common import N_OUTCOMES

# Below this, a probability is treated as this small rather than as zero. The
# log pool takes a logarithm and a market that priced an outcome at 0 would
# otherwise veto it no matter what the model says.
_FLOOR = 1e-6


def _as_matrix(probs):
    return np.asarray(probs, dtype=float).reshape(-1, N_OUTCOMES)


def _check_weight(weight):
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be between 0 and 1, got {weight!r}")
    return float(weight)


def linear_blend(model_probs, market_probs, weight=0.5):
    """`weight` on the model, the rest on the market, averaged probability by probability.

    Hedges: the result is never more confident than the more confident input.
    """
    weight = _check_weight(weight)
    model, market = _as_matrix(model_probs), _as_matrix(market_probs)
    blended = weight * model + (1.0 - weight) * market
    return blended / blended.sum(axis=1, keepdims=True)


def logarithmic_pool(model_probs, market_probs, weight=0.5):
    """A weighted geometric mean of the two vectors, renormalised.

    Sharpens where the two agree, and punishes a source that was confidently
    wrong far harder than linear pooling does.
    """
    weight = _check_weight(weight)
    model = np.clip(_as_matrix(model_probs), _FLOOR, None)
    market = np.clip(_as_matrix(market_probs), _FLOOR, None)
    blended = np.exp(weight * np.log(model) + (1.0 - weight) * np.log(market))
    return blended / blended.sum(axis=1, keepdims=True)


POOLS = {"linear": linear_blend, "logarithmic": logarithmic_pool}


def blend(model_probs, market_probs, weight=0.5, pool="linear"):
    """Dispatch to a named pooling rule."""
    if pool not in POOLS:
        raise ValueError(f"Unknown pool {pool!r}. Expected one of {sorted(POOLS)}.")
    return POOLS[pool](model_probs, market_probs, weight)
