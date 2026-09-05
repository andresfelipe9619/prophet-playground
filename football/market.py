"""The betting market as a baseline: decimal odds in, calibrated probabilities out.

This is football's `lottery/models/baseline.py`. There the baseline was the
exact hypergeometric distribution, computable from the rules with no data.
Here it has to be estimated, because the thing a football model must beat is
not a law of probability but other people's opinions, priced.

**Odds are not probabilities.** Decimal odds of 2.00 look like a 50% chance,
but the three implied probabilities of a match sum to more than 1 — typically
1.02 to 1.08. That excess is the *overround*, the bookmaker's margin, and it
has to be removed before the prices mean anything as a forecast. A model
compared against raw `1/odds` is being compared against a baseline that is
deliberately wrong in the bookmaker's favour, and will look better than it is.

**How the margin is removed is a modelling choice, not a detail.** Three
methods are offered because they disagree most exactly where it matters, on
longshots:

- `multiplicative` divides every implied probability by the overround. Simple,
  standard, and the default. It assumes the margin is applied proportionally,
  which overstates the true probability of longshots — bookmakers load more
  margin onto them.
- `additive` subtracts the excess equally across outcomes. It corrects the
  longshot bias in the opposite direction, and can produce a negative
  probability on an extreme favourite, which is why it is not the default.
- `power` solves for the exponent that normalises `p^k`. It fits the observed
  longshot bias better than either, at the cost of a root-find per match.

None is "correct". The honest move is to pick one, say which, and check that
a conclusion does not flip when you switch — which is what
`compare_methods` is for.
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from football.common import ODDS_COLUMNS, OUTCOMES, PROBABILITY_COLUMNS

METHODS = ("multiplicative", "additive", "power")
DEFAULT_METHOD = "multiplicative"


def overround(odds):
    """Bookmaker margin: how much the implied probabilities exceed 1.

    Returns 0.0 for a fair book, ~0.05 for a typical football market. A
    negative value means the prices are beatable by backing every outcome,
    which does not happen inside one bookmaker and does happen across several
    — so it is reported rather than clipped.
    """
    odds = np.asarray(odds, dtype=float)
    return float(np.sum(1.0 / odds, axis=-1) - 1.0) if odds.ndim == 1 else np.sum(1.0 / odds, axis=-1) - 1.0


def _multiplicative(raw):
    return raw / raw.sum(axis=-1, keepdims=True)


def _additive(raw):
    excess = raw.sum(axis=-1, keepdims=True) - 1.0
    return raw - excess / raw.shape[-1]


def _power_exponent(raw_row):
    """Solve sum(p_i ** k) == 1 for k. k < 1 inflates longshots, k > 1 shrinks them."""
    def gap(k):
        return float(np.sum(raw_row ** k) - 1.0)

    # An overround book has sum > 1 at k = 1, and sum -> 0 as k grows, so the
    # root sits above 1. The bracket is widened rather than assumed.
    low, high = 1.0, 2.0
    for _ in range(20):
        if gap(high) < 0:
            break
        high *= 2
    else:
        return 1.0
    return brentq(gap, low, high, xtol=1e-12)


def _power(raw):
    return np.vstack([row ** _power_exponent(row) for row in np.atleast_2d(raw)]).reshape(raw.shape)


_NORMALISERS = {"multiplicative": _multiplicative, "additive": _additive, "power": _power}


def implied_probabilities(odds, method=DEFAULT_METHOD):
    """Decimal odds -> probabilities summing to 1, with the margin removed.

    `odds` is one (home, draw, away) triple or an array of them, in OUTCOMES
    order. Rows containing a NaN come back as NaN rather than being dropped,
    so the result always lines up with the frame it came from.
    """
    if method not in _NORMALISERS:
        raise ValueError(f"Unknown method {method!r}. Available: {sorted(_NORMALISERS)}")

    odds = np.asarray(odds, dtype=float)
    single = odds.ndim == 1
    matrix = np.atleast_2d(odds)

    if matrix.shape[-1] != len(OUTCOMES):
        raise ValueError(
            f"Expected {len(OUTCOMES)} odds per match in {OUTCOMES} order, got {matrix.shape[-1]}."
        )
    if np.any(matrix[np.isfinite(matrix)] <= 1.0):
        raise ValueError(
            "Decimal odds must be greater than 1.0 — a price of 1.0 returns the stake and "
            "implies certainty. Values at or below 1 usually mean the column is fractional "
            "or American odds rather than decimal."
        )

    out = np.full(matrix.shape, np.nan)
    usable = np.isfinite(matrix).all(axis=-1)
    if usable.any():
        out[usable] = _NORMALISERS[method](1.0 / matrix[usable])

    return out[0] if single else out


def market_probabilities(matches, method=DEFAULT_METHOD):
    """Attach `p_home` / `p_draw` / `p_away` to a match frame from its odds columns.

    The frame keeps its `odds_source` attribute, because a probability
    computed from opening prices and one computed from closing prices are not
    the same baseline and the distinction has to survive this step.
    """
    missing = [c for c in ODDS_COLUMNS if c not in matches.columns]
    if missing:
        raise ValueError(f"Match frame has no odds columns: {missing}")

    probabilities = implied_probabilities(matches[list(ODDS_COLUMNS)].to_numpy(), method=method)
    out = matches.copy()
    for column, values in zip(PROBABILITY_COLUMNS, probabilities.T):
        out[column] = values
    out.attrs.update(matches.attrs)
    out.attrs["probability_method"] = method
    return out


def compare_methods(odds, methods=METHODS):
    """The same prices under every normalisation, side by side.

    Worth running once on real data before trusting a result: if switching
    normalisation changes whether a model beats the market, the finding is
    about the margin model, not about the model.
    """
    rows = []
    for method in methods:
        probabilities = implied_probabilities(odds, method=method)
        row = {"method": method}
        row.update(dict(zip(PROBABILITY_COLUMNS, np.atleast_2d(probabilities)[0])))
        rows.append(row)
    table = pd.DataFrame(rows)
    table["overround"] = float(np.sum(1.0 / np.asarray(odds, dtype=float)) - 1.0)
    return table


def fair_odds(probabilities):
    """Probabilities -> the odds a zero-margin book would offer. The break-even price."""
    probabilities = np.asarray(probabilities, dtype=float)
    with np.errstate(divide="ignore"):
        return 1.0 / probabilities
