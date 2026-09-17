"""Turning a probability disagreement into a stake — and refusing to pretend it is one.

This is football's counterpart to the lottery's jackpot splitting: the module
that answers "so what do I actually do", and the one most able to do harm if
read carelessly. Jackpot splitting is safe because it improves a quantity that
is real whether or not any model works. Nothing here is safe in that way. Every
number below is conditional on the model being right, and the only evidence
about that lives in `football/backtest.py`.

**The two bars, which are different, and confusing them is the whole trap.**

- To judge whether a *model* is good, compare it against the **de-margined**
  market price. That is what `market.py` produces and what `evaluation.py`
  tests, because the bookmaker's margin is not a forecast and a model measured
  against `1/odds` is measured against a baseline deliberately tilted against it.
- To judge whether a *bet* is worth placing, compare it against the **raw**
  `1/odds`. You pay the margin. A model can be genuinely better than the market
  and still have no bet at this price, because the edge it found is smaller than
  the commission.

Those two facts produce the three states `classify` returns, and the middle one
is where most betting systems live: the model disagrees with the market, and the
disagreement is not big enough to pay for the spread.

**Kelly is not a safety feature.** It is the stake that maximises long-run
growth *given a true edge*. Applied to an edge that is not real it does not
merely fail to help — it sizes up precisely when the model is most confidently
wrong, which is how a staking plan turns a small negative expectation into a
fast one. `kelly_fraction` therefore defaults to a quarter stake, and the
dashboard will not show a stake at all until the backtest verdict is on screen
next to it.

**Scope.** One outcome at a time. Backing two outcomes of the same match
simultaneously is a different optimisation (the bets are mutually exclusive, so
the single-bet formula over-stakes), and pretending otherwise would be the same
kind of quiet wrongness this package exists to avoid.
"""

import numpy as np
import pandas as pd

from football.common import N_OUTCOMES, OUTCOMES, outcome_label
from football.market import implied_probabilities

# A quarter of the Kelly stake. Full Kelly is correct only if the probability is
# correct, and a football model's is an estimate with a standard error; the
# growth cost of a quarter stake is small and the drawdown relief is large.
DEFAULT_KELLY_FRACTION = 0.25

NO_VALUE = "no_value"
DISAGREEMENT_ONLY = "disagreement_only"
VALUE = "value"


def break_even_probability(odds):
    """The probability you need for a bet at these odds to be a coin flip: 1/odds.

    Includes the bookmaker's margin, deliberately — this is the betting bar, not
    the modelling one.
    """
    odds = np.asarray(odds, dtype=float)
    return np.where(odds > 0, 1.0 / odds, np.nan)


def expected_value(probabilities, odds):
    """Expected profit per unit staked: `p * odds - 1`.

    Zero means break even, +0.05 means five cents back on every peso staked in
    the long run — if `probabilities` is right.
    """
    return np.asarray(probabilities, dtype=float) * np.asarray(odds, dtype=float) - 1.0


def kelly_fraction(probabilities, odds, fraction=DEFAULT_KELLY_FRACTION):
    """Share of the bankroll to stake, scaled by `fraction`; never negative.

    The full Kelly stake is `(p * odds - 1) / (odds - 1)`. A negative result
    means the bet is bad, not that you should lay it — this returns 0 there,
    because "bet nothing" is the actionable form of that answer.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    odds = np.asarray(odds, dtype=float)
    net_odds = odds - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        full = (probabilities * odds - 1.0) / net_odds
    full = np.where(net_odds > 0, full, np.nan)
    return np.clip(full, 0.0, 1.0) * float(fraction)


def classify(model_probabilities, market_probabilities, odds):
    """Which of the three states each outcome is in.

    `no_value` — the model is no more optimistic than the de-margined market.
    `disagreement_only` — the model is more optimistic than the market but not
    enough to clear `1/odds`, so the disagreement does not pay for the margin.
    `value` — positive expected value at this price, conditional on the model.
    """
    model = np.asarray(model_probabilities, dtype=float)
    market = np.asarray(market_probabilities, dtype=float)
    threshold = break_even_probability(odds)
    return np.where(model > threshold, VALUE,
                    np.where(model > market, DISAGREEMENT_ONLY, NO_VALUE))


def value_table(model_probabilities, odds, method="multiplicative",
                fraction=DEFAULT_KELLY_FRACTION, bankroll=None):
    """One row per outcome, with both bars shown side by side.

    `model_probabilities` and `odds` are length-3 in OUTCOMES order. The market
    column is de-margined with `method`; the break-even column is the raw
    `1/odds` you actually have to beat to profit.
    """
    model = np.asarray(model_probabilities, dtype=float).reshape(N_OUTCOMES)
    odds = np.asarray(odds, dtype=float).reshape(N_OUTCOMES)
    market = implied_probabilities(odds, method=method)

    stakes = kelly_fraction(model, odds, fraction=fraction)
    table = pd.DataFrame({
        "outcome": list(OUTCOMES),
        "label": [outcome_label(o) for o in OUTCOMES],
        "odds": odds,
        "model_probability": model,
        "market_probability": market,
        "break_even_probability": break_even_probability(odds),
        "expected_value": expected_value(model, odds),
        "kelly_stake": stakes,
        "verdict": classify(model, market, odds),
    })
    if bankroll is not None:
        table["stake"] = table["kelly_stake"] * float(bankroll)
    return table


def margin_cost(odds, method="multiplicative"):
    """How much probability the margin eats, per outcome.

    The gap between what you must beat (`1/odds`) and what the market actually
    thinks (de-margined). It is the width of the `disagreement_only` band, and
    seeing it as a number is what stops "my model says 38% and the market says
    35%" reading as a bet.
    """
    odds = np.asarray(odds, dtype=float).reshape(N_OUTCOMES)
    return break_even_probability(odds) - implied_probabilities(odds, method=method)
