"""What running the bets actually feels like, which an edge does not tell you.

`value.py` answers "should I take this bet and how much". It stops there, and
what it leaves out is the part that decides whether a system is runnable by a
person: **the path**. A 3% edge with a 40% chance of a 50% drawdown is not a
system, it is a way to lose your nerve at the worst possible moment, and an
expected value has no room to say so.

So this module simulates the sequence rather than summarising it. Given a set
of bets — each a probability, a price and an outcome — it plays them in order
many times over and reports the distribution of what happened: terminal
bankroll, worst drawdown, how often the bankroll fell past the point of no
return.

**Three things about it are load-bearing.**

**The null path is drawn beside every simulation.** A bankroll chart is the
most persuasive object this project can produce and the easiest to mislead
with: an upward curve reads as a promise no matter how much text sits under it.
`simulate_bankroll` therefore always returns a companion showing what the same
betting behaviour does **when the model's edge is not real** — the same
selections, the same stakes, the same prices, but with the outcomes redrawn
from the de-margined market's probabilities. The gap between the two curves is
the information, and it is the only part of the picture that means anything.
This is the football twin of presenting a lottery hindcast as a prediction, and
the null path is what stops it.

The null is built that way after the obvious construction turned out to be
useless. Replacing the *model's probability* with the market's and re-staking
produces a bettor who **never bets at all**: `value.py`'s Kelly filter compares
a de-margined probability against the raw price it must beat, and a bettor
holding only the market's own numbers never clears the margin. That is the two
bars doing their job, and it is a true and completely uninformative flat line.
Keeping the bets and redrawing the world is the comparison that answers the
question the chart actually raises.

**Under that null the median bankroll falls**, at roughly the rate of the
margin, because the bets still get placed and still pay the overround. A
simulation that shows a flat line there has forgotten the vig, which is the
single most common way these charts lie.

**Kelly is a ceiling, not a target**, and the measurement is more violent than
the argument. Over 237 synthetic matches against a soft book, handing the model
the **literal generative truth** — the best any forecast could possibly be:

    stake      median ROI   median drawdown   risk of ruin
    0.10           +302%               46%             0%
    0.25 (default) +580%               82%             9%
    0.50            -44%               99%            62%
    1.00 (full)    -100%              100%            99%

Full Kelly stakes up to **88% of the bankroll on a single match** here, against
22% at a quarter. A model that is exactly right is ruined by betting its own
advice at full size, because being right about a probability is not the same as
surviving its variance. That is what `value.py`'s quarter default is buying,
and on an edge that is *not* real the full stake sizes up precisely when the
model is most confidently wrong.

Note what the first column does *not* say. Those returns come from a forecast
that knows the truth exactly, against a book blurred on purpose. No real model
is in that position, and the numbers are here to show the shape of the
trade-off between stake and survival, not as anything anyone should expect.
"""

import numpy as np
import pandas as pd

from football.common import OUTCOMES, outcome_index
from football.value import DEFAULT_KELLY_FRACTION, kelly_fraction

# Below this share of the starting bankroll, a run is called ruined. Not zero:
# fractional staking never reaches zero exactly, so a threshold is the only way
# the question has an answer at all. 10% is the point past which nobody is
# still running the system, whatever the arithmetic says.
RUIN_THRESHOLD = 0.10

DEFAULT_PATHS = 500


def _stakes(probabilities, odds, fraction):
    """The share of bankroll to put on each bet, from the model's own numbers."""
    return np.asarray(kelly_fraction(probabilities, odds, fraction=fraction), dtype=float)


def _returns(stakes, odds, won):
    """Multiplicative bankroll factor per bet: 1 + stake * (odds - 1) if won, else 1 - stake."""
    stakes = np.nan_to_num(np.asarray(stakes, dtype=float), nan=0.0)
    return np.where(won, 1.0 + stakes * (np.asarray(odds, dtype=float) - 1.0), 1.0 - stakes)


def _won(outcomes, bet_on):
    return np.asarray([o == b for o, b in zip(outcomes, bet_on, strict=True)], dtype=bool)


def simulate_bankroll(model_probs, market_probs, odds, outcomes, bet_on=None,
                      fraction=DEFAULT_KELLY_FRACTION, n_paths=DEFAULT_PATHS,
                      seed=0, bootstrap=True):
    """Play the bets many times over and report where the bankroll went.

    `model_probs` and `market_probs` are `(n, 3)` in `OUTCOMES` order; `odds`
    is the **raw** price triple, because a bet pays the margin — the two bars
    `value.py` insists on, and this is the bet-side one. `bet_on` names the
    outcome backed on each match, defaulting to the model's favourite.

    `bootstrap=True` resamples the bet order for each path, which is what makes
    a distribution out of one season: the same bets in a different order give a
    different worst drawdown, and the order that happened is one draw from that.
    With `bootstrap=False` every path is the realised sequence and the
    distribution collapses to a point — useful only for reproducing one history.

    Returns a dict with the model's paths and, always, `null_paths`: the same
    selections at the same stakes, with outcomes redrawn from the de-margined
    market — what this betting behaviour does when the edge is not real. A
    bankroll chart without that companion is a promise rather than a
    measurement, and callers are not trusted to remember to ask for it.
    """
    model_probs = np.asarray(model_probs, dtype=float).reshape(-1, 3)
    market_probs = np.asarray(market_probs, dtype=float).reshape(-1, 3)
    odds = np.asarray(odds, dtype=float).reshape(-1, 3)
    outcomes = list(outcomes)

    if bet_on is None:
        bet_on = [OUTCOMES[i] for i in model_probs.argmax(axis=1)]
    picked = np.array([outcome_index(b) for b in bet_on])
    rows = np.arange(len(picked))

    won = _won(outcomes, bet_on)
    chosen_odds = odds[rows, picked]

    model_stakes = _stakes(model_probs[rows, picked], chosen_odds, fraction)
    model_returns = _returns(model_stakes, chosen_odds, won)

    rng = np.random.default_rng(seed)
    n = len(model_returns)
    order = (rng.integers(0, n, size=(int(n_paths), n)) if bootstrap
             else np.tile(np.arange(n), (int(n_paths), 1)))

    # The null: identical selections and stakes, with the world behaving as the
    # de-margined price says rather than as it did. Drawn per path, because the
    # question is what this behaviour does across worlds where the model knows
    # nothing -- one such world would be an anecdote.
    null_hit = market_probs[rows, picked]
    null_won = rng.random((int(n_paths), n)) < null_hit[order]
    null_returns = _returns(np.broadcast_to(model_stakes[order], null_won.shape),
                            chosen_odds[order], null_won)

    return {
        "paths": np.cumprod(model_returns[order], axis=1),
        "null_paths": np.cumprod(null_returns, axis=1),
        "n_bets": n,
        "fraction": float(fraction),
        "bootstrap": bool(bootstrap),
        "mean_stake": float(np.nanmean(model_stakes)),
        "n_staked": int((np.nan_to_num(model_stakes) > 0).sum()),
    }


def drawdown_distribution(paths, quantiles=(0.5, 0.9, 0.95, 0.99)):
    """How deep the worst trough of each path got, as a distribution.

    A drawdown is measured against the running peak, not the start: the number
    that matters is how much was given back from the best it ever looked, which
    is the moment a person actually stops.
    """
    paths = np.atleast_2d(np.asarray(paths, dtype=float))
    peaks = np.maximum.accumulate(paths, axis=1)
    worst = (1.0 - paths / peaks).max(axis=1)
    return {
        "mean": float(worst.mean()),
        "quantiles": {q: float(np.quantile(worst, q)) for q in quantiles},
        "worst": float(worst.max()),
    }


def risk_of_ruin(paths, threshold=RUIN_THRESHOLD):
    """Share of paths that ever fell below `threshold` of the starting bankroll.

    "Ever", not "ended": a system that dips to 8% and recovers has already lost
    the person running it.
    """
    paths = np.atleast_2d(np.asarray(paths, dtype=float))
    return float((paths.min(axis=1) < threshold).mean())


def roi_interval(paths, confidence=0.95):
    """Terminal return and its interval across paths, as a fraction of the stake bank."""
    terminal = np.atleast_2d(np.asarray(paths, dtype=float))[:, -1] - 1.0
    tail = (1.0 - confidence) / 2.0
    return {
        "median_roi": float(np.median(terminal)),
        "mean_roi": float(terminal.mean()),
        "ci_low": float(np.quantile(terminal, tail)),
        "ci_high": float(np.quantile(terminal, 1.0 - tail)),
        "share_losing": float((terminal < 0).mean()),
    }


def summarise(simulation, threshold=RUIN_THRESHOLD, confidence=0.95):
    """Model and null side by side, which is the only way either should be read."""
    rows = []
    for label, paths in (("model", simulation["paths"]), ("null (market)", simulation["null_paths"])):
        roi = roi_interval(paths, confidence=confidence)
        drawdown = drawdown_distribution(paths)
        rows.append({
            "series": label,
            "n_bets": simulation["n_bets"],
            "stake_fraction": simulation["fraction"],
            "median_roi": roi["median_roi"],
            "roi_ci_low": roi["ci_low"],
            "roi_ci_high": roi["ci_high"],
            "share_losing": roi["share_losing"],
            "median_drawdown": drawdown["quantiles"][0.5],
            "worst_drawdown_95": drawdown["quantiles"][0.95],
            "risk_of_ruin": risk_of_ruin(paths, threshold=threshold),
        })
    return pd.DataFrame(rows)


def stake_fraction_sweep(model_probs, market_probs, odds, outcomes, bet_on=None,
                         fractions=(0.1, 0.25, 0.5, 1.0), n_paths=DEFAULT_PATHS, seed=0):
    """The same bets at several Kelly fractions — the picture of why a quarter.

    Full Kelly maximises long-run growth **if the probability is right**. It is
    a ceiling derived under an assumption a model cannot meet, and what the
    sweep shows is the price of pretending otherwise: the drawdown distribution
    grows far faster than the return.
    """
    rows = []
    for fraction in fractions:
        simulation = simulate_bankroll(model_probs, market_probs, odds, outcomes,
                                       bet_on=bet_on, fraction=fraction,
                                       n_paths=n_paths, seed=seed)
        roi = roi_interval(simulation["paths"])
        drawdown = drawdown_distribution(simulation["paths"])
        rows.append({
            "stake_fraction": fraction,
            "median_roi": roi["median_roi"],
            "median_drawdown": drawdown["quantiles"][0.5],
            "worst_drawdown_95": drawdown["quantiles"][0.95],
            "risk_of_ruin": risk_of_ruin(simulation["paths"]),
            "share_losing": roi["share_losing"],
        })
    return pd.DataFrame(rows)
