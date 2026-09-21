"""What edge could this many matches have revealed, if there had been one?

`evaluation.py` answers "did the model beat the closing line". This module
answers the question that has to travel with a No: **could it have said Yes?**
Without it, "the model did not beat the market" and "this backtest could not
have detected it if it had" are the same sentence, and the difference between
them is the whole difference between a measurement and a shrug.

This is `lottery/analysis/power.py` pointed at football, and it mirrors
`beats_market_test` exactly the way that module mirrors `beats_chance_test`.
Same alpha, same one-sided test, same arithmetic — because the point is to
characterise *that* test rather than a nearby one.

**One thing genuinely differs, and it is the interesting part.** On the lottery
side the null's variance is exact: the hypergeometric distribution hands it
over in closed form, so the minimum detectable effect is a function of `n` and
nothing else. Here the quantity being tested is a **paired difference of two
proper scores** — the market's RPS minus the model's, per match — and nothing
gives its variance in advance. It depends on the league, the season, how sharp
the book is, and how far the model strays from it.

So the variance has to be **measured, not assumed**, and every function here
takes it as an argument. `observed_score_sd(model_probs, market_probs, outcomes)`
computes it from a backtest that has already run, and
`REFERENCE_SCORE_SD` is a fallback for the "before you start" case, carrying a
loud caveat about where it came from. A minimum detectable effect quoted from
an assumed variance is a guess with a decimal point on it.

**The headline number is brutal and is meant to be.** A season of one league is
~380 matches. At the measured spread against a sharp book that puts the
smallest reliably detectable RPS improvement at about **0.010** — which is
several times what a good model actually takes out of a closing line. Detecting
a realistic 0.002 would take roughly 10,000 matches, twenty-six seasons of one
league. That is not a defect in the test; it is why `clv.py` exists, and why
`docs/football.md` says a model beating the closing line is nearly unanswerable
in a season.
"""

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from football.scoring import per_match_scores

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80

# Candidate RPS improvements, in the units the scores are actually in. A good
# football model takes something in the low thousandths out of a closing line;
# 0.02 is included as the "obviously would have been seen" end of the range,
# not as an aspiration.
DEFAULT_EDGES = (0.02, 0.01, 0.005, 0.002, 0.001)

# A stand-in for the per-match sd of (market RPS - model RPS), for the case
# where nothing has been run yet.
#
# **Measured on this project's synthetic seasons**, Dixon-Coles and Elo over 80
# held-out matches against books of four sharpnesses:
#
#     market_noise   dixon_coles   elo
#     0.0 (sharp)         0.072    0.093
#     0.5                 0.129    0.137
#     1.0                 0.205    0.205
#     1.5 (soft)          0.249    0.241
#
# 0.08 is the sharp end, which is the realistic one: a closing line is the
# sharp case by definition, and that is the book this domain measures against.
#
# The spread is **not** a property of the test — it rises with how far the
# model strays from the price, so a soft book more than triples it. Required
# match counts scale with its square, so using the wrong end of that table is a
# factor-of-ten error. Replace this with `observed_score_sd` on a real run at
# the first opportunity; it is offered only so `describe()` can say something
# before any backtest exists.
#
# An earlier version of this constant was 0.05, written from intuition before
# the measurement, and it was wrong by a factor of two at the sharp end and
# five at the soft one. That is the whole argument for the module's rule that
# the variance is measured rather than assumed.
REFERENCE_SCORE_SD = 0.08


def observed_score_sd(model_probs, market_probs, outcomes, metric="rps"):
    """The per-match sd of `market_score - model_score`, from a run that happened.

    This is the number every other function here needs, and the only honest
    source of it. Matches where either side is NaN are dropped, the same rule
    `beats_market_test` follows, so a partially-priced frame does not quietly
    inflate or deflate the spread.
    """
    model = np.asarray(model_probs, dtype=float).reshape(-1, 3)
    market = np.asarray(market_probs, dtype=float).reshape(-1, 3)
    outcomes = np.asarray(list(outcomes))

    keep = ~(np.isnan(model).any(axis=1) | np.isnan(market).any(axis=1))
    model, market, outcomes = model[keep], market[keep], outcomes[keep]
    if outcomes.size < 2:
        return float("nan")

    difference = (per_match_scores(market, outcomes, metric)
                  - per_match_scores(model, outcomes, metric))
    return float(np.std(difference, ddof=1))


def minimum_detectable_edge(n_matches, score_sd=REFERENCE_SCORE_SD, alpha=DEFAULT_ALPHA,
                            power=DEFAULT_POWER):
    """The smallest RPS improvement this many matches could reliably detect.

    `absolute` is in the units of the score itself — the amount by which the
    model's mean RPS would have to undercut the market's. That is the form to
    quote, because unlike the lottery there is no natural denominator to make
    it a percentage of: the market's own RPS is not a "chance level", it is a
    genuinely good forecast, and a ratio against it reads as more impressive
    than it is.
    """
    if n_matches < 1:
        raise ValueError(f"n_matches must be at least 1, got {n_matches}")
    if not np.isfinite(score_sd) or score_sd <= 0:
        raise ValueError(f"score_sd must be positive and finite, got {score_sd!r}")

    delta = (norm.ppf(1 - alpha) + norm.ppf(power)) * score_sd / math.sqrt(n_matches)
    return {
        "n_matches": int(n_matches),
        "score_sd": float(score_sd),
        "absolute": float(delta),
        "alpha": alpha,
        "power": power,
    }


def required_matches(edge, score_sd=REFERENCE_SCORE_SD, alpha=DEFAULT_ALPHA,
                     power=DEFAULT_POWER):
    """How many matches it would take to detect an RPS improvement of this size."""
    if edge <= 0:
        raise ValueError(f"edge must be positive, got {edge}")
    if not np.isfinite(score_sd) or score_sd <= 0:
        raise ValueError(f"score_sd must be positive and finite, got {score_sd!r}")

    return math.ceil((((norm.ppf(1 - alpha) + norm.ppf(power)) * score_sd) / edge) ** 2)


def achieved_power(n_matches, edge, score_sd=REFERENCE_SCORE_SD, alpha=DEFAULT_ALPHA):
    """Probability this many matches would flag an edge of this size, if it were real."""
    if not np.isfinite(score_sd) or score_sd <= 0:
        raise ValueError(f"score_sd must be positive and finite, got {score_sd!r}")
    return float(1 - norm.cdf(norm.ppf(1 - alpha) - math.sqrt(n_matches) * edge / score_sd))


def matches_to_seasons(n_matches, matches_per_season=380):
    """Seasons of one league a match count represents.

    380 is a 20-team double round robin — the Premier League, and close enough
    to the other big five. The point of the column is the same as
    `draws_to_years` on the lottery side: a number of matches means nothing
    until it is a number of seasons nobody is going to wait for.
    """
    return n_matches / float(matches_per_season)


def required_matches_table(edges=DEFAULT_EDGES, score_sd=REFERENCE_SCORE_SD,
                           alpha=DEFAULT_ALPHA, power=DEFAULT_POWER,
                           matches_per_season=380):
    """One row per candidate edge: how much football it would take to see it."""
    rows = []
    for edge in edges:
        n = required_matches(edge, score_sd=score_sd, alpha=alpha, power=power)
        rows.append({
            "edge": edge,
            "required_matches": n,
            "seasons_of_one_league": matches_to_seasons(n, matches_per_season),
            "score_sd": float(score_sd),
        })
    return pd.DataFrame(rows)


def power_curve(n_matches, edges=DEFAULT_EDGES, score_sd=REFERENCE_SCORE_SD,
                alpha=DEFAULT_ALPHA):
    """For a fixed number of matches, the chance of catching each candidate edge."""
    return pd.DataFrame([
        {"edge": edge,
         "power": achieved_power(n_matches, edge, score_sd=score_sd, alpha=alpha),
         "n_matches": int(n_matches)}
        for edge in edges
    ])


def describe(n_matches, score_sd=REFERENCE_SCORE_SD, alpha=DEFAULT_ALPHA,
             power=DEFAULT_POWER):
    """The sentence a null result needs attached to it, ready to print.

    Names the standard deviation it used, because the whole number turns on it
    and a reader who does not know whether it was measured or assumed cannot
    tell a resolution from a guess.
    """
    mde = minimum_detectable_edge(n_matches, score_sd=score_sd, alpha=alpha, power=power)
    source = ("the reference spread" if score_sd == REFERENCE_SCORE_SD
              else "the spread measured on this run")
    return (
        f"With {n_matches} matches, the smallest RPS improvement this test could find "
        f"{power:.0%} of the time is {mde['absolute']:.4f} — using {source} "
        f"(sd = {score_sd:.4f}). Below that, 'did not beat the market' is a statement "
        f"about the sample size."
    )
