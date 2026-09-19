"""Scoring a forecast of a finishing order. Lower is better, everywhere.

Cycling's `football/scoring.py`, and it has the same shape: several rules, one
of which is the verdict and the rest of which are diagnostics.

**The Plackett-Luce log score is the verdict.** It is the only rule here that is
proper over the actual target — a whole ordering — rather than over some summary
of it. Rank correlations reward getting the middle of the bunch roughly right,
which is the easy and worthless part; top-N accuracy throws away everything
about how wrong a miss was. Both are here because both are readable, and neither
decides anything. This mirrors "the pooled test is the verdict" on the lottery
side and "RPS is the verdict" on the football side.

**Non-finishers stay in the denominator, and that is structural.** The
Plackett-Luce likelihood places riders one at a time, and at each step the
denominator is the total worth of everyone *not yet placed* — which includes
every rider who abandoned. A forecast that put its money on a rider who did not
finish is charged for it, exactly as it should be: they were at risk of winning
and did not. Dropping abandons would silently renormalise the field to the
riders who made it, turning "predict the finishing order" into the far easier
"predict the order among those who finished", which is the thing this domain's
contract already refuses to do to the data. Here the refusal is not a check to
remember — it falls out of the arithmetic.
"""

import numpy as np
from scipy import stats

from cycling.common import FINISHED

METRICS = ("plackett_luce", "winner_log", "winner_brier")
DEFAULT_METRIC = "plackett_luce"

# A probability floor for the winner log score. A baseline that gave the actual
# winner exactly zero would score infinity and take the whole race average with
# it; this caps the penalty at ~16 nats, which is already a rout.
_PROBABILITY_FLOOR = 1e-7


def _as_worths(values):
    worths = np.asarray(values, dtype=float)
    if np.any(worths <= 0) or not np.all(np.isfinite(worths)):
        raise ValueError("Worths must all be finite and strictly positive.")
    return worths


def finish_order(result, riders):
    """Indices into `riders` of the ranked finishers, best placed first.

    Rows without a rank or without a finished status are left out of the
    *order*, not out of the field — they remain in the scoring denominator,
    which is the point.
    """
    index = {rider: i for i, rider in enumerate(riders)}
    ranked = result[(result["status"] == FINISHED) & result["rank"].notna()]
    ranked = ranked.sort_values("rank")

    unknown = sorted(set(ranked["rider"]) - index.keys())
    if unknown:
        raise ValueError(
            f"{len(unknown)} rider(s) finished but carry no forecast, e.g. {unknown[:3]}. "
            "A forecast has to cover the whole start list, or the score is over a field "
            "the forecaster never saw."
        )
    return [index[rider] for rider in ranked["rider"]]


def plackett_luce_log_score(worths, order):
    """Mean negative log-likelihood per placed rider. Lower is better.

    Places riders one at a time: the contribution of the rider taking position
    k is their share of the worth still unplaced. Everyone who never finishes
    stays in that denominator for the whole race.
    """
    worths = _as_worths(worths)
    order = list(order)
    if not order:
        return float("nan")

    remaining = float(worths.sum())
    total = 0.0
    for index in order:
        total += float(np.log(worths[index]) - np.log(remaining))
        remaining -= float(worths[index])
        if remaining <= 0:
            break  # the last rider placed: nothing left to choose from
    return -total / len(order)


def winner_log_score(probabilities, winner_index):
    """`-log p(winner)`. The purest "how surprised were you" measure, and a harsh one."""
    probabilities = np.asarray(probabilities, dtype=float)
    return float(-np.log(max(float(probabilities[winner_index]), _PROBABILITY_FLOOR)))


def winner_brier_score(probabilities, winner_index):
    """Squared error against the one-hot winner, summed over the start list.

    Gentler than the log score and bounded, but over ~180 riders it is dominated
    by the many near-zero probabilities, so small differences in it say more
    about the tail than about who was picked.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    truth = np.zeros_like(probabilities)
    truth[winner_index] = 1.0
    return float(((probabilities - truth) ** 2).sum())


def race_score(worths, riders, result, metric=DEFAULT_METRIC):
    """One number for one race. Lower is better, whichever metric."""
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}. Expected one of {list(METRICS)}.")
    order = finish_order(result, riders)
    if not order:
        return float("nan")

    if metric == "plackett_luce":
        return plackett_luce_log_score(worths, order)

    from cycling.baseline import win_probabilities

    probabilities = win_probabilities(worths)
    if metric == "winner_log":
        return winner_log_score(probabilities, order[0])
    return winner_brier_score(probabilities, order[0])


# ------------------------------------------------------------------ diagnostics
#
# Readable, and decide nothing. Each reduces the ordering to a summary, and the
# summary is what makes them easy to read and unfit to rule on.

def spearman(predicted_order, result):
    """Rank correlation between the forecast order and the finish, over finishers.

    +1 is a perfect ordering, 0 is no relationship. Reads well and rewards
    getting the middle of the bunch roughly right, which is the part nobody
    cares about — a forecast can score well here and have missed every podium.
    """
    ranks = _paired_ranks(predicted_order, result)
    if len(ranks[0]) < 3:
        return float("nan")
    return float(stats.spearmanr(ranks[0], ranks[1]).statistic)


def kendall_tau(predicted_order, result):
    """Share of rider pairs put in the right relative order, rescaled to -1..+1."""
    ranks = _paired_ranks(predicted_order, result)
    if len(ranks[0]) < 3:
        return float("nan")
    return float(stats.kendalltau(ranks[0], ranks[1]).statistic)


def top_n_accuracy(predicted_order, result, n=10):
    """How many of the forecast's top `n` actually finished in the top `n`."""
    ranked = result[(result["status"] == FINISHED) & result["rank"].notna()]
    actual = set(ranked.sort_values("rank")["rider"].head(int(n)))
    predicted = set(list(predicted_order)[:int(n)])
    if not actual:
        return float("nan")
    return len(actual & predicted) / float(len(actual))


def _paired_ranks(predicted_order, result):
    """(predicted rank, actual rank) for every rider who has both."""
    predicted_rank = {rider: i + 1 for i, rider in enumerate(predicted_order)}
    ranked = result[(result["status"] == FINISHED) & result["rank"].notna()]
    pairs = [(predicted_rank[row.rider], float(row.rank))
             for row in ranked.itertuples() if row.rider in predicted_rank]
    if not pairs:
        return np.array([]), np.array([])
    predicted, actual = zip(*pairs, strict=True)
    return np.array(predicted, dtype=float), np.array(actual, dtype=float)
