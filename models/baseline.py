"""Chance-level reference points every model has to beat to be worth using.

Baloto draws 5 distinct main balls from a pool of 43 and 1 superbalota from a
pool of 16, all uniformly at random. If you commit to `m` distinct numbers as
your "prediction" for the main balls, the number of them that actually get
drawn follows a hypergeometric distribution — that's the chance baseline,
computed exactly (no simulation needed) and used by backtest.py to judge
whether Prophet/StatsForecast/XGBoost add any real signal.
"""

from functools import lru_cache

import numpy as np
from scipy.stats import hypergeom, norm

from models.common import MAIN_BALLS_DRAWN, MAIN_POOL, SUPER_POOL


@lru_cache(maxsize=None)
def expected_main_matches(m_guessed, pool_size=MAIN_POOL, n_drawn=MAIN_BALLS_DRAWN):
    """Mean and variance of matches when guessing m_guessed distinct numbers.

    Cached because the backtest asks for the same handful of m values
    (0..5) once per window, and building a frozen scipy distribution is
    the dominant cost of summarizing a run.
    """
    dist = hypergeom(pool_size, m_guessed, n_drawn)
    return {"mean": float(dist.mean()), "var": float(dist.var())}


def expected_super_match_rate(pool_size=SUPER_POOL, m_guessed=1):
    return m_guessed / pool_size


def beats_chance_test(observed_hits, m_guessed, pool_size=MAIN_POOL, n_drawn=MAIN_BALLS_DRAWN,
                      confidence=0.95):
    """z-test: is this model's average number of matches distinguishable from pure chance?

    Each historical draw is an independent hypergeometric trial (the pool
    resets every draw), so the sum of `observed_hits` is asymptotically normal
    around the chance mean under the null "the model has no real signal".
    `m_guessed` may be a single count (same every window) or a per-window list,
    since collisions between positions can make the distinct-guess count vary.

    Two p-values come back, and they answer different questions. `p_value` is
    two-sided ("does this differ from chance at all?"); `p_value_greater` is
    one-sided ("is this *better* than chance?"). Only the one-sided one may be
    used to claim a model beats chance — a model significantly worse than
    chance also gets a small two-sided p-value.

    `effect` and its confidence interval come back too, because a p-value alone
    hides precision: "no edge detected" over 15 windows and over 1,000 draws
    read identically as p-values, while their intervals differ by an order of
    magnitude. The interval is the honest summary of what was measured.

    **The exact hypergeometric variance is used even when several tickets are
    scored against the same draw**, and that was checked rather than assumed.
    Sharing a draw does induce positive correlation in principle, so this test
    was measured under the null at 1 and 5 tickets per draw: sd(z) came out at
    0.997 and 0.927 over 60 runs, against the 1.0 a calibrated statistic gives.
    No inflation — if anything slightly conservative at 5. A cluster-robust
    variance was written for this and then removed, because it corrected a
    distortion that is not there. See docs/evaluation.md for how the phantom
    came to be believed in the first place.
    """
    observed_hits = np.asarray(observed_hits, dtype=float)
    empty = {"z": np.nan, "p_value": np.nan, "p_value_greater": np.nan,
             "observed_mean": np.nan, "chance_mean": np.nan, "effect": np.nan,
             "ci_low": np.nan, "ci_high": np.nan, "relative_effect": np.nan,
             "n_observations": 0}
    if len(observed_hits) == 0:
        return empty

    m_list = np.broadcast_to(np.asarray(m_guessed), observed_hits.shape)
    chance = [expected_main_matches(int(m), pool_size, n_drawn) for m in m_list]
    means = np.array([c["mean"] for c in chance])
    variances = np.array([c["var"] for c in chance])

    observed_mean = float(observed_hits.mean())
    chance_mean = float(means.mean())
    se_sum = float(np.sqrt(variances.sum()))

    base = {"observed_mean": observed_mean, "chance_mean": chance_mean,
            "n_observations": len(observed_hits)}
    if se_sum == 0:
        return {**empty, **base}

    z = float((observed_hits.sum() - means.sum()) / se_sum)
    effect = observed_mean - chance_mean
    margin = float(norm.ppf(0.5 + confidence / 2) * se_sum / len(observed_hits))
    return {
        "z": z,
        "p_value": float(2 * (1 - norm.cdf(abs(z)))),
        "p_value_greater": float(1 - norm.cdf(z)),
        "effect": effect,
        "ci_low": effect - margin,
        "ci_high": effect + margin,
        "relative_effect": effect / chance_mean if chance_mean else np.nan,
        **base,
    }


def most_frequent_pick(position_series, upto=None):
    """The 'play the hottest number in each slot' strategy, as a comparison point.

    No predictive edge for i.i.d. draws — it is here to be beaten, and to give
    the backtest and the dashboard one shared definition of the baseline.
    `upto` truncates the history, for walk-forward use.
    """
    return {
        pos: int((frame.iloc[:upto] if upto is not None else frame)["y"].mode().iloc[0])
        for pos, frame in position_series.items()
    }
