"""Chance-level reference points every model has to beat to be worth using.

Baloto draws 5 distinct main balls from a pool of 43 and 1 superbalota from a
pool of 16, all uniformly at random. If you commit to `m` distinct numbers as
your "prediction" for the main balls, the number of them that actually get
drawn follows a hypergeometric distribution — that's the chance baseline,
computed exactly (no simulation needed) and used by lottery/backtest.py to judge
whether Prophet/StatsForecast/XGBoost add any real signal.
"""

from functools import lru_cache

import numpy as np
from scipy.stats import hypergeom

from core.significance import EMPTY_RESULT, z_test_against_null
from lottery.models.common import MAIN_BALLS_DRAWN, MAIN_POOL, SUPER_POOL


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

    This is the lottery's half of the contract with core.significance: the
    pool resets every draw, so each historical draw is an independent
    hypergeometric trial, and this function's only job is to turn
    `m_guessed` into the per-draw mean and variance of that trial. The
    arithmetic, both p-values and the effect size come from core.

    `m_guessed` may be a single count (same every window) or a per-window
    list, since collisions between positions can make the distinct-guess
    count vary.

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
    if observed_hits.size == 0:
        return _as_chance_result(dict(EMPTY_RESULT))

    m_list = np.broadcast_to(np.asarray(m_guessed), observed_hits.shape)
    chance = [expected_main_matches(int(m), pool_size, n_drawn) for m in m_list.ravel()]
    means = np.array([c["mean"] for c in chance]).reshape(observed_hits.shape)
    variances = np.array([c["var"] for c in chance]).reshape(observed_hits.shape)

    return _as_chance_result(z_test_against_null(observed_hits, means, variances,
                                                 confidence=confidence))


def _as_chance_result(result):
    """`chance_mean` is this domain's name for core's `null_mean`.

    Kept as a rename rather than a second key because every lottery surface —
    the backtest summary, the strategy table, the dashboard — reads
    `chance_mean`, and two names for one number is how they drift apart.
    """
    result = dict(result)
    result["chance_mean"] = result.pop("null_mean")
    return result


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
