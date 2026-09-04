"""Chance-level reference points every model has to beat to be worth using.

Baloto draws 5 distinct main balls from a pool of 43 and 1 superbalota from a
pool of 16, all uniformly at random. If you commit to `m` distinct numbers as
your "prediction" for the main balls, the number of them that actually get
drawn follows a hypergeometric distribution — that's the chance baseline,
computed exactly (no simulation needed) and used by backtest.py to judge
whether Prophet/StatsForecast/XGBoost add any real signal.
"""

import numpy as np
from scipy.stats import hypergeom, norm

MAIN_POOL = 43
MAIN_DRAWN = 5
SUPER_POOL = 16


def expected_main_matches(m_guessed, pool_size=MAIN_POOL, n_drawn=MAIN_DRAWN):
    """Mean and variance of matches when guessing m_guessed distinct numbers."""
    dist = hypergeom(pool_size, m_guessed, n_drawn)
    return {"mean": float(dist.mean()), "var": float(dist.var())}


def expected_super_match_rate(pool_size=SUPER_POOL, m_guessed=1):
    return m_guessed / pool_size


def beats_chance_test(observed_hits, m_guessed, pool_size=MAIN_POOL, n_drawn=MAIN_DRAWN):
    """z-test: is this model's average number of matches distinguishable from pure chance?

    Each historical draw is an independent hypergeometric trial (the pool
    resets every draw), so the sum of `observed_hits` is asymptotically normal
    around the chance mean under the null "the model has no real signal".
    `m_guessed` may be a single count (same every window) or a per-window list,
    since collisions between positions can make the distinct-guess count vary.
    """
    observed_hits = np.asarray(observed_hits, dtype=float)
    n = len(observed_hits)
    if n == 0:
        return {"z": np.nan, "p_value": np.nan, "observed_mean": np.nan, "chance_mean": np.nan}

    m_list = np.broadcast_to(np.asarray(m_guessed), observed_hits.shape)
    means = np.array([expected_main_matches(int(m), pool_size, n_drawn)["mean"] for m in m_list])
    variances = np.array([expected_main_matches(int(m), pool_size, n_drawn)["var"] for m in m_list])

    observed_mean = observed_hits.mean()
    chance_mean = means.mean()
    se_sum = np.sqrt(variances.sum())
    if se_sum == 0:
        return {"z": np.nan, "p_value": np.nan, "observed_mean": observed_mean, "chance_mean": chance_mean}

    z = (observed_hits.sum() - means.sum()) / se_sum
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return {"z": float(z), "p_value": float(p_value), "observed_mean": float(observed_mean),
            "chance_mean": float(chance_mean)}


def empirical_frequency_pick(freq_table, k, number_col="number", count_col="count"):
    """The 'pick the historically hottest numbers' strategy. No predictive edge for i.i.d.
    draws, but it's a common heuristic and a useful comparison point in the dashboard."""
    top = freq_table.sort_values(count_col, ascending=False).head(k)
    return sorted(top[number_col].tolist())


def uniform_random_pick(low, high, k, rng=None):
    rng = rng or np.random.default_rng()
    return sorted(rng.choice(np.arange(low, high + 1), size=k, replace=False).tolist())
