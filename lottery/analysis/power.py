"""How large an edge could this dataset actually detect?

Every test in this project answers "did I find an edge?". None of them answers
the question that has to come first: **could I have found one if it were
there?** Those are different, and conflating them is how a null result gets
over-read.

With the chance mean at 5x5/43 = 0.58 matches and a standard deviation of 0.68,
a few hundred draws can only reveal a fairly large edge. A backtest over 15
windows that reports "no model beats chance" has established almost nothing —
it could not have detected a 50% edge. The same sentence over 1,000 draws is a
real finding. This module makes that distinction computable, so a negative
result can be stated with its own resolution attached:

    "No edge detected, and this run could only have detected one above +23%."

The arithmetic mirrors `lottery/models/baseline.py:beats_chance_test` exactly, since the
point is to characterise *that* test:

    z = (sum(hits) - N*mu) / sqrt(N*var) = sqrt(N) * (observed_mean - mu) / sd

so for a true mean of mu + delta the one-sided test at level alpha has

    power = 1 - Phi(z_alpha - sqrt(N) * delta / sd)

**One approximation, stated rather than buried.** The alternative is assumed to
have the same variance as the null. A model with a real edge has no
first-principles variance — it depends on how the edge arises — and the null
variance is the only defensible default. For the small edges that matter here
the difference is negligible; for a huge edge the required-N figures are
slightly pessimistic, which is the safe direction to be wrong in.
"""

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from lottery.models.baseline import expected_main_matches, expected_super_match_rate
from lottery.models.common import DRAW_WEEKDAYS, MAIN_BALLS_DRAWN, MAIN_POOL, SUPER_POOL

DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80

# The relative edges worth tabulating: from "impossible to miss" down to
# "no amount of lottery history will ever show this".
DEFAULT_RELATIVE_EDGES = (0.50, 0.25, 0.10, 0.05, 0.02)


def chance_moments(m_guessed=MAIN_BALLS_DRAWN, pool_size=MAIN_POOL, n_drawn=MAIN_BALLS_DRAWN):
    """(mean, sd) of main-ball matches per draw under pure chance."""
    moments = expected_main_matches(m_guessed, pool_size, n_drawn)
    return moments["mean"], math.sqrt(moments["var"])


def minimum_detectable_effect(n_draws, m_guessed=MAIN_BALLS_DRAWN, alpha=DEFAULT_ALPHA,
                              power=DEFAULT_POWER):
    """The smallest edge this many draws could reliably detect.

    Returns the effect in three forms because each answers a different
    question: `absolute` is the extra matches per draw, `relative` is that as a
    fraction of the chance mean (the form worth quoting), and
    `detectable_mean` is the observed average a model would need to hit.
    """
    if n_draws < 1:
        raise ValueError(f"n_draws must be at least 1, got {n_draws}")

    mean, sd = chance_moments(m_guessed)
    delta = (norm.ppf(1 - alpha) + norm.ppf(power)) * sd / math.sqrt(n_draws)
    return {
        "n_draws": int(n_draws),
        "chance_mean": mean,
        "sd": sd,
        "absolute": delta,
        "relative": delta / mean,
        "detectable_mean": mean + delta,
        "alpha": alpha,
        "power": power,
    }


def required_draws(relative_edge, m_guessed=MAIN_BALLS_DRAWN, alpha=DEFAULT_ALPHA,
                   power=DEFAULT_POWER):
    """How many draws it would take to detect an edge of this relative size."""
    if relative_edge <= 0:
        raise ValueError(f"relative_edge must be positive, got {relative_edge}")

    mean, sd = chance_moments(m_guessed)
    delta = mean * relative_edge
    return math.ceil((((norm.ppf(1 - alpha) + norm.ppf(power)) * sd) / delta) ** 2)


def achieved_power(n_draws, relative_edge, m_guessed=MAIN_BALLS_DRAWN, alpha=DEFAULT_ALPHA):
    """Probability this many draws would flag an edge of this size, if it were real."""
    mean, sd = chance_moments(m_guessed)
    delta = mean * relative_edge
    return float(1 - norm.cdf(norm.ppf(1 - alpha) - math.sqrt(n_draws) * delta / sd))


def draws_to_years(n_draws, weekdays=DRAW_WEEKDAYS):
    """Calendar years of history a draw count represents, on the real schedule."""
    return n_draws / (len(weekdays) * 52.0)


def required_draws_table(relative_edges=DEFAULT_RELATIVE_EDGES, m_guessed=MAIN_BALLS_DRAWN,
                         alpha=DEFAULT_ALPHA, power=DEFAULT_POWER, weekdays=DRAW_WEEKDAYS):
    """One row per candidate edge: how much history it would take to see it.

    The `years_of_history` column is what makes the table land — several rows
    exceed the age of the game, which is the honest answer to "why has nobody
    proven a lottery system works?".
    """
    mean, _ = chance_moments(m_guessed)
    rows = []
    for edge in relative_edges:
        n = required_draws(edge, m_guessed=m_guessed, alpha=alpha, power=power)
        rows.append({
            "relative_edge": edge,
            "target_mean": mean * (1 + edge),
            "required_draws": n,
            "years_of_history": draws_to_years(n, weekdays),
        })
    return pd.DataFrame(rows)


def power_curve(n_draws, relative_edges=None, m_guessed=MAIN_BALLS_DRAWN, alpha=DEFAULT_ALPHA):
    """Power against edge size for a fixed amount of data — the curve to plot."""
    if relative_edges is None:
        relative_edges = np.linspace(0.01, 1.0, 100)

    mean, _ = chance_moments(m_guessed)
    return pd.DataFrame({
        "relative_edge": relative_edges,
        "target_mean": [mean * (1 + e) for e in relative_edges],
        "power": [achieved_power(n_draws, e, m_guessed=m_guessed, alpha=alpha)
                  for e in relative_edges],
    })


def super_minimum_detectable_effect(n_draws, pool_size=SUPER_POOL, alpha=DEFAULT_ALPHA,
                                    power=DEFAULT_POWER):
    """Same question for the superbalota, which is a Bernoulli trial, not hypergeometric.

    Guessing one number out of `pool_size` hits with probability 1/pool_size,
    so the per-draw variance is p(1-p) rather than a hypergeometric variance.
    Kept separate rather than folded into the main helper because using the
    wrong variance here would understate the required data by roughly a factor
    of three, and the two are easy to confuse.
    """
    if n_draws < 1:
        raise ValueError(f"n_draws must be at least 1, got {n_draws}")

    p = expected_super_match_rate(pool_size)
    sd = math.sqrt(p * (1 - p))
    delta = (norm.ppf(1 - alpha) + norm.ppf(power)) * sd / math.sqrt(n_draws)
    return {
        "n_draws": int(n_draws),
        "chance_rate": p,
        "sd": sd,
        "absolute": delta,
        "relative": delta / p,
        "detectable_rate": p + delta,
        "alpha": alpha,
        "power": power,
    }


def describe(n_draws, m_guessed=MAIN_BALLS_DRAWN, alpha=DEFAULT_ALPHA, power=DEFAULT_POWER):
    """A one-line English summary of what a run of this size can and cannot show."""
    mde = minimum_detectable_effect(n_draws, m_guessed=m_guessed, alpha=alpha, power=power)
    return (
        f"With {mde['n_draws']} draws, a one-sided test at alpha={alpha:g} has {power:.0%} power "
        f"only against edges of +{mde['relative']:.0%} or more (an average of "
        f"{mde['detectable_mean']:.3f} matches against the chance level of {mde['chance_mean']:.3f}). "
        f"A smaller real edge would most likely go unnoticed, so 'no edge detected' here means "
        f"'no edge above +{mde['relative']:.0%}', not 'no edge'."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-draws", type=int, default=1000,
                        help="how much data you have (or plan to have)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--power", type=float, default=DEFAULT_POWER)
    args = parser.parse_args()

    pd.set_option("display.width", 120)
    print(describe(args.n_draws, alpha=args.alpha, power=args.power))

    print("\n=== How much history each edge size would need ===")
    table = required_draws_table(alpha=args.alpha, power=args.power)
    print(table.to_string(index=False, formatters={
        "relative_edge": "{:.0%}".format, "target_mean": "{:.4f}".format,
        "required_draws": "{:,}".format, "years_of_history": "{:.1f}".format,
    }))

    print(f"\n=== Power of {args.n_draws} draws against each edge ===")
    curve = power_curve(args.n_draws, relative_edges=DEFAULT_RELATIVE_EDGES, alpha=args.alpha)
    print(curve.to_string(index=False, formatters={
        "relative_edge": "{:.0%}".format, "target_mean": "{:.4f}".format, "power": "{:.1%}".format,
    }))

    super_mde = super_minimum_detectable_effect(args.n_draws, alpha=args.alpha, power=args.power)
    print(f"\nSuperbalota: chance rate {super_mde['chance_rate']:.4f}, smallest detectable rate "
          f"{super_mde['detectable_rate']:.4f} (+{super_mde['relative']:.0%}).")
