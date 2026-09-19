"""Order-agnostic summaries of a ticket, against their exact reference distributions.

Every function here reduces the 5 main balls of a draw to one number that does
not depend on the order they are stored in — their sum, how many are odd, how
many fall below a threshold — and compares the observed distribution against the
exact combinatorial one.

**Why this exists next to `randomness.py`.** The pooled uniformity test asks
whether each number shows up equally often. That is a question about *marginals*,
and a machine can pass it while still being broken: if it favoured drawing balls
that sit close together, every number would still appear equally often and the
pooled test would see nothing, while the sums would pile up in a narrower band
than combinatorics allows. These summaries are sensitive to exactly that kind of
dependence *between* the balls, which makes them a second, independent verdict
rather than a restatement of the first.

**They are also immune to the sorted-data trap**, and for a stronger reason than
the pooled test is. Sorting a draw cannot change its sum, its odd count or how
many of its balls are below 31 — those are properties of the *set*. So no amount
of order-statistic structure in the source data can move these numbers at all.

**And they are the honest home for the "all combinations are equally likely"
lesson.** They are, and the sum distribution is the clearest way to see why that
does not make every *summary* equally likely: there is exactly one way to draw a
sum of 15 (1-2-3-4-5) and 14,090 ways to draw a sum of 110. That is a fact about
how many sets share a sum, not about any set being favoured — which is also why
picking an unusual sum cannot improve `P(win)`, only who you split with.
"""

from math import comb

import numpy as np
import pandas as pd
from scipy import stats

from lottery.models.common import (
    MAIN_BALLS_DRAWN,
    MAIN_POOL,
    main_positions,
    range_for_position,
)

# Numbers 1-31 fit on a calendar, so they are massively over-played relative to
# 32-43. Nothing about them is more likely; the split is here because it is the
# single largest driver of how many people you would split a jackpot with.
CALENDAR_THRESHOLD = 31


def _pool_bounds(balls_expanded, positions):
    """The one ball range shared by `positions`, or a refusal.

    Mirrors `pooled_uniformity_test`: pooling columns that span different ranges
    is not expressible rather than silently wrong, since 1-16 is a subset of
    1-43 and no value range check would catch it.
    """
    n_columns = balls_expanded.shape[1]
    ranges = {range_for_position(p, n_columns) for p in positions}
    if len(ranges) != 1:
        raise ValueError(
            f"Positions {list(positions)} span different ball ranges {sorted(ranges)} and cannot "
            "be summarised together."
        )
    return ranges.pop()


def _selected(balls_expanded, positions):
    """The chosen columns as an integer array, one row per draw."""
    return balls_expanded.iloc[:, list(positions)].to_numpy(dtype=np.int64)


def sum_counts(pool=MAIN_POOL, drawn=MAIN_BALLS_DRAWN, low=1):
    """How many distinct `drawn`-number sets from `low..pool` add up to each total.

    Exact integer counts by dynamic programming, not a simulation. The whole
    table sums to C(pool - low + 1, drawn), which `structure_report` relies on
    to turn the counts into probabilities.
    """
    numbers = range(low, pool + 1)
    max_sum = sum(sorted(numbers)[-drawn:])
    # counts[k][s] = ways to pick k of the numbers seen so far adding to s.
    counts = [[0] * (max_sum + 1) for _ in range(drawn + 1)]
    counts[0][0] = 1
    for number in numbers:
        # Descending in k and s so each number is used at most once per set.
        for k in range(drawn, 0, -1):
            for total in range(max_sum, number - 1, -1):
                if counts[k - 1][total - number]:
                    counts[k][total] += counts[k - 1][total - number]
    return counts[drawn]


def sum_distribution(balls_expanded, positions=None):
    """Observed vs exact distribution of the sum of the main balls.

    Returns one row per attainable sum with its exact probability, the count
    observed in this history, and the count you would expect from that
    probability.
    """
    positions = list(main_positions(balls_expanded.shape[1]) if positions is None else positions)
    low, high = _pool_bounds(balls_expanded, positions)
    drawn = len(positions)

    counts = sum_counts(pool=high, drawn=drawn, low=low)
    total_sets = comb(high - low + 1, drawn)
    attainable = [s for s, c in enumerate(counts) if c]

    observed = pd.Series(_selected(balls_expanded, positions).sum(axis=1))
    observed_counts = observed.value_counts().reindex(attainable, fill_value=0)
    n_draws = len(observed)

    probability = np.array([counts[s] / total_sets for s in attainable], dtype=float)
    return pd.DataFrame({
        "sum": attainable,
        "n_combinations": [counts[s] for s in attainable],
        "probability": probability,
        "observed": observed_counts.to_numpy(),
        "expected": probability * n_draws,
    })


def _hypergeometric_split(balls_expanded, positions, member):
    """Exact distribution of how many drawn balls satisfy `member(number)`.

    Drawing without replacement from a pool split in two is hypergeometric, so
    this needs no simulation either.
    """
    positions = list(positions)
    low, high = _pool_bounds(balls_expanded, positions)
    drawn = len(positions)

    pool = np.arange(low, high + 1)
    n_matching = int(sum(member(int(n)) for n in pool))
    n_pool = len(pool)

    values = _selected(balls_expanded, positions)
    observed = pd.Series(np.array([[member(int(v)) for v in row] for row in values]).sum(axis=1))
    observed_counts = observed.value_counts().reindex(range(drawn + 1), fill_value=0)
    n_draws = len(observed)

    probability = np.array([
        comb(n_matching, k) * comb(n_pool - n_matching, drawn - k) / comb(n_pool, drawn)
        if 0 <= drawn - k <= n_pool - n_matching and k <= n_matching else 0.0
        for k in range(drawn + 1)
    ], dtype=float)

    return pd.DataFrame({
        "k": range(drawn + 1),
        "probability": probability,
        "observed": observed_counts.to_numpy(),
        "expected": probability * n_draws,
    }), n_matching, n_pool


def parity_distribution(balls_expanded, positions=None):
    """Observed vs exact distribution of how many of the drawn balls are odd."""
    positions = main_positions(balls_expanded.shape[1]) if positions is None else positions
    table, _, _ = _hypergeometric_split(balls_expanded, positions, lambda n: n % 2 == 1)
    return table.rename(columns={"k": "n_odd"})


def range_split_distribution(balls_expanded, positions=None, threshold=CALENDAR_THRESHOLD):
    """Observed vs exact distribution of how many drawn balls are <= `threshold`.

    At the default 31 this is the calendar split, the one that drives how many
    people share a jackpot. At the pool midpoint it is the classic low/high
    split. Same arithmetic either way, so it is one function with a parameter.
    """
    positions = main_positions(balls_expanded.shape[1]) if positions is None else positions
    table, n_matching, n_pool = _hypergeometric_split(
        balls_expanded, positions, lambda n: n <= threshold)
    table = table.rename(columns={"k": "n_at_or_below"})
    table.attrs["threshold"] = threshold
    table.attrs["pool_share"] = n_matching / n_pool
    return table


def goodness_of_fit(table, min_expected=5.0):
    """Chi-square of `observed` against `expected`, pooling the thin tails first.

    A chi-square needs a handful of observations per cell to mean anything, and
    the sum distribution has 191 cells of which most are nearly empty on any
    real history. Cells below `min_expected` are merged with their neighbours
    from the outside in, which keeps the test valid without throwing draws away.

    Returns NaN rather than a number when fewer than two cells survive: with one
    cell there is nothing left to test, and reporting a p-value there would be
    inventing a verdict out of an empty comparison.
    """
    observed = np.asarray(table["observed"], dtype=float)
    expected = np.asarray(table["expected"], dtype=float)

    kept_observed, kept_expected = [], []
    carry_observed = carry_expected = 0.0
    for obs, exp in zip(observed, expected, strict=True):
        carry_observed += obs
        carry_expected += exp
        if carry_expected >= min_expected:
            kept_observed.append(carry_observed)
            kept_expected.append(carry_expected)
            carry_observed = carry_expected = 0.0
    if carry_expected > 0:
        if kept_expected:
            # A leftover tail joins the last full bin rather than standing as a
            # thin cell of its own, which is the whole point of the pooling.
            kept_observed[-1] += carry_observed
            kept_expected[-1] += carry_expected
        else:
            kept_observed.append(carry_observed)
            kept_expected.append(carry_expected)

    n_bins = len(kept_expected)
    if n_bins < 2:
        return {"chi2": float("nan"), "p_value": float("nan"), "n_bins": n_bins,
                "n_observations": float(observed.sum())}

    # chisquare requires both sides to total the same; float error in `expected`
    # is enough to trip its check, so rescale rather than let it raise.
    kept_expected = np.array(kept_expected) * (sum(kept_observed) / sum(kept_expected))
    stat, p_value = stats.chisquare(kept_observed, kept_expected)
    return {"chi2": float(stat), "p_value": float(p_value), "n_bins": n_bins,
            "n_observations": float(observed.sum())}


def structure_report(balls_expanded, positions=None):
    """All three summaries with their verdicts, for the dashboard's one call.

    `looks_random` is the conjunction, and it carries the same caveat as every
    other multi-test surface here: three tests at alpha = 0.05 flag a clean
    history about 14% of the time, so a single "No" is not a finding.
    """
    positions = list(main_positions(balls_expanded.shape[1]) if positions is None else positions)
    sums = sum_distribution(balls_expanded, positions)
    parity = parity_distribution(balls_expanded, positions)
    calendar = range_split_distribution(balls_expanded, positions)

    tests = {
        "sum": goodness_of_fit(sums),
        "parity": goodness_of_fit(parity),
        "calendar": goodness_of_fit(calendar),
    }
    p_values = [t["p_value"] for t in tests.values()]
    return {
        "sum_distribution": sums,
        "parity_distribution": parity,
        "calendar_distribution": calendar,
        "tests": tests,
        "observed_mean_sum": float(_selected(balls_expanded, positions).sum(axis=1).mean()),
        "expected_mean_sum": float((sums["sum"] * sums["probability"]).sum()),
        "looks_random": all(np.isnan(p) or p > 0.05 for p in p_values),
        "n_draws": int(len(balls_expanded)),
    }


def sum_percentile(main_numbers, pool=MAIN_POOL, drawn=MAIN_BALLS_DRAWN, low=1):
    """What fraction of all possible tickets have a sum at or below this one.

    Near 0.5 means a thoroughly ordinary sum, which is where most people play.
    Near 0 or 1 means an unusual one — no less likely to win, but shared with
    fewer people if it does.
    """
    counts = sum_counts(pool=pool, drawn=drawn, low=low)
    total_sets = comb(pool - low + 1, drawn)
    target = int(sum(main_numbers))
    return float(sum(counts[: target + 1]) / total_sets)
