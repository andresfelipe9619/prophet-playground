"""The order-agnostic summaries and their exact reference distributions.

Two properties carry this module and are what these tests pin:

**The references are exact, not simulated.** `sum_counts` is a combinatorial
count, so it has arithmetic identities a simulation would only approximate — it
totals C(43,5) on the nose and is symmetric about the mean. If a refactor ever
turns it into sampling, those identities break immediately.

**The summaries are sort-proof.** Sorting a draw cannot change its sum, its odd
count or how many of its balls sit below 31, because all three are properties of
the set. That is a stronger guarantee than the pooled uniformity test has, and
`test_every_summary_is_blind_to_column_order` is the test that actually holds
the module to it.
"""

from math import comb

import numpy as np
import pandas as pd
import pytest

from lottery.analysis.structure import (
    CALENDAR_THRESHOLD,
    goodness_of_fit,
    parity_distribution,
    range_split_distribution,
    structure_report,
    sum_counts,
    sum_distribution,
    sum_percentile,
)
from lottery.models.common import MAIN_BALLS_DRAWN, MAIN_POOL, main_positions


def test_sum_counts_totals_every_possible_ticket():
    # The identity that proves this is a count and not an estimate.
    assert sum(sum_counts()) == comb(MAIN_POOL, MAIN_BALLS_DRAWN)


def test_sum_counts_are_symmetric_about_the_mean():
    counts = sum_counts()
    lowest, highest = 1 + 2 + 3 + 4 + 5, 39 + 40 + 41 + 42 + 43
    assert counts[lowest] == counts[highest] == 1  # 1-2-3-4-5 and 39-...-43, one way each
    for offset in range(0, 40):
        assert counts[lowest + offset] == counts[highest - offset]


def test_the_extremes_are_as_likely_as_any_other_single_ticket():
    # The lesson the sum chart exists to teach: one sum being rarer than another
    # is a fact about how many sets share it, never about a set being favoured.
    counts = sum_counts()
    assert counts[15] == 1
    assert counts[110] == 14090
    assert counts[110] / sum(counts) > counts[15] / sum(counts)


@pytest.mark.parametrize("builder", [sum_distribution, parity_distribution, range_split_distribution])
def test_reference_probabilities_are_a_distribution(sample, builder):
    table = builder(sample[1])
    assert table["probability"].sum() == pytest.approx(1.0)
    assert (table["probability"] >= 0).all()


def test_expected_mean_sum_is_the_combinatorial_one(sample):
    # 5 balls drawn from 1..43 average 5 x 22 = 110, exactly.
    report = structure_report(sample[1])
    assert report["expected_mean_sum"] == pytest.approx(110.0)


def test_observed_counts_account_for_every_draw(sample):
    df, balls_expanded = sample
    for table in (sum_distribution(balls_expanded), parity_distribution(balls_expanded),
                  range_split_distribution(balls_expanded)):
        assert table["observed"].sum() == len(df)
        assert table["expected"].sum() == pytest.approx(len(df))


def test_every_summary_is_blind_to_column_order(sample):
    """The property that makes this module immune to the sorted-data trap.

    Not "mostly unaffected" — identical. Shuffling the main columns within each
    row is exactly what an order statistic does to a per-position test, and it
    must do nothing at all here.
    """
    _, balls_expanded = sample
    n_columns = balls_expanded.shape[1]
    positions = list(main_positions(n_columns))

    rng = np.random.default_rng(0)
    shuffled = balls_expanded.copy()
    values = shuffled.iloc[:, positions].to_numpy()
    shuffled.iloc[:, positions] = np.array([rng.permutation(row) for row in values])

    for builder in (sum_distribution, parity_distribution, range_split_distribution):
        pd.testing.assert_frame_equal(builder(balls_expanded), builder(shuffled))


def test_pooling_positions_of_different_ranges_is_refused(sample):
    _, balls_expanded = sample
    n_columns = balls_expanded.shape[1]
    # The superbalota is 1-16; folding it in with the 1-43 main balls is the
    # mistake no value range check could catch, since 1-16 is a subset of 1-43.
    with pytest.raises(ValueError, match="different ball ranges"):
        sum_distribution(balls_expanded, positions=list(range(n_columns)))


def test_goodness_of_fit_pools_the_thin_tails():
    # 191 sum cells over a few hundred draws leaves most of them nearly empty,
    # and a chi-square over those cells is not a test of anything.
    table = pd.DataFrame({"observed": [0, 0, 1, 50, 1, 0, 0],
                          "expected": [0.1, 0.2, 1.0, 49.4, 1.0, 0.2, 0.1]})
    result = goodness_of_fit(table, min_expected=5.0)
    assert result["n_bins"] < len(table)
    assert result["n_observations"] == 52


def test_goodness_of_fit_refuses_a_verdict_it_cannot_support():
    # One surviving bin means there is nothing left to compare; a p-value there
    # would be invented rather than measured.
    table = pd.DataFrame({"observed": [3], "expected": [3.0]})
    result = goodness_of_fit(table)
    assert np.isnan(result["p_value"])


def test_a_clean_history_passes_all_three(sample):
    report = structure_report(sample[1])
    assert report["looks_random"]
    assert report["observed_mean_sum"] == pytest.approx(110.0, abs=6.0)


def test_a_planted_dependence_the_pooled_test_cannot_see_is_caught():
    """The reason this module is not a restatement of `pooled_uniformity_test`.

    Each draw is five consecutive numbers wrapping around the pool, so over a
    full cycle of starts every number appears exactly five times: the marginals
    are perfectly flat and a pooled uniformity test has nothing to object to.
    What these draws do not do is pick the five independently, so their sums
    land on 43 values out of the 191 combinatorics allows. Only a summary that
    looks at the balls *together* can see it.
    """
    starts = [(i % MAIN_POOL) + 1 for i in range(10 * MAIN_POOL)]
    balls_expanded = pd.DataFrame(
        [[((start - 1 + offset) % MAIN_POOL) + 1 for offset in range(MAIN_BALLS_DRAWN)]
         + [(start % 16) + 1]
         for start in starts]
    )

    counts = pd.Series(
        balls_expanded.iloc[:, list(main_positions(balls_expanded.shape[1]))].to_numpy().ravel()
    ).value_counts()
    assert counts.nunique() == 1  # every number drawn exactly as often as every other

    assert goodness_of_fit(sum_distribution(balls_expanded))["p_value"] < 0.01


def test_calendar_split_defaults_to_the_numbers_that_fit_a_date(sample):
    table = range_split_distribution(sample[1])
    assert table.attrs["threshold"] == CALENDAR_THRESHOLD
    assert table.attrs["pool_share"] == pytest.approx(31 / 43)


def test_sum_percentile_is_bounded_and_monotone():
    assert sum_percentile([1, 2, 3, 4, 5]) == pytest.approx(1 / comb(43, 5))
    assert sum_percentile([39, 40, 41, 42, 43]) == pytest.approx(1.0)
    assert sum_percentile([1, 2, 3, 4, 6]) > sum_percentile([1, 2, 3, 4, 5])
    # An utterly ordinary sum lands in the middle, which is where most people play.
    assert 0.4 < sum_percentile([5, 15, 25, 30, 35]) < 0.7
