"""Scoring a finishing order — cycling/scoring.py.

The invariant this file exists for is the denominator. The Plackett-Luce
likelihood places riders one at a time against everyone not yet placed, and
"everyone" includes the riders who abandoned. That is how the domain's
keep-the-non-finishers rule stops being a check somebody has to remember and
becomes arithmetic: a forecast that backed a rider who climbed off is charged
for it, because they were available to win every position and took none.

`test_an_abandon_still_costs_the_forecast_that_backed_them` is that invariant.
Delete the non-finishers from the field and the score silently improves, which
is exactly the easier problem this package refuses to solve.
"""

import numpy as np
import pandas as pd
import pytest

from cycling.common import DNF, FINISHED, STAGE
from cycling.scoring import (
    METRICS,
    finish_order,
    kendall_tau,
    plackett_luce_log_score,
    race_score,
    spearman,
    top_n_accuracy,
    winner_brier_score,
    winner_log_score,
)


def _result(rows):
    """A tidy result frame: (rider, rank or None, status)."""
    return pd.DataFrame([
        {"ds": pd.Timestamp("2024-07-01"), "race": "r", "kind": STAGE, "stage": 1.0,
         "rank": rank, "rider": rider, "team": "t",
         "status": FINISHED if rank is not None else DNF,
         "time_seconds": 100.0 if rank is not None else np.nan}
        for rider, rank in rows
    ])


def test_the_log_score_is_the_textbook_likelihood():
    # Two riders, worths 3 and 1. The stronger winning costs log(3/4); the
    # second place is then forced, so it costs nothing.
    assert plackett_luce_log_score([3.0, 1.0], [0, 1]) == pytest.approx(-np.log(0.75) / 2)
    assert plackett_luce_log_score([3.0, 1.0], [1, 0]) == pytest.approx(-np.log(0.25) / 2)


def test_an_abandon_still_costs_the_forecast_that_backed_them():
    """The keep-the-non-finishers rule, as arithmetic rather than as a check.

    Three riders start and the forecast's favourite (worth 6) abandons. The two
    who finish are placed against a field that still contains him, so both
    placings are charged for the worth wasted on a rider who took nothing. Drop
    him from the field and the score improves — which is the easier problem
    ("predict the order among those who finished") this must never silently
    become.
    """
    with_abandon = plackett_luce_log_score([3.0, 1.0, 6.0], [0, 1])
    without_abandon = plackett_luce_log_score([3.0, 1.0], [0, 1])

    expected = -(np.log(3 / 10) + np.log(1 / 7)) / 2
    assert with_abandon == pytest.approx(expected)
    assert with_abandon > without_abandon


def test_the_scale_of_the_worths_changes_nothing():
    # Every probability is a ratio, so doubling every worth is the same model.
    worths = np.array([4.0, 2.0, 1.0, 0.5])
    assert plackett_luce_log_score(worths, [1, 0, 3]) == pytest.approx(
        plackett_luce_log_score(worths * 17.0, [1, 0, 3]))


def test_knowing_the_order_beats_not_knowing_it():
    result = _result([("a", 1), ("b", 2), ("c", 3), ("d", 4)])
    riders = ["a", "b", "c", "d"]
    informed = race_score([8.0, 4.0, 2.0, 1.0], riders, result)
    uninformed = race_score([1.0, 1.0, 1.0, 1.0], riders, result)
    backwards = race_score([1.0, 2.0, 4.0, 8.0], riders, result)
    assert informed < uninformed < backwards


@pytest.mark.parametrize("metric", METRICS)
def test_every_metric_prefers_the_right_answer(metric):
    result = _result([("a", 1), ("b", 2), ("c", 3), ("d", 4)])
    riders = ["a", "b", "c", "d"]
    assert race_score([8.0, 4.0, 2.0, 1.0], riders, result, metric=metric) < \
        race_score([1.0, 2.0, 4.0, 8.0], riders, result, metric=metric)


def test_an_unknown_metric_is_refused():
    result = _result([("a", 1), ("b", 2)])
    with pytest.raises(ValueError, match="Unknown metric"):
        race_score([1.0, 1.0], ["a", "b"], result, metric="spearman")


def test_a_finisher_with_no_forecast_is_refused_not_dropped():
    # Scoring against a field the forecaster never saw is not a score at all,
    # and dropping the rider quietly would make it look like one.
    result = _result([("a", 1), ("sorpresa", 2)])
    with pytest.raises(ValueError, match="no forecast"):
        finish_order(result, ["a", "b"])


def test_non_finishers_are_not_part_of_the_order():
    result = _result([("a", 1), ("b", None), ("c", 2)])
    assert finish_order(result, ["a", "b", "c"]) == [0, 2]


def test_winner_scores_pick_out_the_winner():
    probabilities = np.array([0.5, 0.3, 0.2])
    assert winner_log_score(probabilities, 0) == pytest.approx(-np.log(0.5))
    assert winner_log_score(probabilities, 0) < winner_log_score(probabilities, 2)
    assert winner_brier_score(probabilities, 0) == pytest.approx(0.25 + 0.09 + 0.04)


def test_a_zero_probability_winner_is_capped_not_infinite():
    # One race scoring infinity would take the whole average with it.
    assert np.isfinite(winner_log_score(np.array([0.0, 1.0]), 0))


# ------------------------------------------------------------------ diagnostics

def test_the_rank_correlations_are_readable_and_decide_nothing():
    result = _result([("a", 1), ("b", 2), ("c", 3), ("d", 4)])
    perfect = ["a", "b", "c", "d"]
    assert spearman(perfect, result) == pytest.approx(1.0)
    assert kendall_tau(perfect, result) == pytest.approx(1.0)
    assert spearman(list(reversed(perfect)), result) == pytest.approx(-1.0)


def test_top_n_accuracy_counts_the_overlap():
    result = _result([("a", 1), ("b", 2), ("c", 3), ("d", 4)])
    assert top_n_accuracy(["a", "b", "c", "d"], result, n=2) == pytest.approx(1.0)
    assert top_n_accuracy(["c", "d", "a", "b"], result, n=2) == pytest.approx(0.0)
    assert top_n_accuracy(["a", "d", "b", "c"], result, n=2) == pytest.approx(0.5)
