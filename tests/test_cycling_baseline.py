"""The pre-race ranking as a distribution — cycling/baseline.py.

Two things matter more than the arithmetic.

**`as_of` has to actually hold.** `form_worths` builds a ranking out of earlier
results, and a ranking that has seen the race it is ranking for is not a
baseline, it is an answer key. `test_form_worths_cannot_see_the_race_it_ranks`
is the test that would notice a `<` becoming a `<=`.

**A uniform draw is not a baseline**, and it is implemented here so the mistake
has a name. The test below pins what it actually is: 1/n for everyone, which any
forecast beats by knowing a single rider.
"""

import numpy as np
import pandas as pd
import pytest

from cycling.baseline import (
    baseline_frame,
    form_worths,
    predicted_order,
    sample_orders,
    top_n_probabilities,
    uniform_worths,
    win_probabilities,
    worths_from_points,
    worths_from_rating,
)
from cycling.sample_data import load_sample_and_preprocess


@pytest.fixture(scope="module")
def results():
    return load_sample_and_preprocess(seed=0)


def test_win_probabilities_are_shares_of_the_total_worth():
    probabilities = win_probabilities([3.0, 1.0, 1.0])
    assert probabilities.sum() == pytest.approx(1.0)
    assert probabilities[0] == pytest.approx(0.6)


def test_a_uniform_draw_is_exactly_one_over_n():
    # Named, implemented, and documented as not a baseline — 180 riders at 0.55%
    # each is beaten by knowing one name.
    probabilities = win_probabilities(uniform_worths(180))
    assert probabilities == pytest.approx(np.full(180, 1 / 180))


def test_worths_must_be_positive_and_finite():
    for bad in ([1.0, 0.0, 2.0], [1.0, -1.0], [1.0, np.nan]):
        with pytest.raises(ValueError, match="positive"):
            win_probabilities(bad)


def test_points_pass_through_with_a_floor():
    # Points are already roughly proportional to how often a rider wins, which
    # is what a Luce worth is; transforming them would make the baseline a model.
    worths = worths_from_points([500.0, 120.0, 0.0])
    assert worths[0] == 500.0 and worths[1] == 120.0
    assert worths[2] > 0  # a zero-point rider is unlikely, not impossible


def test_a_rating_scale_only_changes_confidence_not_order():
    ratings = np.array([2.0, 0.0, -1.0])
    tight = win_probabilities(worths_from_rating(ratings, scale=0.5))
    loose = win_probabilities(worths_from_rating(ratings, scale=4.0))
    assert list(np.argsort(-tight)) == list(np.argsort(-loose))
    assert tight[0] > loose[0]  # a smaller scale is a more confident forecast


def test_sampling_reproduces_the_win_probabilities():
    """The Gumbel-max shortcut has to be the real Plackett-Luce distribution.

    Sampling by adding a Gumbel to each log-worth and sorting is only valid if
    the winner's frequency matches its share of the total worth. That is the
    identity the whole sampling path depends on.
    """
    worths = np.array([6.0, 3.0, 1.0])
    orders = sample_orders(worths, n_samples=40_000, seed=0)
    observed = np.bincount(orders[:, 0], minlength=3) / len(orders)
    assert observed == pytest.approx(win_probabilities(worths), abs=0.01)


def test_top_n_probabilities_are_bounded_and_sum_to_n():
    worths = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    probabilities = top_n_probabilities(worths, n=2, n_samples=5000, seed=1)
    assert (probabilities >= 0).all() and (probabilities <= 1).all()
    # Exactly two riders fill the top two of every sampled order.
    assert probabilities.sum() == pytest.approx(2.0)
    assert probabilities[0] > probabilities[-1]


def test_predicted_order_is_the_ranking_itself():
    riders = ["a", "b", "c"]
    assert predicted_order([1.0, 9.0, 4.0], riders) == ["b", "c", "a"]


def test_form_worths_cannot_see_the_race_it_ranks(results):
    """The leakage test, and the reason `as_of` is not optional in practice.

    A rider who wins a stage must not get credit for it in the ranking built
    *for* that stage. Only results strictly earlier count, so the worths built
    as of a date cannot move when that date's results change.
    """
    stage_five = results[results["stage"] == 5]
    as_of = stage_five["ds"].min()
    riders = list(stage_five["rider"])

    before = form_worths(results, riders, as_of=as_of)

    # Rewrite stage 5 entirely — reverse every placing — and rebuild.
    tampered = results.copy()
    mask = tampered["stage"] == 5
    tampered.loc[mask, "rank"] = tampered.loc[mask, "rank"].max() + 1 - tampered.loc[mask, "rank"]
    after = form_worths(tampered, riders, as_of=as_of)

    assert before == pytest.approx(after)


def test_form_worths_reward_finishing_near_the_front(results):
    as_of = results["ds"].max()
    riders = sorted(results["rider"].unique())
    worths = form_worths(results, riders, as_of=as_of)

    ranked = results[results["rank"].notna()]
    mean_rank = ranked.groupby("rider")["rank"].mean().reindex(riders)
    both = pd.DataFrame({"worth": worths, "mean_rank": mean_rank.to_numpy()}).dropna()
    # Better average placing, higher worth.
    assert both["worth"].corr(both["mean_rank"]) < -0.5


def test_baseline_frame_is_sorted_and_complete(results):
    riders = sorted(results["rider"].unique())[:20]
    worths = form_worths(results, riders, as_of=results["ds"].max())
    table = baseline_frame(riders, worths, n=5, n_samples=500, seed=0)

    assert len(table) == len(riders)
    assert list(table["predicted_rank"]) == list(range(1, len(riders) + 1))
    assert list(table["worth"]) == sorted(table["worth"], reverse=True)
    assert table["p_win"].sum() == pytest.approx(1.0)
