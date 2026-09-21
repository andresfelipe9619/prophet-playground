"""Outright prices as the hard baseline — cycling/market.py.

Three things here are not football's tests wearing a jersey.

**The field is the unit.** Football normalises a triple; this normalises ~180
mutually exclusive runners, so a *partial* field is the characteristic mistake
and it is refused rather than normalised, which would hand the missing runners'
probability to the ones that are present.

**`additive` is pinned as unusable, not as broken.** On a realistic outright
book it drives most of the field negative, and the module clips and
renormalises. The test does not assert that it is wrong — it asserts that
`n_zeroed` still says so after the renormalisation has made the vector look
like a distribution again.

**A worth is not a probability.** `worths_from_market` inverts the Plackett-Luce
win probability, which is exact only up to the scale the model does not
identify, so what is pinned is the ratio between two riders and the mean-1
normalisation — never a single worth's magnitude.
"""

import numpy as np
import pytest

from cycling.market import (
    DEFAULT_METHOD,
    METHODS,
    MIN_DECIMAL_ODDS,
    MarketFormatError,
    compare_methods,
    implied_probabilities,
    market_frame,
    overround,
    worths_from_market,
)


def book(n_runners=180, margin=1.40, seed=0):
    """A synthetic outright book: a plausible worth curve priced with a margin.

    The margin is applied multiplicatively to the true probabilities, which is
    the *easiest* book to de-margin — a real one takes more out of the tail.
    That makes the tests conservative: a method that cannot recover this one
    has no chance against a real favourite-longshot shape.
    """
    rng = np.random.default_rng(seed)
    worths = np.exp(rng.normal(0.0, 1.6, n_runners))
    probabilities = worths / worths.sum()
    return 1.0 / (probabilities * margin)


def test_an_outright_book_overrounds_far_more_than_a_football_one():
    # Football's 2-8% is not the scale here: 180 priced runners carry a margin
    # each, and a method tuned for the small excess has a big job. Reported as
    # the excess, the same convention `football/market.py` uses — two functions
    # of one name meaning different things is a trap nobody catches by reading
    # either one.
    assert overround(book(margin=1.40)) == pytest.approx(0.40, rel=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_every_method_returns_a_distribution(method):
    # Including `additive`, which only gets there by clipping and renormalising.
    # A vector that does not sum to 1 is not a baseline at all, whatever it says
    # about the favourite.
    probabilities = implied_probabilities(book(), method=method)
    assert probabilities.sum() == pytest.approx(1.0)
    assert (probabilities >= 0).all()


def test_the_methods_disagree_about_the_tail_and_that_is_the_point():
    odds = book()
    tails = {}
    for method in METHODS:
        probabilities = np.sort(implied_probabilities(odds, method=method))[::-1]
        tails[method] = probabilities[len(probabilities) // 2:].sum()
    # Multiplicative keeps the bookmaker's shape and so leaves the most in the
    # tail; power reshapes it. If these ever converge, `compare_methods` has
    # stopped being the load-bearing read the module says it is.
    assert tails["multiplicative"] > tails["power"]
    assert tails["power"] > tails["additive"]


def test_additive_zeroes_most_of_the_field_and_reports_it():
    # The renormalisation makes the vector look fine; `n_zeroed` is the only
    # place the damage stays visible, so that is what is pinned.
    table = compare_methods(book()).set_index("method")
    assert table.loc["additive", "n_zeroed"] > 50
    assert table.loc["multiplicative", "n_zeroed"] == 0
    assert table.loc["power", "n_zeroed"] == 0


def test_compare_methods_carries_the_book_it_described():
    table = compare_methods(book(n_runners=120, margin=1.3))
    assert table.attrs["overround"] == pytest.approx(0.3, rel=1e-6)
    assert table.attrs["n_runners"] == 120


def test_a_partial_field_is_refused_rather_than_normalised():
    # Half a start list still sums over 1 on a real book, so this needs a book
    # thin enough to fall under it — which is exactly the state that would
    # otherwise hand the absent riders' probability to the present ones.
    with pytest.raises(MarketFormatError, match="partial"):
        implied_probabilities([4.0, 5.0, 6.0])


def test_prices_that_are_not_decimal_odds_are_refused():
    with pytest.raises(MarketFormatError, match="Decimal odds"):
        implied_probabilities([MIN_DECIMAL_ODDS, 3.0, 4.0, 10.0])


def test_an_unknown_method_names_the_ones_that_exist():
    with pytest.raises(MarketFormatError, match="Unknown method"):
        implied_probabilities(book(), method="shin")


def test_missing_prices_stay_missing_and_do_not_shift_the_rest():
    odds = book(n_runners=40)
    with_gap = odds.copy()
    with_gap[5] = np.nan
    probabilities = implied_probabilities(with_gap, method=DEFAULT_METHOD)
    assert np.isnan(probabilities[5])
    # The priced runners still form a distribution between them: an unpriced
    # rider is one the book did not quote, not one the book says cannot win.
    assert np.nansum(probabilities) == pytest.approx(1.0)


def test_worths_recover_the_ratio_between_two_riders():
    # Under Plackett-Luce a win probability is w_i / sum(w), so the worths are
    # the probabilities up to a scale — the ratio is the part that is real.
    odds = book(n_runners=30)
    probabilities = implied_probabilities(odds, method="multiplicative")
    worths = worths_from_market(odds, method="multiplicative")
    assert worths.mean() == pytest.approx(1.0)
    assert worths[0] / worths[1] == pytest.approx(probabilities[0] / probabilities[1])


def test_market_frame_refuses_a_field_that_is_not_one_field():
    riders = [f"r{i}" for i in range(10)]
    odds = book(n_runners=10, margin=1.25)
    frame = market_frame(riders, odds)
    assert list(frame["rider"]) == riders
    assert frame["probability"].sum() == pytest.approx(1.0)

    with pytest.raises(MarketFormatError, match="prices"):
        market_frame(riders[:9], odds)
    with pytest.raises(MarketFormatError, match="twice"):
        market_frame(["a"] * 10, odds)


def test_unquoted_riders_get_the_books_own_floor_not_a_zero():
    # A zero is not "unlikely", it is "cannot win", and it takes the log score
    # to minus infinity the first time one of the 140 unquoted riders wins a
    # stage. They get the longest quoted price instead, and the count comes
    # back so the extrapolation is never silent.
    from cycling.market import field_worths

    riders = [f"r{i}" for i in range(10)]
    worths, n_unpriced = field_worths(riders, riders[:4], [2.0, 3.0, 5.0, 8.0])
    assert n_unpriced == 6
    assert (worths > 0).all()
    assert worths.mean() == pytest.approx(1.0)
    # Every unquoted rider lands on the same floor, and it is the smallest thing
    # in the field — the longest price the book was willing to put up.
    assert worths[4:] == pytest.approx(np.full(6, worths[3]))
    assert worths[3] == worths.min()


def test_filling_the_unquoted_field_weakens_the_market_baseline():
    # Deliberately: handing 6 riders a probability each takes it away from the
    # favourite, so a model beating this has not yet beaten the book.
    from cycling.market import field_worths

    riders = [f"r{i}" for i in range(10)]
    filled, _ = field_worths(riders, riders[:4], [2.0, 3.0, 5.0, 8.0])
    quoted, _ = field_worths(riders[:4], riders[:4], [2.0, 3.0, 5.0, 8.0])
    assert filled[0] / filled.sum() < quoted[0] / quoted.sum()


def test_refusing_is_available_for_a_caller_who_would_rather_drop_the_race():
    from cycling.market import field_worths

    riders = [f"r{i}" for i in range(10)]
    with pytest.raises(MarketFormatError, match="unquoted"):
        field_worths(riders, riders[:4], [2.0, 3.0, 5.0, 8.0], unpriced="refuse")
