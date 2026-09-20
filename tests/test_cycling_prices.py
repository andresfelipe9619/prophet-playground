"""The cycling price contract — cycling/prices.py.

The tests that matter here are refusals, and they are refusals about *identity*
rather than about values. Every number in a mixed price frame is a valid
decimal price; what is wrong is that they came from two books, or describe two
races, and no range check can see either. That is the same shape as football's
opening/closing guard and cycling's own one-kind-per-frame rule, so these tests
are written against the same question: does the frame describe one market
quoted by one book?
"""

import numpy as np
import pandas as pd
import pytest

from cycling.prices import (
    PriceFormatError,
    load_books,
    market_slice,
    preprocess_prices,
)


def raw(n=8, race="tour-de-france", kind="gc", stage=None, book="bookA",
        margin=1.35, seed=0, riders=None):
    rng = np.random.default_rng(seed)
    worths = np.exp(rng.normal(0.0, 1.2, n))
    probabilities = worths / worths.sum()
    return pd.DataFrame({
        "Date": ["01/07/2024"] * n,
        "Race": [race] * n,
        "Kind": [kind] * n,
        "Stage": [stage] * n,
        "Rider": riders if riders is not None else [f"rider {i}" for i in range(n)],
        "Odds": 1.0 / (probabilities * margin),
        "Book": [book] * n,
    })


def test_a_clean_file_carries_its_book_and_its_market():
    # Recorded on the frame rather than re-derived later: a guard that has to
    # work out what a frame is will eventually work it out from a mixed one.
    prices = preprocess_prices(raw())
    assert prices.attrs["book"] == "bookA"
    assert prices.attrs["market"]["race"] == "tour-de-france"
    assert prices.attrs["overround"] == pytest.approx(1.35, rel=1e-6)
    assert prices.attrs["n_runners"] == 8


def test_two_books_in_one_frame_are_refused():
    # The characteristic mistake: both halves are valid prices, and the merged
    # field has more runners than the race had starters.
    mixed = pd.concat([raw(book="bookA"), raw(book="bookB", seed=1)], ignore_index=True)
    with pytest.raises(PriceFormatError, match="different fields at different margins"):
        preprocess_prices(mixed)


def test_two_markets_in_one_frame_are_refused():
    # A GC price and a stage price normalised against each other describe a
    # race nobody ran — cycling's one-kind-per-frame rule, one layer up.
    mixed = pd.concat([raw(kind="gc"), raw(kind="stage", stage=7, seed=1)], ignore_index=True)
    with pytest.raises(PriceFormatError, match="different markets"):
        preprocess_prices(mixed)


def test_one_rider_priced_twice_is_refused():
    doubled = raw(n=6, riders=["a", "b", "c", "d", "e", "a"])
    with pytest.raises(PriceFormatError, match="more than once"):
        preprocess_prices(doubled)


def test_a_partial_field_is_refused():
    # Under 1.0 means runners are missing, and normalising then hands their
    # probability to whoever is left.
    thin = raw(n=3)
    thin["Odds"] = [8.0, 9.0, 10.0]
    with pytest.raises(PriceFormatError, match="partial"):
        preprocess_prices(thin)


def test_unconverted_odds_are_refused():
    fractional = raw(n=4)
    fractional["Odds"] = [0.5, 2.0, 3.0, 4.0]
    with pytest.raises(PriceFormatError, match="Decimal odds"):
        preprocess_prices(fractional)


def test_an_unknown_kind_is_refused_rather_than_guessed():
    with pytest.raises(PriceFormatError, match="Unknown result kind"):
        preprocess_prices(raw(kind="prologue"))


def test_a_missing_column_names_the_contract():
    without = raw().drop(columns=["Book"])
    with pytest.raises(PriceFormatError, match="Missing column"):
        preprocess_prices(without)


def test_load_books_refuses_files_from_different_books(tmp_path):
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    raw(book="bookA").to_csv(first, index=False)
    raw(book="bookB", seed=1).to_csv(second, index=False)
    with pytest.raises(PriceFormatError, match="different books"):
        load_books([first, second])


def test_load_books_stacks_several_markets_from_one_book(tmp_path):
    first = tmp_path / "gc.csv"
    second = tmp_path / "stage7.csv"
    raw(kind="gc").to_csv(first, index=False)
    raw(kind="stage", stage=7, seed=1).to_csv(second, index=False)

    merged = load_books([first, second])
    # Several races from one book is an ordinary season; the market is no longer
    # single, so the frame says so rather than claiming one.
    assert merged.attrs["book"] == "bookA"
    assert merged.attrs["market"] is None
    assert merged.attrs["n_markets"] == 2

    one = market_slice(merged, "tour-de-france", "stage", stage=7)
    assert one.attrs["market"] == {"race": "tour-de-france", "kind": "stage", "stage": 7}
    assert one.attrs["overround"] == pytest.approx(1.35, rel=1e-6)


def test_market_slice_refuses_a_market_that_is_not_there(tmp_path):
    path = tmp_path / "gc.csv"
    raw().to_csv(path, index=False)
    with pytest.raises(PriceFormatError, match="No prices"):
        market_slice(load_books([path]), "giro", "gc")


def _book_for(results, n_quoted=40, margin=1.45, seed=0):
    """A book quoting part of each stage's start list, as a real one would."""
    rng = np.random.default_rng(seed)
    frames = []
    for (race, kind, stage), group in results.groupby(["race", "kind", "stage"], dropna=False):
        riders = list(group["rider"])[:n_quoted]
        worths = np.exp(rng.normal(0.0, 1.4, len(riders)))
        probabilities = worths / worths.sum()
        frames.append(preprocess_prices(pd.DataFrame({
            "Date": group["ds"].min().strftime("%d/%m/%Y"),
            "Race": race, "Kind": kind, "Stage": stage, "Rider": riders,
            "Odds": 1.0 / (probabilities * margin), "Book": "bookA",
        })))
    merged = pd.concat(frames, ignore_index=True)
    merged.attrs["book"] = "bookA"
    return merged


def test_the_market_reads_as_a_forecaster_over_a_whole_start_list():
    from cycling.prices import market_forecaster
    from cycling.sample_data import load_sample_and_preprocess

    results = load_sample_and_preprocess(seed=0)
    forecast = market_forecaster(_book_for(results))

    stage = results[results["stage"] == 3]
    riders = list(stage["rider"])
    worths = forecast(results[results["ds"] < stage["ds"].min()], riders, stage["ds"].min())

    # Aligned to the start list the race actually had — 176 riders out of a book
    # that quoted 40 — with the rest on the book's own floor rather than a zero.
    assert len(worths) == len(riders)
    assert (worths > 0).all()


def test_a_race_with_no_price_is_not_scored_rather_than_guessed():
    from cycling.prices import market_forecaster
    from cycling.sample_data import load_sample_and_preprocess

    results = load_sample_and_preprocess(seed=0)
    prices = _book_for(results[results["stage"] <= 5])
    forecast = market_forecaster(prices)

    later = results[results["stage"] == 12]
    assert forecast(results, list(later["rider"]), later["ds"].min()) is None


def test_a_forecast_that_is_the_market_scores_exactly_zero_against_it():
    # The endpoint every domain in this project pins: football's weight-0 blend,
    # cycling's forecast that *is* the ranking, and now the market baseline
    # itself. If this drifts, nothing read off a market comparison means
    # anything, because the zero is what the effect is measured from.
    from cycling.evaluation import compare_forecasters
    from cycling.prices import market_forecaster
    from cycling.sample_data import load_sample_and_preprocess

    results = load_sample_and_preprocess(seed=0)
    forecast = market_forecaster(_book_for(results))

    table, _ = compare_forecasters(
        results, {"market": forecast, "same book": forecast}, baseline="market")
    row = table.iloc[0]
    assert row["effect"] == pytest.approx(0.0, abs=1e-12)
    assert not row["beats_baseline"]
    assert row["n_races"] > 0
